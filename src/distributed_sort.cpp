/**
 * distributed_sort.cpp – Main entry point.
 *
 * LOW-RAM DESIGN SUMMARY:
 *   • Default mem_limit_mb = 64 MB (safe on systems with 2 GB available RAM).
 *   • Chunk reading is paged: never reads more than mem_limit_mb at once.
 *   • Bucket exchange is fully streaming – no full-partition in-memory copy.
 *   • Global merge keeps at most merge_batch records per worker stream.
 *   • All temporary files are deleted immediately when no longer needed.
 *   • The system refuses to start if available free space < 2 × input file size
 *     (prevents the OSError: [Errno 28] No space left on device crash).
 *
 * Compile:
 *   mpic++ -std=c++17 -O2 -I include -o distributed_sort src/distributed_sort.cpp -lstdc++fs
 *
 * Run (4 processes = coordinator + 3 workers):
 *   mpirun -np 4 ./distributed_sort -i data.bin -o sorted.bin -m 64 -v --verify
 *
 * On a machine with only 2 GB free RAM, use:
 *   mpirun -np 2 ./distributed_sort -i data.bin -o sorted.bin -m 64
 *
 * Options:
 *   -i <path>    Input file (required)
 *   -o <path>    Output file (default: sorted_output.bin)
 *   -m <MB>      Per-node RAM budget in MB (default: 64, minimum: 16)
 *   -t <dir>     Temp directory (default: /tmp/dsort)
 *   -s <N>       Samples per node for splitter (default: 200)
 *   -b <N>       Records per merge batch (default: 512)
 *   -v           Verbose logging
 *   --verify     Verify output after sort
 *   --check-space Skip disk-space pre-check
 */

#include "../include/common.hpp"
#include "../include/external_sort.hpp"
#include "../include/partitioner.hpp"
#include "../include/fault_tolerance.hpp"
#include "../include/global_merge.hpp"
#include "../include/verifier.hpp"

#include <filesystem>
namespace fs = std::filesystem;

// ── Argument parsing ──────────────────────────────────────────────────────────
static bool s_do_verify   = false;
static bool s_skip_check  = false;

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a=="-i" && i+1<argc) cfg.input_file       = argv[++i];
        else if (a=="-o" && i+1<argc) cfg.output_file      = argv[++i];
        else if (a=="-m" && i+1<argc) cfg.mem_limit_mb     = std::stoul(argv[++i]);
        else if (a=="-t" && i+1<argc) cfg.tmp_dir          = argv[++i];
        else if (a=="-s" && i+1<argc) cfg.samples_per_node = std::stoi(argv[++i]);
        else if (a=="-b" && i+1<argc) cfg.merge_batch      = std::stoi(argv[++i]);
        else if (a=="-v")             cfg.verbose           = true;
        else if (a=="--verify")       s_do_verify           = true;
        else if (a=="--skip-check")   s_skip_check          = true;
    }
    // Enforce minimum to avoid zero-size buffers
    cfg.mem_limit_mb = std::max(cfg.mem_limit_mb, (size_t)16);
    return cfg;
}

// ── Broadcast config from rank 0 ──────────────────────────────────────────────
void broadcast_config(Config& cfg) {
    auto bcast_str = [&](std::string& s) {
        int len = static_cast<int>(s.size());
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        s.resize(len);
        MPI_Bcast(s.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    };
    bcast_str(cfg.input_file);
    bcast_str(cfg.output_file);
    bcast_str(cfg.tmp_dir);

    unsigned long long mem = cfg.mem_limit_mb;
    MPI_Bcast(&mem, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    cfg.mem_limit_mb = mem;

    MPI_Bcast(&cfg.samples_per_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cfg.merge_batch,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cfg.heartbeat_sec,    1, MPI_INT, 0, MPI_COMM_WORLD);

    int vb = cfg.verbose    ? 1:0; MPI_Bcast(&vb,1,MPI_INT,0,MPI_COMM_WORLD); cfg.verbose    = vb;
    int ss = cfg.stable_sort? 1:0; MPI_Bcast(&ss,1,MPI_INT,0,MPI_COMM_WORLD); cfg.stable_sort= ss;
}

// ── Disk space pre-check ──────────────────────────────────────────────────────
// Requires at least 2 × input size free on the temp dir's filesystem.
static bool check_disk_space(const std::string& input_file,
                              const std::string& tmp_dir,
                              bool verbose) {
    std::error_code ec;
    auto info = fs::space(tmp_dir, ec);
    if (ec) return true;  // can't check — proceed anyway

    auto input_bytes = static_cast<uintmax_t>(
        file_record_count(input_file) * RECORD_SIZE);
    uintmax_t needed = input_bytes * 2;

    if (info.available < needed) {
        log_err(0, "Disk space check FAILED on " + tmp_dir
            + ": need " + std::to_string(needed / (1024*1024)) + " MB, "
            + "available " + std::to_string(info.available / (1024*1024)) + " MB. "
            + "Use -t <dir> to point to a partition with more space, "
            + "or use --skip-check to bypass.");
        return false;
    }
    if (verbose)
        log(0, "Disk check OK: " + std::to_string(info.available/(1024*1024))
               + " MB available, need ~" + std::to_string(needed/(1024*1024)) + " MB");
    return true;
}

// ── Paged chunk copy: read input[offset..offset+count] → output file ──────────
// Reads in mem_limit_mb pages – never more than that in RAM at once.
static void copy_chunk_paged(const std::string& input_path,
                             size_t offset, size_t count,
                             const std::string& out_path,
                             const Config& cfg) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input: " + input_path);
    in.seekg(static_cast<std::streamoff>(offset * RECORD_SIZE));

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write chunk: " + out_path);

    size_t page_cap = safe_records_per_mb(cfg.mem_limit_mb, 0.75);
    std::vector<Record> buf(page_cap);
    size_t remaining = count;

    while (remaining > 0) {
        size_t rd  = std::min(remaining, page_cap);
        size_t got = read_records(in, buf, rd);
        if (got == 0) break;
        write_records(out, buf.data(), got);
        remaining -= got;
    }
}

// ── Single-node mode ──────────────────────────────────────────────────────────
void run_single_node(const Config& cfg) {
    log(0, "Single-node mode");
    size_t total = file_record_count(cfg.input_file);
    log(0, std::to_string(total) + " records, "
         + std::to_string(total * RECORD_SIZE / (1024*1024)) + " MB, "
         + "RAM budget: " + std::to_string(cfg.mem_limit_mb) + " MB");

    if (!s_skip_check && !check_disk_space(cfg.input_file, cfg.tmp_dir, cfg.verbose))
        MPI_Abort(MPI_COMM_WORLD, 1);

    std::string rank_tmp = cfg.tmp_dir + "/rank_0";
    fs::create_directories(rank_tmp);
    std::string tmp_out = rank_tmp + "/sorted_final.bin";

    Timer t("Sort", true);
    ExternalSorter sorter(0, cfg);
    sorter.sort_file(cfg.input_file, tmp_out);
    t.report();

    auto out_parent = fs::path(cfg.output_file).parent_path();
    if (!out_parent.empty()) fs::create_directories(out_parent);
    fs::rename(tmp_out, cfg.output_file);
    fs::remove_all(rank_tmp);

    log(0, "Output: " + cfg.output_file + " (" + std::to_string(total) + " records)");
}

// ── Coordinator ───────────────────────────────────────────────────────────────
void run_coordinator(const Config& cfg, int world_size) {
    int num_workers = world_size - 1;
    Timer total_t("Total", true);

    size_t total = file_record_count(cfg.input_file);
    log(0, "Input: " + cfg.input_file
         + " (" + std::to_string(total) + " records, "
         + std::to_string(total * RECORD_SIZE / (1024*1024)) + " MB), "
         + "workers=" + std::to_string(num_workers)
         + ", RAM/node=" + std::to_string(cfg.mem_limit_mb) + " MB");

    if (!s_skip_check && !check_disk_space(cfg.input_file, cfg.tmp_dir, cfg.verbose))
        MPI_Abort(MPI_COMM_WORLD, 1);

    // Phase 1: Distribute chunk boundaries (just metadata – no data copy)
    {
        std::vector<size_t> offsets, counts;
        Partitioner::distribute_chunks(cfg.input_file, num_workers, offsets, counts);
        for (int w = 0; w < num_workers; ++w) {
            long long meta[2] = {(long long)offsets[w], (long long)counts[w]};
            MPI_Send(meta, 2, MPI_LONG_LONG, w+1, Tag::CHUNK_META, MPI_COMM_WORLD);
        }
        if (cfg.verbose) log(0, "Chunk metadata sent");
    }

    // Phase 2: Collect samples, compute & broadcast splitters
    {
        Partitioner part(0, world_size, cfg);
        part.coordinator_compute_splitters(num_workers);
    }

    // Phase 3: Workers sort locally + exchange
    log(0, "Waiting for workers to finish local sort + exchange...");
    MPI_Barrier(MPI_COMM_WORLD);
    log(0, "All workers ready for merge");

    // Phase 4: Global K-way merge
    {
        std::vector<int> worker_ranks;
        for (int w = 1; w <= num_workers; ++w) worker_ranks.push_back(w);
        auto out_parent = fs::path(cfg.output_file).parent_path();
        if (!out_parent.empty()) fs::create_directories(out_parent);
        GlobalMerger merger(0, cfg);
        merger.coordinator_merge(worker_ranks, cfg.output_file);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_t.report();
    log(0, "Done → " + cfg.output_file);
}

// ── Worker ────────────────────────────────────────────────────────────────────
void run_worker(const Config& cfg, int rank, int world_size) {
    int num_workers = world_size - 1;
    int my_idx      = rank - 1;
    std::string rank_tmp = cfg.tmp_dir + "/rank_" + std::to_string(rank);
    fs::create_directories(rank_tmp);

    // Phase 1: Receive chunk metadata, copy chunk from input (paged)
    long long meta[2];
    MPI_Recv(meta, 2, MPI_LONG_LONG, 0, Tag::CHUNK_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    size_t offset = static_cast<size_t>(meta[0]);
    size_t count  = static_cast<size_t>(meta[1]);

    if (cfg.verbose)
        log(rank, "Chunk offset=" + std::to_string(offset)
                + " count=" + std::to_string(count));

    std::string chunk_file  = rank_tmp + "/chunk.bin";
    std::string sorted_file = rank_tmp + "/sorted.bin";

    // Paged copy – never more than mem_limit_mb in RAM
    copy_chunk_paged(cfg.input_file, offset, count, chunk_file, cfg);

    // Phase 2: Local external sort (strict RAM cap)
    {
        Timer t("LocalSort rank=" + std::to_string(rank), cfg.verbose);
        ExternalSorter sorter(rank, cfg);
        sorter.sort_file(chunk_file, sorted_file);
        t.report();
    }
    fs::remove(chunk_file);   // free disk space immediately

    // Phase 2b: Sample from sorted file (reservoir – O(S) RAM only)
    Partitioner part(rank, world_size, cfg);
    auto samples   = part.sample_from_file(sorted_file);
    auto splitters = part.worker_exchange_splitters(samples, 0);
    samples.clear(); samples.shrink_to_fit();

    // Phase 3: Streaming bucket exchange + write repartitioned file
    std::string repartitioned = rank_tmp + "/repartitioned.bin";
    {
        Timer t("Exchange rank=" + std::to_string(rank), cfg.verbose);
        part.streaming_exchange(sorted_file, splitters,
                                num_workers, my_idx, repartitioned);
        t.report();
    }
    fs::remove(sorted_file);

    // Phase 3b: Final local sort of received partition
    std::string final_sorted = rank_tmp + "/final_sorted.bin";
    {
        Timer t("FinalSort rank=" + std::to_string(rank), cfg.verbose);
        ExternalSorter sorter(rank, cfg);
        sorter.sort_file(repartitioned, final_sorted);
        t.report();
    }
    fs::remove(repartitioned);

    // Signal coordinator that local work is done
    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 4: Serve merge batches to coordinator
    {
        GlobalMerger merger(0, cfg);
        merger.worker_serve_batches(final_sorted, 0);
    }
    fs::remove(final_sorted);
    MPI_Barrier(MPI_COMM_WORLD);

    // Cleanup
    std::error_code ec;
    fs::remove_all(rank_tmp, ec);
    if (cfg.verbose) log(rank, "Done");
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Config cfg;
    if (rank == 0) cfg = parse_args(argc, argv);
    broadcast_config(cfg);

    if (cfg.input_file.empty()) {
        if (rank == 0) {
            std::cerr << "\nUsage: mpirun -np <N> ./distributed_sort -i <file> [options]\n\n"
                      << "  -i <file>   Input binary file (required)\n"
                      << "  -o <file>   Output file     (default: sorted_output.bin)\n"
                      << "  -m <MB>     RAM per node    (default: 64, min: 16)\n"
                      << "  -t <dir>    Temp directory  (default: /tmp/dsort)\n"
                      << "  -s <N>      Samples/node    (default: 200)\n"
                      << "  -b <N>      Merge batch     (default: 512)\n"
                      << "  -v          Verbose\n"
                      << "  --verify    Verify output\n"
                      << "  --skip-check  Skip disk space pre-check\n\n"
                      << "Low-RAM example (2 GB machine):\n"
                      << "  mpirun -np 2 ./distributed_sort -i data.bin -m 64\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (world_size == 1) {
        run_single_node(cfg);
    } else {
        if (rank == 0) run_coordinator(cfg, world_size);
        else           run_worker(cfg, rank, world_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0 && s_do_verify) {
        log(0, "Verifying output...");
        size_t expected = file_record_count(cfg.input_file);
        auto r = Verifier::verify(cfg.output_file, expected, cfg.verbose);
        if (r.ok) log(0, "✓ " + r.message);
        else      log_err(0, "✗ " + r.message);
    }

    MPI_Finalize();
    return 0;
}
