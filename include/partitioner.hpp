#pragma once
/**
 * partitioner.hpp – Splitter-based range partitioning.
 *
 * LOW-RAM DESIGN:
 *   • Sampling uses reservoir sampling over the on-disk file – O(S) RAM, not O(N).
 *   • Bucket exchange: records are read from disk in pages and streamed to peers.
 *     Each page is at most PAGE_MB megabytes, flushed immediately after sending.
 *   • Received data is written straight to a temporary file – not accumulated in RAM.
 *   • No large vectors are held across phases.
 */

#include "common.hpp"
#include <random>
#include <algorithm>
#include <set>

// How many MB to use for each page when streaming bucket data
constexpr size_t PAGE_MB = 8;   // 8 MB pages ≈ 80 000 records per send/recv

class Partitioner {
public:
    Partitioner(int rank, int world_size, const Config& cfg)
        : rank_(rank), world_size_(world_size), cfg_(cfg) {}

    // ── Coordinator: gather samples → compute splitters → broadcast ──────────
    std::vector<Record> coordinator_compute_splitters(int num_workers) {
        int total_samples = num_workers * cfg_.samples_per_node;
        // Reuse a single fixed-size buffer – no unbounded growth
        std::vector<Record> all_samples(total_samples);

#ifndef NO_MPI
        for (int w = 1; w <= num_workers; ++w) {
            MPI_Recv(
                reinterpret_cast<char*>(
                    all_samples.data() + (w-1) * cfg_.samples_per_node),
                cfg_.samples_per_node * RECORD_SIZE,
                MPI_BYTE, w, Tag::SAMPLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#endif

        std::sort(all_samples.begin(), all_samples.end());

        std::vector<Record> splitters;
        splitters.reserve(num_workers - 1);
        for (int i = 1; i < num_workers; ++i) {
            int idx = (i * total_samples) / num_workers;
            splitters.push_back(all_samples[idx]);
        }

        // Free sample buffer immediately
        all_samples.clear();
        all_samples.shrink_to_fit();

#ifndef NO_MPI
        int nsplit = static_cast<int>(splitters.size());
        MPI_Bcast(&nsplit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (nsplit > 0)
            MPI_Bcast(reinterpret_cast<char*>(splitters.data()),
                      nsplit * RECORD_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
        if (cfg_.verbose)
            log(rank_, "Splitters computed: " + std::to_string(splitters.size()));
        return splitters;
    }

    // ── Worker: sample from on-disk file (reservoir – O(S) RAM) ─────────────
    // Reads the file in a single pass without loading it into memory.
    std::vector<Record> sample_from_file(const std::string& path) {
        size_t total = file_record_count(path);
        int    S     = cfg_.samples_per_node;
        std::vector<Record> samp;
        samp.reserve(S);

        std::mt19937_64 rng(static_cast<uint64_t>(rank_) * 0xdeadbeefULL + 1);
        std::ifstream f(path, std::ios::binary);
        if (!f) return samp;

        Record r;
        for (size_t i = 0; i < total; ++i) {
            if (!f.read(reinterpret_cast<char*>(&r), RECORD_SIZE)) break;
            if ((int)samp.size() < S) {
                samp.push_back(r);
            } else {
                size_t j = rng() % (i + 1);
                if ((int)j < S) samp[j] = r;
            }
        }
        // Pad to exactly S if file was smaller
        while ((int)samp.size() < S)
            samp.push_back(samp.empty() ? Record{} : samp.back());

        return samp;
    }

    // ── Worker: send samples, receive splitters ───────────────────────────────
    std::vector<Record> worker_exchange_splitters(
        const std::vector<Record>& samples, int coordinator = 0)
    {
#ifndef NO_MPI
        MPI_Send(reinterpret_cast<const char*>(samples.data()),
                 cfg_.samples_per_node * RECORD_SIZE,
                 MPI_BYTE, coordinator, Tag::SAMPLE, MPI_COMM_WORLD);

        int nsplit = 0;
        MPI_Bcast(&nsplit, 1, MPI_INT, coordinator, MPI_COMM_WORLD);
        std::vector<Record> splitters(nsplit);
        if (nsplit > 0)
            MPI_Bcast(reinterpret_cast<char*>(splitters.data()),
                      nsplit * RECORD_SIZE, MPI_BYTE, coordinator, MPI_COMM_WORLD);
        return splitters;
#else
        return {};
#endif
    }

    // ── Worker: streaming bucket exchange ────────────────────────────────────
    // Reads sorted_file in pages, sends each record to its destination worker.
    // Simultaneously receives pages from other workers and writes to recv_file.
    // Peak RAM: one send-page (PAGE_MB) + one recv-page (PAGE_MB) per active peer.
    //
    // Simple but correct approach: each worker sends all its outgoing data first
    // (non-blocking), then receives all incoming data.
    // For large N, a round-robin overlap would be better, but this is safe for
    // typical node counts (≤32).
    void streaming_exchange(const std::string& sorted_file,
                            const std::vector<Record>& splitters,
                            int num_workers,
                            int my_idx,         // 0-based among workers
                            const std::string& output_file)
    {
#ifndef NO_MPI
        size_t page_cap = batch_records(PAGE_MB);  // records per send page

        // --- Count how many records go to each bucket (one sequential scan) ---
        std::vector<size_t> bucket_counts(num_workers, 0);
        {
            std::ifstream scan(sorted_file, std::ios::binary);
            Record r;
            while (scan.read(reinterpret_cast<char*>(&r), RECORD_SIZE))
                ++bucket_counts[assign_bucket(r, splitters, num_workers)];
        }

        // Tell each peer how many records it will receive from us
        std::vector<MPI_Request> count_reqs(num_workers, MPI_REQUEST_NULL);
        std::vector<long long> send_counts(num_workers);
        for (int w = 0; w < num_workers; ++w) {
            send_counts[w] = static_cast<long long>(bucket_counts[w]);
            if (w == my_idx) continue;
            MPI_Isend(&send_counts[w], 1, MPI_LONG_LONG,
                      w + 1, Tag::DATA_SIZE, MPI_COMM_WORLD, &count_reqs[w]);
        }

        // Receive expected counts from each peer
        std::vector<size_t> recv_counts(num_workers, 0);
        for (int w = 0; w < num_workers; ++w) {
            if (w == my_idx) { recv_counts[w] = bucket_counts[w]; continue; }
            long long rc = 0;
            MPI_Recv(&rc, 1, MPI_LONG_LONG,
                     w + 1, Tag::DATA_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_counts[w] = static_cast<size_t>(rc);
        }
        MPI_Waitall(num_workers, count_reqs.data(), MPI_STATUSES_IGNORE);

        // --- Stream data: second scan of sorted_file, send page by page -------
        // Open one file stream per output peer (in page-sized chunks)
        // We send bucket data in a second sequential pass to avoid O(N) buffers.
        // Approach: sort records into per-bucket temp files, then send each.

        std::string bkt_dir = cfg_.tmp_dir + "/rank_" + std::to_string(rank_+1)
                            + "/buckets";
        fs::create_directories(bkt_dir);

        // Write each bucket to its own temp file (second scan)
        std::vector<std::string> bkt_files(num_workers);
        std::vector<std::ofstream> bkt_ofs(num_workers);
        for (int w = 0; w < num_workers; ++w) {
            if (w == my_idx || bucket_counts[w] == 0) continue;
            bkt_files[w] = bkt_dir + "/b" + std::to_string(w) + ".bin";
            bkt_ofs[w].open(bkt_files[w], std::ios::binary);
        }
        {
            std::ifstream scan2(sorted_file, std::ios::binary);
            Record r;
            while (scan2.read(reinterpret_cast<char*>(&r), RECORD_SIZE)) {
                int b = assign_bucket(r, splitters, num_workers);
                if (b == my_idx) continue;   // handled separately below
                if (bkt_ofs[b].is_open())
                    bkt_ofs[b].write(reinterpret_cast<const char*>(&r), RECORD_SIZE);
            }
        }
        for (auto& of : bkt_ofs) if (of.is_open()) of.close();

        // Non-blocking sends: stream each bucket file page by page
        std::vector<Record> page_buf(page_cap);
        for (int w = 0; w < num_workers; ++w) {
            if (w == my_idx || bucket_counts[w] == 0) continue;
            std::ifstream bf(bkt_files[w], std::ios::binary);
            size_t left = bucket_counts[w];
            while (left > 0) {
                size_t chunk = std::min(left, page_cap);
                bf.read(reinterpret_cast<char*>(page_buf.data()), chunk * RECORD_SIZE);
                size_t got = static_cast<size_t>(bf.gcount()) / RECORD_SIZE;
                if (got == 0) break;
                MPI_Send(page_buf.data(), static_cast<int>(got * RECORD_SIZE),
                         MPI_BYTE, w + 1, Tag::DATA, MPI_COMM_WORLD);
                left -= got;
            }
            // Delete bucket file immediately after sending
            bf.close();
            fs::remove(bkt_files[w]);
        }

        // --- Receive data from all peers, write directly to output_file -------
        // Also copy own bucket from sorted_file
        std::ofstream out(output_file, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write: " + output_file);

        // Own data: copy from sorted_file where bucket == my_idx
        {
            std::ifstream sf(sorted_file, std::ios::binary);
            Record r;
            while (sf.read(reinterpret_cast<char*>(&r), RECORD_SIZE)) {
                if (assign_bucket(r, splitters, num_workers) == my_idx)
                    out.write(reinterpret_cast<const char*>(&r), RECORD_SIZE);
            }
        }

        // Receive from each peer
        for (int w = 0; w < num_workers; ++w) {
            if (w == my_idx) continue;
            size_t left = recv_counts[w];
            while (left > 0) {
                size_t chunk = std::min(left, page_cap);
                MPI_Status st;
                MPI_Recv(page_buf.data(),
                         static_cast<int>(chunk * RECORD_SIZE),
                         MPI_BYTE, w + 1, Tag::DATA,
                         MPI_COMM_WORLD, &st);
                int bytes;
                MPI_Get_count(&st, MPI_BYTE, &bytes);
                size_t got = static_cast<size_t>(bytes) / RECORD_SIZE;
                out.write(reinterpret_cast<const char*>(page_buf.data()),
                          got * RECORD_SIZE);
                left -= got;
            }
        }

        // Clean up bucket dir
        std::error_code ec;
        fs::remove_all(bkt_dir, ec);
#endif
    }

    // ── Chunk distribution ────────────────────────────────────────────────────
    static void distribute_chunks(const std::string& input_path,
                                  int num_workers,
                                  std::vector<size_t>& offsets,
                                  std::vector<size_t>& counts) {
        size_t total = file_record_count(input_path);
        offsets.resize(num_workers);
        counts.resize(num_workers);
        size_t base = total / num_workers, extra = total % num_workers, off = 0;
        for (int w = 0; w < num_workers; ++w) {
            offsets[w] = off;
            counts[w]  = base + (static_cast<size_t>(w) < extra ? 1 : 0);
            off += counts[w];
        }
    }

private:
    int    rank_, world_size_;
    Config cfg_;

    int assign_bucket(const Record& r,
                      const std::vector<Record>& splitters,
                      int num_workers) const {
        int lo = 0, hi = static_cast<int>(splitters.size());
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (r <= splitters[mid]) hi = mid; else lo = mid + 1;
        }
        return std::min(lo, num_workers - 1);
    }
};
