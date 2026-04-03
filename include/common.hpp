#pragma once
/**
 * common.hpp – Shared types, constants, and utilities.
 *
 * Memory discipline enforced here:
 *   - Default mem_limit_mb = 64 MB (safe on 2 GB available RAM)
 *   - All I/O uses fixed-size stack or small heap buffers
 *   - No STL containers that grow unboundedly without a cap
 */

#ifndef NO_MPI
#  include <mpi.h>
#endif

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

// ── Record layout ─────────────────────────────────────────────────────────────
// 100-byte fixed record compatible with Google/Terasort benchmark:
//   10-byte key (lexicographic) + 90-byte payload (opaque)
constexpr int KEY_SIZE     = 10;
constexpr int PAYLOAD_SIZE = 90;
constexpr int RECORD_SIZE  = KEY_SIZE + PAYLOAD_SIZE;   // 100 bytes

struct Record {
    char key[KEY_SIZE];
    char payload[PAYLOAD_SIZE];

    bool operator<(const Record& o)  const { return std::memcmp(key, o.key, KEY_SIZE) < 0; }
    bool operator==(const Record& o) const { return std::memcmp(key, o.key, KEY_SIZE) == 0; }
    bool operator<=(const Record& o) const { return !(o < *this); }
};
static_assert(sizeof(Record) == RECORD_SIZE, "Record size mismatch");

// ── MPI message tags ──────────────────────────────────────────────────────────
namespace Tag {
    constexpr int SAMPLE       = 100;
    constexpr int SPLITTER     = 101;
    constexpr int DATA         = 102;
    constexpr int DATA_SIZE    = 103;
    constexpr int MERGE_REQ    = 104;
    constexpr int MERGE_BATCH  = 105;
    constexpr int DONE         = 106;
    constexpr int HEARTBEAT    = 107;
    constexpr int HEARTBEAT_ACK= 108;
    constexpr int REASSIGN     = 109;
    constexpr int CHUNK_META   = 110;
}

// ── Configuration ─────────────────────────────────────────────────────────────
struct Config {
    std::string input_file;
    std::string output_file     = "sorted_output.bin";
    std::string tmp_dir         = "/tmp/dsort";

    // *** LOW-RAM DEFAULTS ***
    // On a system with 2 GB available, 64 MB per node leaves room for the OS,
    // MPI runtime, stack, and other processes.
    size_t mem_limit_mb         = 64;     // per-node RAM budget (MB)

    // Keep sample count small – 200 per node is plenty for splitter accuracy
    int    samples_per_node     = 200;

    // Merge batch: 512 records = 51 200 bytes ≈ 50 KB per worker stream
    // With 8 workers that's 400 KB peak coordinator buffer – very safe.
    int    merge_batch          = 512;

    int    heartbeat_sec        = 5;
    double fault_threshold      = 0.30;
    bool   stable_sort          = true;
    bool   verbose              = false;
};

// ── How many records fit in mem_limit_mb, with a safety headroom ──────────────
// Reserves 25% headroom for OS page cache, stack, etc.
inline size_t safe_records_per_mb(size_t mb, double headroom = 0.75) {
    return static_cast<size_t>(mb * 1024ULL * 1024ULL * headroom) / RECORD_SIZE;
}

// ── Timer ─────────────────────────────────────────────────────────────────────
class Timer {
    using Clk = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clk> t0_;
    std::string label_;
    bool        verbose_;
public:
    explicit Timer(std::string l, bool v = true)
        : t0_(Clk::now()), label_(std::move(l)), verbose_(v) {
        if (verbose_) std::cout << "[Timer] " << label_ << " started\n" << std::flush;
    }
    double elapsed_s() const {
        return std::chrono::duration<double>(Clk::now() - t0_).count();
    }
    void report() const {
        if (verbose_)
            std::cout << "[Timer] " << label_
                      << " done in " << elapsed_s() << " s\n" << std::flush;
    }
};

// ── Logging ───────────────────────────────────────────────────────────────────
inline void log(int rank, const std::string& msg) {
    std::cout << "[Rank " << rank << "] " << msg << "\n" << std::flush;
}
inline void log_err(int rank, const std::string& msg) {
    std::cerr << "[Rank " << rank << " ERR] " << msg << "\n" << std::flush;
}

// ── File helpers ──────────────────────────────────────────────────────────────
inline size_t file_record_count(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    auto sz = f.tellg();
    return static_cast<size_t>(sz < std::streampos(0) ? std::streampos(0) : sz) / RECORD_SIZE;
}

// Write a slice of records — never allocates extra memory
inline bool write_records(std::ofstream& f, const Record* data, size_t count) {
    return static_cast<bool>(
        f.write(reinterpret_cast<const char*>(data), count * RECORD_SIZE));
}
inline bool write_records(std::ofstream& f, const std::vector<Record>& v) {
    return v.empty() ? true : write_records(f, v.data(), v.size());
}

// Read exactly [count] records into a pre-sized vector; returns actually read
inline size_t read_records(std::ifstream& f, std::vector<Record>& buf, size_t count) {
    if (count == 0) return 0;
    buf.resize(count);
    f.read(reinterpret_cast<char*>(buf.data()), count * RECORD_SIZE);
    size_t got = static_cast<size_t>(f.gcount()) / RECORD_SIZE;
    buf.resize(got);
    return got;
}

// ── Memory-safe batch size helper ────────────────────────────────────────────
// Returns a batch size (in records) that fits within [mb] megabytes, capped
// to avoid individual allocations larger than [hard_cap_mb].
inline size_t batch_records(size_t mb, size_t hard_cap_mb = 128) {
    size_t capped = std::min(mb, hard_cap_mb);
    return safe_records_per_mb(capped);
}
