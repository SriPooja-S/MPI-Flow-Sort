#pragma once
/**
 * external_sort.hpp – Local external K-way merge sort.
 *
 * LOW-RAM DESIGN:
 *   • Run buffer is strictly capped to (mem_limit_mb * 0.75) / RECORD_SIZE records.
 *   • During K-way merge only ONE record per run is live in RAM (the heap head).
 *   • Output is flushed in 4 KB pages – never accumulates a large write buffer.
 *   • Temporary run files are deleted immediately after each merge pass.
 *   • The sorter never holds two copies of data simultaneously.
 */

#include "common.hpp"
#include <queue>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

// ── Lazy run reader ───────────────────────────────────────────────────────────
// Holds exactly ONE record in memory at a time. Reads from disk on demand.
struct RunReader {
    std::ifstream file;
    Record        current{};
    bool          exhausted = true;
    size_t        remaining = 0;

    RunReader() = default;

    void open(const std::string& path, size_t record_count) {
        file.open(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open run: " + path);
        remaining = record_count;
        exhausted = false;
        advance();
    }

    void advance() {
        if (remaining == 0) { exhausted = true; return; }
        if (!file.read(reinterpret_cast<char*>(&current), RECORD_SIZE)) {
            exhausted = true;
        } else {
            --remaining;
        }
    }
};

struct RunHeapCmp {
    bool operator()(const RunReader* a, const RunReader* b) const {
        return b->current < a->current;   // min-heap on key
    }
};

// ── ExternalSorter ────────────────────────────────────────────────────────────
class ExternalSorter {
public:
    ExternalSorter(int rank, const Config& cfg)
        : rank_(rank), cfg_(cfg) {
        tmp_dir_ = cfg_.tmp_dir + "/rank_" + std::to_string(rank_);
        fs::create_directories(tmp_dir_);
    }

    ~ExternalSorter() { cleanup_runs(); }

    // ── Sort an on-disk file → on-disk sorted file ───────────────────────────
    // Peak RAM = one run buffer (≤ mem_limit_mb * 0.75) + one Record per run.
    size_t sort_file(const std::string& input_path,
                     const std::string& output_path) {
        size_t total = file_record_count(input_path);
        if (total == 0) {
            std::ofstream(output_path, std::ios::binary);   // touch
            return 0;
        }

        size_t run_cap = run_capacity();
        if (cfg_.verbose)
            log(rank_, "ExternalSort: " + std::to_string(total) + " records, "
                + "run_cap=" + std::to_string(run_cap)
                + ", mem=" + std::to_string(cfg_.mem_limit_mb) + " MB");

        // Phase 1: produce sorted run files
        produce_runs(input_path, total, run_cap);

        // Phase 2: K-way merge all runs → output
        if (run_files_.size() == 1) {
            fs::rename(run_files_[0], output_path);
            run_files_.clear();
        } else {
            kway_merge(output_path);
            cleanup_runs();
        }

        return total;
    }

    // ── Sort an in-memory vector (already loaded by caller) ──────────────────
    // Caller is responsible for ensuring the vector fits in their RAM budget.
    void sort_in_memory(std::vector<Record>& records) {
        if (cfg_.stable_sort)
            std::stable_sort(records.begin(), records.end());
        else
            std::sort(records.begin(), records.end());
    }

    void cleanup_runs() {
        for (auto& p : run_files_) {
            std::error_code ec;
            fs::remove(p, ec);
        }
        run_files_.clear();
    }

private:
    int         rank_;
    Config      cfg_;
    std::string tmp_dir_;
    std::vector<std::string> run_files_;

    // How many records fit in our RAM budget (75% of limit for safety)
    size_t run_capacity() const {
        return safe_records_per_mb(cfg_.mem_limit_mb, 0.75);
    }

    // ── Phase 1 ───────────────────────────────────────────────────────────────
    void produce_runs(const std::string& input_path,
                      size_t total, size_t run_cap) {
        std::ifstream in(input_path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + input_path);

        std::vector<Record> buf;
        buf.reserve(run_cap);   // allocate once, reuse per run

        size_t remaining = total;
        size_t run_idx   = 0;

        while (remaining > 0) {
            size_t to_read = std::min(remaining, run_cap);
            size_t got     = read_records(in, buf, to_read);
            if (got == 0) break;
            remaining -= got;

            if (cfg_.stable_sort)
                std::stable_sort(buf.begin(), buf.end());
            else
                std::sort(buf.begin(), buf.end());

            std::string rpath = tmp_dir_ + "/run_"
                              + std::to_string(run_idx++) + ".bin";
            {
                std::ofstream rf(rpath, std::ios::binary);
                if (!rf) throw std::runtime_error("Cannot write run: " + rpath);
                write_records(rf, buf);
            }   // close immediately – frees OS buffers

            run_files_.push_back(rpath);

            if (cfg_.verbose)
                log(rank_, "  run " + std::to_string(run_idx-1)
                         + ": " + std::to_string(got) + " records");
        }

        // Free the run buffer immediately
        buf.clear();
        buf.shrink_to_fit();
    }

    // ── Phase 2 – streaming K-way merge ───────────────────────────────────────
    // Peak RAM: K RunReader objects (each holds 1 Record) + 4 KB write buffer
    void kway_merge(const std::string& output_path) {
        size_t K = run_files_.size();

        // Open all run readers (each holds exactly 1 Record)
        std::vector<RunReader> readers(K);
        for (size_t i = 0; i < K; ++i) {
            size_t cnt = file_record_count(run_files_[i]);
            readers[i].open(run_files_[i], cnt);
        }

        // Min-heap over run heads
        std::priority_queue<RunReader*,
                            std::vector<RunReader*>,
                            RunHeapCmp> heap;
        for (auto& r : readers)
            if (!r.exhausted) heap.push(&r);

        std::ofstream out(output_path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write: " + output_path);

        // Tiny write buffer: 4096 records = 400 KB max, flush when full
        constexpr size_t WBUF = 4096;
        std::vector<Record> wbuf;
        wbuf.reserve(WBUF);

        while (!heap.empty()) {
            RunReader* top = heap.top(); heap.pop();
            wbuf.push_back(top->current);
            top->advance();
            if (!top->exhausted) heap.push(top);

            if (wbuf.size() >= WBUF) {
                write_records(out, wbuf);
                wbuf.clear();
            }
        }
        if (!wbuf.empty()) write_records(out, wbuf);

        // Close run files and delete them immediately
        for (auto& r : readers) r.file.close();
    }
};
