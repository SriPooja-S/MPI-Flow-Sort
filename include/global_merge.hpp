#pragma once
/**
 * global_merge.hpp – Coordinator-driven streaming K-way merge.
 *
 * LOW-RAM DESIGN:
 *   • Coordinator keeps exactly ONE batch (merge_batch records) per worker stream
 *     in RAM at any time.  With merge_batch=512 and 8 workers that is 8×512×100
 *     bytes = ~400 KB coordinator peak.
 *   • Workers read from their sorted file in merge_batch pages, never more.
 *   • Output is written to disk immediately; no accumulation.
 */

#include "common.hpp"
#include <queue>
#include <fstream>

// ── Per-worker stream (coordinator side) ──────────────────────────────────────
struct WorkerStream {
    int    rank     = 0;
    size_t pos      = 0;
    bool   done     = false;
    std::vector<Record> buffer;   // at most merge_batch records

    bool has_current() const { return !done && pos < buffer.size(); }
    const Record& current() const { return buffer[pos]; }
};
struct StreamCmp {
    bool operator()(const WorkerStream* a, const WorkerStream* b) const {
        return b->current() < a->current();
    }
};

// ── GlobalMerger ──────────────────────────────────────────────────────────────
class GlobalMerger {
public:
    GlobalMerger(int coord, const Config& cfg) : coord_(coord), cfg_(cfg) {}

    // ── Coordinator: merge N sorted worker streams → output file ─────────────
    void coordinator_merge(const std::vector<int>& worker_ranks,
                           const std::string& output_path) {
        Timer t("GlobalMerge", cfg_.verbose);

        std::ofstream out(output_path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open output: " + output_path);

        int N = static_cast<int>(worker_ranks.size());
        std::vector<WorkerStream> streams(N);
        for (int i = 0; i < N; ++i) streams[i].rank = worker_ranks[i];

        // Prime all streams
        for (auto& s : streams) request_batch(s);

        std::priority_queue<WorkerStream*,
                            std::vector<WorkerStream*>,
                            StreamCmp> heap;
        for (auto& s : streams)
            if (s.has_current()) heap.push(&s);

        // Tiny output write buffer (≤ 4096 records = 400 KB)
        constexpr size_t WBUF = 4096;
        std::vector<Record> wbuf;
        wbuf.reserve(WBUF);
        size_t total_written = 0;

        while (!heap.empty()) {
            WorkerStream* top = heap.top(); heap.pop();
            wbuf.push_back(top->current());
            ++top->pos;
            if (top->pos >= top->buffer.size() && !top->done)
                request_batch(*top);
            if (top->has_current()) heap.push(top);

            if (wbuf.size() >= WBUF) {
                write_records(out, wbuf);
                total_written += wbuf.size();
                wbuf.clear();
            }
        }
        if (!wbuf.empty()) {
            write_records(out, wbuf);
            total_written += wbuf.size();
        }

        if (cfg_.verbose)
            log(coord_, "GlobalMerge wrote " + std::to_string(total_written) + " records");
        t.report();
    }

    // ── Worker: serve batches on demand until sorted file exhausted ───────────
    void worker_serve_batches(const std::string& sorted_file, int coordinator) {
        std::ifstream in(sorted_file, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + sorted_file);

        std::vector<Record> buf(cfg_.merge_batch);

        while (true) {
#ifndef NO_MPI
            int req = 0;
            MPI_Recv(&req, 1, MPI_INT, coordinator,
                     Tag::MERGE_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
            size_t got = read_records(in, buf, cfg_.merge_batch);
            if (got == 0) {
#ifndef NO_MPI
                int done_flag = 0;
                MPI_Send(&done_flag, 1, MPI_INT, coordinator,
                         Tag::DONE, MPI_COMM_WORLD);
#endif
                break;
            }
#ifndef NO_MPI
            int n = static_cast<int>(got);
            MPI_Send(&n, 1, MPI_INT, coordinator, Tag::MERGE_BATCH, MPI_COMM_WORLD);
            MPI_Send(buf.data(), n * RECORD_SIZE,
                     MPI_BYTE, coordinator, Tag::MERGE_BATCH, MPI_COMM_WORLD);
#endif
        }
    }

private:
    int    coord_;
    Config cfg_;

    void request_batch(WorkerStream& s) {
#ifndef NO_MPI
        int req = 1;
        MPI_Send(&req, 1, MPI_INT, s.rank, Tag::MERGE_REQ, MPI_COMM_WORLD);

        MPI_Status status;
        MPI_Probe(s.rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == Tag::DONE) {
            int f; MPI_Recv(&f,1,MPI_INT,s.rank,Tag::DONE,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            s.done = true; s.buffer.clear(); s.pos = 0;
            return;
        }
        int n = 0;
        MPI_Recv(&n, 1, MPI_INT, s.rank, Tag::MERGE_BATCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        s.buffer.resize(n); s.pos = 0;
        if (n > 0)
            MPI_Recv(s.buffer.data(), n * RECORD_SIZE,
                     MPI_BYTE, s.rank, Tag::MERGE_BATCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            s.done = true;
#endif
    }
};
