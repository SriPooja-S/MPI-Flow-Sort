#pragma once
/**
 * fault_tolerance.hpp – Heartbeat-based failure detection and chunk reassignment.
 *
 * Memory: negligible – only a few integers per worker tracked by coordinator.
 */

#include "common.hpp"
#include <set>
#include <map>
#include <chrono>
#include <thread>
#include <atomic>

// ── Worker-side heartbeat sender ──────────────────────────────────────────────
class HeartbeatSender {
public:
    HeartbeatSender(int rank, int coordinator, int interval_sec)
        : rank_(rank), coord_(coordinator), interval_(interval_sec), running_(false) {}

    void start() { running_ = true; thread_ = std::thread([this]{ loop(); }); }
    void stop()  { running_ = false; if (thread_.joinable()) thread_.join(); }
    ~HeartbeatSender() { stop(); }

private:
    int rank_, coord_, interval_;
    std::atomic<bool> running_;
    std::thread thread_;

    void loop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(interval_));
            if (!running_) break;
#ifndef NO_MPI
            MPI_Request req;
            int dummy = rank_;
            MPI_Isend(&dummy, 1, MPI_INT, coord_,
                      Tag::HEARTBEAT, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
#endif
        }
    }
};

// ── Coordinator-side failure detector ─────────────────────────────────────────
class FailureDetector {
    using Clock     = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
public:
    FailureDetector(int num_workers, int timeout_sec)
        : timeout_sec_(timeout_sec) {
        auto now = Clock::now();
        for (int w = 1; w <= num_workers; ++w) {
            last_seen_[w] = now;
            alive_.insert(w);
        }
    }

    void poll() {
#ifndef NO_MPI
        MPI_Status status; int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, Tag::HEARTBEAT, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE,
                     Tag::HEARTBEAT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            last_seen_[status.MPI_SOURCE] = Clock::now();
        }
#endif
    }

    std::set<int> failed_workers() {
        auto now = Clock::now();
        std::set<int> failed;
        for (auto& [rank, t] : last_seen_) {
            if (!alive_.count(rank)) continue;
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(now-t).count();
            if (secs > timeout_sec_ * 2) failed.insert(rank);
        }
        return failed;
    }

    void mark_failed(int rank) { alive_.erase(rank); dead_.insert(rank); }
    bool is_alive(int rank) const { return alive_.count(rank) > 0; }
    const std::set<int>& alive_set() const { return alive_; }

private:
    int  timeout_sec_;
    std::map<int, TimePoint> last_seen_;
    std::set<int> alive_, dead_;
};

// ── Chunk metadata ────────────────────────────────────────────────────────────
struct ChunkMeta { size_t offset, count; int original_owner; };
