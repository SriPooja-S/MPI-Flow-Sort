#pragma once
/**
 * verifier.hpp – Streaming correctness checker.
 * Reads in 64 KB pages – never loads the full file into RAM.
 */

#include "common.hpp"
#include <random>

class Verifier {
public:
    struct Result {
        bool   ok = false; size_t total_checked = 0, violations = 0;
        std::string message;
    };

    static Result verify(const std::string& path,
                         size_t expected_count = 0, bool verbose = true) {
        Result res;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { res.message = "Cannot open: " + path; return res; }

        size_t file_bytes = static_cast<size_t>(f.tellg());
        if (file_bytes % RECORD_SIZE != 0) {
            res.message = "File size not a multiple of RECORD_SIZE";
            return res;
        }
        size_t total = file_bytes / RECORD_SIZE;
        if (expected_count > 0 && total != expected_count) {
            res.message = "Count mismatch: got " + std::to_string(total)
                        + " expected " + std::to_string(expected_count);
            return res;
        }
        if (total == 0) { res.ok = true; res.message = "Empty (trivially sorted)"; return res; }

        f.seekg(0, std::ios::beg);

        // Stream in 64 KB pages (640 records)
        constexpr size_t BUF = 640;
        std::vector<Record> buf(BUF);
        Record prev{};
        bool   first = true;
        size_t idx   = 0;

        while (idx < total) {
            size_t got = read_records(f, buf, std::min(BUF, total - idx));
            for (size_t i = 0; i < got; ++i) {
                if (!first && buf[i] < prev) {
                    ++res.violations;
                    if (verbose && res.violations <= 3)
                        std::cerr << "  Violation at record " << (idx+i) << "\n";
                }
                prev = buf[i];
                first = false;
            }
            idx += got;
            if (got == 0) break;
        }
        res.total_checked = total;
        res.ok            = (res.violations == 0);
        res.message       = res.ok
            ? ("OK — " + std::to_string(total) + " records correctly sorted")
            : ("FAILED — " + std::to_string(res.violations) + " violations");
        return res;
    }

    // Sample-based quick check (useful for very large files)
    static bool quick_check(const std::string& path, int samples = 10000) {
        size_t total = file_record_count(path);
        if (total < 2) return true;
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::mt19937_64 rng(42);
        Record a, b;
        for (int i = 0; i < samples; ++i) {
            size_t pos = rng() % (total - 1);
            f.seekg(static_cast<std::streamoff>(pos * RECORD_SIZE));
            f.read(reinterpret_cast<char*>(&a), RECORD_SIZE);
            f.read(reinterpret_cast<char*>(&b), RECORD_SIZE);
            if (b < a) return false;
        }
        return true;
    }
};
