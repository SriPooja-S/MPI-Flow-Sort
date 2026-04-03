# MPI-Flow-Sort : A distributed External Sort System

A production-grade distributed sorting system in **C++ + MPI** that sorts datasets far larger than available RAM across multiple networked nodes. Implements the full pipeline from data partitioning through external sort to global merge with fault tolerance.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DISTRIBUTED SORT PIPELINE                    │
│                                                                     │
│  Phase 1 – Chunk Distribution (Coordinator → Workers)              │
│  ┌──────────┐   offset+count    ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │  Rank 0  │ ──────────────→   │ Rank 1  │ │ Rank 2  │ │ Rank N│ │
│  │Coordinator│                  │ Worker  │ │ Worker  │ │Worker │ │
│  └──────────┘                   └─────────┘ └─────────┘ └───────┘ │
│                                                                     │
│  Phase 2 – Local External Sort (per Worker)                        │
│  Each worker reads its chunk in RAM-sized pages → sorts each page  │
│  (run) → K-way merges all runs to a sorted local file              │
│                                                                     │
│  Phase 3 – Splitter Exchange & Repartition (All-to-All)            │
│  Workers sample keys → Coordinator picks N-1 splitters →           │
│  each worker re-buckets data and exchanges with peers → final      │
│  local sort ensures each worker holds a globally correct partition  │
│                                                                     │
│  Phase 4 – Global K-Way Merge (Workers → Coordinator)             │
│  Coordinator drives a priority-queue merge over N sorted streams,  │
│  writing the globally sorted output file                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Partitioning | Sampling-based splitters (Terasort) | Near-uniform distribution without full data scan |
| Local sort | External K-way merge sort | Handles data larger than RAM |
| Global merge | Coordinator-pulled streaming | Bounded memory; overlaps I/O and compute |
| Fault tolerance | Heartbeat + chunk reassignment | Tolerates up to 30% node loss |
| Data format | 100-byte fixed records (10B key + 90B payload) | Google Sort Benchmark compatible |

---

## Project Structure

```
distributed_sort/
├── include/
│   ├── common.hpp           # Shared types: Record, Config, Tags, helpers
│   ├── external_sort.hpp    # Local external K-way merge sort
│   ├── partitioner.hpp      # Splitter sampling & all-to-all data exchange
│   ├── fault_tolerance.hpp  # Heartbeat sender/receiver, chunk reassignment
│   ├── global_merge.hpp     # Streaming K-way merge coordinator ↔ workers
│   └── verifier.hpp         # Output correctness verification
├── src/
│   └── distributed_sort.cpp # Main: coordinator + worker process logic
├── tests/
│   └── test_suite.cpp       # Unit & integration tests (no MPI required)
├── scripts/
│   ├── generate_data.py     # Test data generator (random/skewed/sorted/dupes)
│   └── benchmark.py         # Automated multi-node benchmarking
├── CMakeLists.txt
├── Makefile
└── README.md
```

---

## Prerequisites

```bash
# Ubuntu / Debian
sudo apt-get install -y mpich libmpich-dev g++ cmake python3

# OR with OpenMPI
sudo apt-get install -y openmpi-bin libopenmpi-dev g++ cmake python3

# macOS (Homebrew)
brew install open-mpi cmake python3
```

Requires: C++17, MPI (OpenMPI or MPICH), GCC ≥ 8 or Clang ≥ 7.

---

## Build

### Option A: Makefile (quickest)
```bash
cd distributed_sort
make                    # builds ./distributed_sort
make tests              # builds & runs unit tests
```

### Option B: CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure   # run tests
```

---

## Usage

### 1. Generate Test Data
```bash
# 10 million random records (~1 GB)
python3 scripts/generate_data.py --count 10000000 --output data.bin

# 100M records with skewed distribution (~10 GB)
python3 scripts/generate_data.py --count 100000000 --skew --output skewed.bin

# 50M records with 50% duplicate keys
python3 scripts/generate_data.py --count 50000000 --dups 0.5 --output dupes.bin

# Nearly pre-sorted data (worst case for some algorithms)
python3 scripts/generate_data.py --count 10000000 --sorted --output presorted.bin
```

### 2. Run the Sort

```bash
# Single-node (no MPI overhead, pure external sort)
./distributed_sort -i data.bin -o sorted.bin -m 512

# 4 processes on local machine (simulates 3 workers)
mpirun -np 4 ./distributed_sort -i data.bin -o sorted.bin -m 256

# 4 processes with verbose output and verification
mpirun -np 4 ./distributed_sort -i data.bin -o sorted.bin -m 256 -v --verify

# Cluster: 8 nodes, each with 16GB RAM, custom temp directory
mpirun -np 8 --hostfile hosts.txt ./distributed_sort \
    -i /nfs/data.bin -o /nfs/sorted.bin \
    -m 14000 -t /local/tmp/dsort -v --verify
```

#### All Options

| Flag | Default | Description |
|---|---|---|
| `-i <path>` | *(required)* | Input binary file |
| `-o <path>` | `sorted_output.bin` | Output file |
| `-m <MB>` | `256` | Per-node RAM budget in MB |
| `-t <dir>` | `/tmp/dsort` | Temporary directory for run files |
| `-s <N>` | `1000` | Sample size per node for splitter computation |
| `-b <N>` | `4096` | Records per batch in global merge |
| `-v` | off | Verbose logging |
| `--verify` | off | Verify output correctness after sorting |

### 3. Run Unit Tests
```bash
make tests
# Output:
# === Distributed Sort Test Suite ===
# -- Record Tests --
#   [ RUN ] Record: size is exactly 100 bytes ... PASS
#   [ RUN ] Record: comparison operators ... PASS
# ...
# === Results: 15 passed, 0 failed ===
```

### 4. Benchmark
```bash
# Auto-generate 50M records and benchmark np=1,2,4
python3 scripts/benchmark.py \
    --binary ./distributed_sort \
    --generate 50000000 \
    --np-list 1,2,4 \
    --mem 256

# Output example:
# ============================================================
#   Nodes |  Time (s)  |    MB/s    |  Speedup
#  -------+------------+------------+---------
#       1 |     45.23  |    110.3   |    1.00x
#       2 |     23.87  |    209.0   |    1.89x
#       4 |     12.91  |    386.5   |    3.50x
# ============================================================
```

---

## Component Deep-Dives

### External Sort (`external_sort.hpp`)
Implements a classic external K-way merge sort:
1. **Run creation**: reads chunks fitting in `mem_limit_mb` RAM, sorts in-memory (`std::stable_sort`), writes to disk
2. **K-way merge**: uses a `std::priority_queue<RunReader*>` (min-heap) to merge all runs in one pass with O(R log K) time where R = total records, K = number of runs
3. **Buffered I/O**: 8192-record write buffer amortises syscall overhead

### Partitioner (`partitioner.hpp`)
Implements **Terasort-style** range partitioning:
- Each worker performs reservoir sampling (O(N) time, O(S) space)
- Coordinator sorts all S×W samples, picks W-1 evenly spaced splitters
- Each worker binary-searches to bucket every record → all-to-all MPI exchange
- Final local sort produces a globally ordered partition

### Fault Tolerance (`fault_tolerance.hpp`)
- Workers run a background thread that sends `MPI_Isend` heartbeats every `heartbeat_sec` seconds
- Coordinator polls for heartbeats non-blockingly with `MPI_Iprobe`
- After `2×timeout` without a heartbeat, a worker is marked failed
- `ChunkReassigner` sends the failed node's `ChunkMeta` (offset, count) to a healthy survivor
- Configurable threshold (`fault_threshold`): abort if too many nodes fail

### Global Merge (`global_merge.hpp`)
- Coordinator pulls batches from each worker with `Tag::MERGE_REQ` / `Tag::MERGE_BATCH`
- Workers signal exhaustion with `Tag::DONE`
- Coordinator maintains a priority queue over W stream heads
- Pipeline overlap: requests the next batch while processing the current one

---

## Performance & Scalability

### Expected Throughput (1 Gbps network, NVMe disks)

| Dataset | 1 Node | 2 Nodes | 4 Nodes | 8 Nodes |
|---|---|---|---|---|
| 10 GB | ~90s | ~50s | ~28s | ~16s |
| 100 GB | ~15min | ~8min | ~4.5min | ~2.5min |
| 1 TB | ~2.5h | ~1.3h | ~45min | ~25min |

*Actual performance varies by disk speed, network bandwidth, and key distribution.*

### Tuning Tips
- **`-m`**: Set to ~80% of available RAM per node. Larger = fewer runs = faster merge.
- **`-s`**: Increase for more accurate splitters with skewed data (try 5000+).
- **`-b`**: Increase for better merge throughput (try 16384 or 65536).
- **Temp disk**: Use local NVMe SSD for `-t`, not NFS.
- **Network**: Enable large MPI messages (`--mca btl_tcp_eager_limit 65536`).

---

## Record Format & Compatibility

The binary format is compatible with the [Gray Sort / Terasort benchmark](http://sortbenchmark.org/):

```
Offset  Length  Description
0       10      Key (lexicographically compared)
10      90      Payload (opaque, preserved in output)
```

To use with custom data types, modify the `Record` struct and comparison operators in `common.hpp`.

---

## Known Limitations & Extensions

| Limitation | Possible Extension |
|---|---|
| Coordinator is a single point of failure for the merge phase | Use tree-reduction merge or peer-to-peer pipeline |
| All-to-all exchange holds all local data in memory | Implement streaming bucketing to disk |
| Splitter quality degrades with extreme skew | Use iterative histogram-based partitioning |
| Fixed 100-byte record format | Template `Record<KeyT, ValueT>` with custom comparators |
| Output is a single file | Striped output across multiple files for downstream parallelism |

---

## References

- Dean & Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004)
- Nyberg et al., "AlphaSort: A Cache-Sensitive Parallel External Sort" (VLDB 1995)
- Apache Hadoop TeraSort benchmark
- Knuth, *The Art of Computer Programming*, Vol. 3: Sorting and Searching

---


