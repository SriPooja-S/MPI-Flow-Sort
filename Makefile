# Makefile – Distributed Sort v2 (low-RAM edition)
# Usage:
#   make              → build distributed_sort
#   make tests        → build & run unit tests
#   make run          → quick end-to-end test (1M records, np=1, -m 64)
#   make bench        → benchmark with np=1,2
#   make clean        → remove build artifacts and temp files

CXX       := mpic++
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -I include
LDFLAGS   := -lstdc++fs
TESTCXX   := g++
TESTFLAGS := -std=c++17 -O2 -Wall -DNO_MPI -I include

BIN     := distributed_sort
TESTBIN := run_tests
SRC     := src/distributed_sort.cpp
TESTSRC := tests/test_suite.cpp

.PHONY: all tests run bench clean

all: $(BIN)

$(BIN): $(SRC) include/*.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓  Built $(BIN)"

$(TESTBIN): $(TESTSRC) include/*.hpp
	$(TESTCXX) $(TESTFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓  Built $(TESTBIN)"

tests: $(TESTBIN)
	@echo "\nRunning unit tests..."
	./$(TESTBIN)

# Quick smoke test: 1M records, single-node, 64 MB RAM cap
NP ?= 1
run: $(BIN)
	@echo "Generating 1M records..."
	python3 scripts/generate_data.py --count 1000000 --output /tmp/test_in.bin
	@echo "Sorting (np=$(NP), -m 64)..."
	mpirun -np $(NP) ./$(BIN) -i /tmp/test_in.bin -o /tmp/test_out.bin -m 64 -v --verify
	@echo "Done."

# Multi-node benchmark (conservative for 2 GB RAM machine)
bench: $(BIN)
	python3 scripts/generate_data.py --count 2000000 --output /tmp/bench_in.bin
	python3 scripts/benchmark.py \
		--binary ./$(BIN) \
		--input  /tmp/bench_in.bin \
		--np-list 1,2 \
		--mem    64 \
		--verify

clean:
	rm -f $(BIN) $(TESTBIN)
	rm -rf /tmp/dsort /tmp/test_in.bin /tmp/test_out.bin /tmp/bench_in.bin /tmp/bench_out.bin
	@echo "✓  Cleaned"
