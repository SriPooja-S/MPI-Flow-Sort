cmake_minimum_required(VERSION 3.14)
project(DistributedSort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build options ─────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit test suite" ON)
option(ASAN        "Enable AddressSanitizer" OFF)

# ── Find MPI ──────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

# ── Compiler flags ────────────────────────────────────────────────────────────
set(BASE_FLAGS "-Wall -Wextra -O2")
if(ASAN)
    set(BASE_FLAGS "${BASE_FLAGS} -fsanitize=address,undefined -g")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS}")

# ── Main distributed_sort binary ──────────────────────────────────────────────
add_executable(distributed_sort src/distributed_sort.cpp)
target_include_directories(distributed_sort PRIVATE include ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(distributed_sort PRIVATE MPI::MPI_CXX stdc++fs)

# ── Unit test binary (no MPI needed) ─────────────────────────────────────────
if(BUILD_TESTS)
    add_executable(run_tests tests/test_suite.cpp)
    target_include_directories(run_tests PRIVATE include)
    target_link_libraries(run_tests PRIVATE stdc++fs)

    enable_testing()
    add_test(NAME UnitTests COMMAND run_tests)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS distributed_sort DESTINATION bin)
install(PROGRAMS scripts/generate_data.py scripts/benchmark.py DESTINATION bin)
