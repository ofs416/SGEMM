cmake_minimum_required(VERSION 3.20)
project(sgemm CXX)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-march=native -Wall -Wextra)

include(FetchContent)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2)
FetchContent_Declare(benchmark  GIT_REPOSITORY https://github.com/google/benchmark.git  GIT_TAG v1.9.1)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googletest benchmark)

add_executable(main main.cpp src/sgemm.cpp)
target_include_directories(main PRIVATE include)

add_executable(bench_sgemm bench/bench_sgemm.cpp src/sgemm.cpp)
target_include_directories(bench_sgemm PRIVATE include)
target_link_libraries(bench_sgemm PRIVATE benchmark::benchmark)

# OpenBLAS as the reference SGEMM to benchmark against
find_library(OPENBLAS_LIB openblas REQUIRED)
find_path(OPENBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas openblas-pthread REQUIRED)
target_include_directories(bench_sgemm PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(bench_sgemm PRIVATE ${OPENBLAS_LIB})
