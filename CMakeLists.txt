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

# One source file per kernel; add new kernels here
add_library(sgemm STATIC src/sgemm_naive.cpp src/sgemm_blocked.cpp src/matmul.cpp)
target_include_directories(sgemm PUBLIC include)

add_executable(bench_sgemm bench/bench_sgemm.cpp)
target_link_libraries(bench_sgemm PRIVATE sgemm benchmark::benchmark)

enable_testing()
foreach(t test_matrix test_sgemm)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE sgemm)
  target_compile_options(${t} PRIVATE -UNDEBUG)  # tests use assert
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# OpenBLAS as the reference SGEMM to benchmark against
find_library(OPENBLAS_LIB openblas REQUIRED)
find_path(OPENBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas openblas-pthread REQUIRED)
target_include_directories(bench_sgemm PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(bench_sgemm PRIVATE ${OPENBLAS_LIB})
