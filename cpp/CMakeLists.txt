cmake_minimum_required(VERSION 3.10)
project(Miacis)

# 共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.13.1)
find_package(Torch REQUIRED)

# 各ゲームで共通する部分のソース
file(GLOB SRCS ./*.cpp)

# 実行ファイルの定義
add_executable(nerf_practice ${SRCS})
