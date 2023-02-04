cmake_minimum_required(VERSION 3.10)
project(nerf_practice)

# コンパイル設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.13.1)
find_package(Torch REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# ソース
file(GLOB SRCS ./*.cpp)

# 実行ファイルの定義
add_executable(nerf_practice ${SRCS})

# OpenCVの設定
target_include_directories(nerf_practice PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(nerf_practice PUBLIC ${OpenCV_LIBRARIES})
