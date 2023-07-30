cmake_minimum_required(VERSION 3.10)
project(nerf_practice)

# コンパイル設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH /home/sakoda/work/f2-nerf/External/libtorch)
find_package(Torch REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# YAML
include_directories(/home/sakoda/work/f2-nerf/External/yaml-cpp/include)
link_libraries(yaml-cpp)

# ソース
file(GLOB SRCS ./*.cpp)
list(APPEND SRCS ./dim-slam-nerf/voxel_based_radiance_field.cpp)

# 実行ファイルの定義
add_executable(train ./main/train.cpp ${SRCS})
add_executable(infer ./main/infer.cpp ${SRCS})
add_executable(test ./main/test.cpp ${SRCS})
add_executable(test_dim-slam-nerf ./main/test_dim-slam-nerf.cpp ${SRCS})

# OpenCVの設定
target_include_directories(train PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} /usr/include/eigen3)
target_include_directories(infer PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} /usr/include/eigen3)
target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} /usr/include/eigen3)
target_include_directories(test_dim-slam-nerf PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} /usr/include/eigen3)
target_link_libraries(train PUBLIC ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(infer PUBLIC ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(test PUBLIC ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(test_dim-slam-nerf PUBLIC ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
