cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pima-classifier)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pima-classifier src/pima-classifier.cpp)
target_link_libraries(pima-classifier "${TORCH_LIBRARIES}")
set_property(TARGET pima-classifier PROPERTY CXX_STANDARD 17)
