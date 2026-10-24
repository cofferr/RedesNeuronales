cmake_minimum_required(VERSION 3.16)
project(nn_eigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Use a system Eigen 3.4+ if available, otherwise download it.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found on the system, fetching 3.4.0")
  include(FetchContent)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(Eigen3)
endif()

add_executable(nn_eigen src/main.cpp)
target_link_libraries(nn_eigen PRIVATE Eigen3::Eigen)

# Default data directory (can be overridden with the first command-line argument).
get_filename_component(PROJECT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data" ABSOLUTE)
target_compile_definitions(nn_eigen PRIVATE PROJECT_DATA_DIR="${PROJECT_DATA_DIR}")
