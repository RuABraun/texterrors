# For this to work there needs to be a symlink to pybind11 in the libs directory!
cmake_minimum_required(VERSION 3.9.5)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "-O3 ")

project(condutor)

include_directories("libs/")

add_subdirectory(libs/pybind11)
pybind11_add_module(texterrors_align libs/stringvector.cc libs/texterrors_align.cc)

set_target_properties(texterrors_align PROPERTIES LIBRARY_OUTPUT_NAME "texterrors_align")
