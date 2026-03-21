cmake_minimum_required(VERSION 3.15...3.30)

project(texterrors VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(PYTHON_DEV_COMPONENT Development)
else()
  set(PYTHON_DEV_COMPONENT Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${PYTHON_DEV_COMPONENT} REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  texterrors_align
  NOMINSIZE
  libs/stringvector.cc
  libs/texterrors_align.cc
)

target_include_directories(texterrors_align PRIVATE libs)

install(TARGETS texterrors_align LIBRARY DESTINATION . RUNTIME DESTINATION .)
