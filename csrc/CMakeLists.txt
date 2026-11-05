# Native kernels of spectral-analyzer: the pybind11 module ``_spectral_kernels``
# (ray tracer, field solver, acoustics, rasterizer, document renderer,
# exposure backend, optical handlers, wave T4) plus the four standalone C-API
# DLLs the Python side loads through ctypes (``csrc_bin/*.dll``).
#
# Reconstructed 2026-11-05 from the sources and the retained build log; the
# original file was never tracked and left with the emptied checkout that
# held the last build.  The build log's targets are kept by name:
# _spectral_kernels, base_rasterizer_c, player_clip_c, router_step_c,
# tile_overlap_c.  Configure and build from the repository root:
#
#   cmake -S csrc -B csrc_build -A x64
#   cmake --build csrc_build --config Release --target _spectral_kernels -j 8
#
# The module lands at the repository root (``_spectral_kernels.cp311-win_amd64.pyd``),
# the DLLs in ``csrc_bin/``, as before.
cmake_minimum_required(VERSION 3.21)
project(spectral_kernels LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

get_filename_component(SPECTRAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Python and pybind11: the interpreter that will import the module, and the
# pybind11 package installed beside it (``python -m pybind11 --cmakedir``).
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_PROBE)
  if(NOT PYBIND11_PROBE EQUAL 0)
    message(FATAL_ERROR "pybind11 is not installed for ${Python3_EXECUTABLE}: pip install pybind11")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

# Eigen: header-only; the vendored fftfree build beside this repository
# already holds a fetched copy, else point EIGEN_INCLUDE_DIR at one.
set(EIGEN_INCLUDE_DIR "${SPECTRAL_ROOT}/../fftfree/build/_deps/eigen-src" CACHE PATH "Eigen headers")
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Core")
  message(FATAL_ERROR "Eigen not found at ${EIGEN_INCLUDE_DIR}; set EIGEN_INCLUDE_DIR")
endif()
# fftfree: header-only, vendored (wave_t4 includes plan_support.hpp).
set(FFTFREE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/fftfree")

find_package(OpenMP)

file(GLOB SPECTRAL_KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp")
list(FILTER SPECTRAL_KERNEL_SOURCES EXCLUDE REGEX "\\.old\\.cpp$")
set(SPECTRAL_MODULE_SOURCES
  bindings/pybind_kernels.cpp
  exposure_backend.cpp
  optical_handlers.cpp
  transforms/picard_step.cpp
  ${SPECTRAL_KERNEL_SOURCES})

function(spectral_common_settings target)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${FFTFREE_INCLUDE_DIR}"
    "${EIGEN_INCLUDE_DIR}")
  target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
  if(MSVC)
    target_compile_options(${target} PRIVATE /bigobj /EHsc /utf-8 $<$<CONFIG:Release>:/O2>)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

pybind11_add_module(_spectral_kernels MODULE ${SPECTRAL_MODULE_SOURCES})
spectral_common_settings(_spectral_kernels)
if(WIN32)
  # gl_compute resolves GL entry points through wglGetProcAddress.
  target_link_libraries(_spectral_kernels PRIVATE opengl32)
endif()
set_target_properties(_spectral_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${SPECTRAL_ROOT}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SPECTRAL_ROOT}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SPECTRAL_ROOT}"
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${SPECTRAL_ROOT}"
  RUNTIME_OUTPUT_DIRECTORY "${SPECTRAL_ROOT}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SPECTRAL_ROOT}")

# The standalone C-API DLLs (ctypes tier of the Python wrappers).
function(spectral_c_dll target output source)
  add_library(${target} SHARED ${source} ${ARGN})
  spectral_common_settings(${target})
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${output}"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${SPECTRAL_ROOT}/csrc_bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SPECTRAL_ROOT}/csrc_bin"
    LIBRARY_OUTPUT_DIRECTORY "${SPECTRAL_ROOT}/csrc_bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SPECTRAL_ROOT}/csrc_bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endfunction()
spectral_c_dll(base_rasterizer_c base_rasterizer kernels/base_rasterizer.cpp)
spectral_c_dll(player_clip_c player_clip kernels/player_clip.cpp)
spectral_c_dll(router_step_c router_step kernels/router_step.cpp kernels/field_march.cpp
  kernels/acoustic_pressure_backend.cpp transforms/picard_step.cpp)
target_compile_definitions(router_step_c PRIVATE BUILDING_ROUTER_STEP_DLL)
spectral_c_dll(tile_overlap_c tile_overlap kernels/tile_overlap.cpp)
