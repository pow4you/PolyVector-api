# Builds the two upstream command-line programs that polyvec wraps.
#
#   polyvec-field   "Vectorization of Line Drawings via PolyVector Fields"
#                   (Bessmeltsev & Solomon 2019)        sources: ../vendor/pv
#   polyvec-flow    "Keypoint-Driven Line Drawing Vectorization via PolyVector Flow"
#                   (Puhachov et al. 2021)              sources: ../vendor/pvflow
#
# The upstream sources are compiled exactly as vendored; nothing in ../vendor is
# patched.  This file only replaces the upstream CMakeLists, which hard-code
# machine-specific paths (Gurobi, OpenCV) and require Qt even for the CLI build.
#
# Typical use (normally driven by `python -m polyvec build`):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel
#   cmake --install build --prefix <somewhere>
#
# Options:
#   POLYVEC_BUILD_FIELD=ON|OFF
#   POLYVEC_BUILD_FLOW=ON|OFF
#   POLYVEC_SOLVER=auto|gurobi|highs   solver used by polyvec-flow (see below)
#   GUROBI_HOME=<path>                 Gurobi install, e.g. /opt/gurobi1103/linux64

cmake_minimum_required(VERSION 3.16)
project(polyvec_backends LANGUAGES CXX)

option(POLYVEC_BUILD_FIELD "Build the PolyVector-field backend (2019)" ON)
option(POLYVEC_BUILD_FLOW  "Build the PolyVector-flow backend (2021)"  ON)
set(POLYVEC_SOLVER "auto" CACHE STRING
    "Optimisation library for polyvec-flow: auto, gurobi or highs")
set_property(CACHE POLYVEC_SOLVER PROPERTY STRINGS auto gurobi highs)
set(POLYVEC_HIGHS_TAG "v1.7.2" CACHE STRING
    "HiGHS release fetched when POLYVEC_SOLVER=highs and no installed HiGHS is found")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(POLYVEC_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor" ABSOLUTE)
set(PV_DIR   "${POLYVEC_VENDOR_DIR}/pv")
set(PVF_DIR  "${POLYVEC_VENDOR_DIR}/pvflow")
set(PAAL_DIR "${POLYVEC_VENDOR_DIR}/paal")

# ---------------------------------------------------------------------------
# Dependencies shared by both programs
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost 1.58 REQUIRED)          # header-only parts (graph, range)
find_package(OpenMP)

# Everything both targets have in common.
function(polyvec_common_setup target)
  target_include_directories(${target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(MSVC)
    target_compile_definitions(${target} PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(${target} PRIVATE /bigobj)
  else()
    # Research code: keep the build log readable.
    target_compile_options(${target} PRIVATE -w)
  endif()
  install(TARGETS ${target} RUNTIME DESTINATION bin)
endfunction()

# ---------------------------------------------------------------------------
# polyvec-field  (Bessmeltsev & Solomon 2019) — plain CLI, no GUI
# ---------------------------------------------------------------------------
if(POLYVEC_BUILD_FIELD)
  file(GLOB PV_SOURCES CONFIGURE_DEPENDS "${PV_DIR}/src/*.cpp")
  # The Qt viewer is only compiled upstream when WITH_GUI is set.
  list(FILTER PV_SOURCES EXCLUDE REGEX "/(imagequiverviewer|mainwindow|MyScrollArea)\\.cpp$")

  add_executable(polyvec-field ${PV_SOURCES})
  target_include_directories(polyvec-field PRIVATE "${PV_DIR}/src")
  target_compile_features(polyvec-field PRIVATE cxx_std_14)
  polyvec_common_setup(polyvec-field)

  if(MSVC)
    # Without WITH_QT upstream times itself with <sys/time.h>, which MSVC lacks;
    # the WITH_QT code path only needs QtCore (QElapsedTimer), not a GUI.
    find_package(Qt5 REQUIRED COMPONENTS Core)
    target_compile_definitions(polyvec-field PRIVATE WITH_QT=1)
    target_link_libraries(polyvec-field PRIVATE Qt5::Core)
  endif()
endif()

# ---------------------------------------------------------------------------
# polyvec-flow  (Puhachov et al. 2021)
# ---------------------------------------------------------------------------
if(POLYVEC_BUILD_FLOW)
  # Upstream's main() always constructs its Qt debug window (it is simply not
  # shown when an output path is given), so Qt Widgets is a hard dependency.
  find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)

  file(GLOB PVF_SOURCES CONFIGURE_DEPENDS
       "${PVF_DIR}/src/*.cpp" "${PVF_DIR}/src/*.h" "${PVF_DIR}/src/*.hpp")
  # Files upstream's own src/CMakeLists.txt leaves out of the build.
  list(FILTER PVF_SOURCES EXCLUDE REGEX
       "/(chopThisCurve|GraphToTracing|OrientRoots)\\.(cpp|h)$|/hs071_nlp\\.hpp$")

  add_executable(polyvec-flow ${PVF_SOURCES})
  set_target_properties(polyvec-flow PROPERTIES AUTOMOC ON AUTOUIC ON)
  target_include_directories(polyvec-flow PRIVATE
      "${PVF_DIR}/src"
      "${PVF_DIR}"          # LBFGS/
      "${PAAL_DIR}")        # paal/greedy/steiner_tree_greedy.hpp
  target_compile_features(polyvec-flow PRIVATE cxx_std_17)
  target_compile_definitions(polyvec-flow PRIVATE WITH_GUI=1)
  target_link_libraries(polyvec-flow PRIVATE Qt5::Core Qt5::Gui Qt5::Widgets)
  polyvec_common_setup(polyvec-flow)

  # ---- solver -------------------------------------------------------------
  # Upstream is written against the Gurobi C++ API (commercial licence).  Two
  # ways to satisfy it:
  #   gurobi  link the real thing (needs GUROBI_HOME and a licence at run time)
  #   highs   compile compat/gurobi, a small implementation of the handful of
  #           Gurobi classes upstream uses, on top of the MIT-licensed HiGHS
  # "auto" picks gurobi when GUROBI_HOME points at an installation, else highs.
  if(NOT GUROBI_HOME AND DEFINED ENV{GUROBI_HOME})
    set(GUROBI_HOME "$ENV{GUROBI_HOME}")
  endif()

  set(_solver "${POLYVEC_SOLVER}")
  if(_solver STREQUAL "auto")
    if(GUROBI_HOME AND EXISTS "${GUROBI_HOME}/include/gurobi_c++.h")
      set(_solver "gurobi")
    else()
      set(_solver "highs")
    endif()
  endif()
  message(STATUS "polyvec-flow solver: ${_solver}")

  if(_solver STREQUAL "gurobi")
    find_path(GUROBI_INCLUDE_DIR gurobi_c++.h HINTS "${GUROBI_HOME}/include" REQUIRED)
    # The C library carries the version in its name (libgurobi110.so, gurobi110.lib).
    file(GLOB _gurobi_c_libs
         "${GUROBI_HOME}/lib/libgurobi[0-9]*.so"
         "${GUROBI_HOME}/lib/libgurobi[0-9]*.dylib"
         "${GUROBI_HOME}/lib/gurobi[0-9]*.lib")
    list(GET _gurobi_c_libs 0 GUROBI_C_LIBRARY)
    if(MSVC)
      file(GLOB _gurobi_cxx_libs "${GUROBI_HOME}/lib/gurobi_c++md20*.lib")
      list(SORT _gurobi_cxx_libs)
      list(GET _gurobi_cxx_libs -1 GUROBI_CXX_LIBRARY)
    else()
      find_library(GUROBI_CXX_LIBRARY NAMES gurobi_c++ HINTS "${GUROBI_HOME}/lib" REQUIRED)
    endif()
    if(NOT GUROBI_C_LIBRARY OR NOT GUROBI_CXX_LIBRARY)
      message(FATAL_ERROR "Gurobi libraries not found under GUROBI_HOME=${GUROBI_HOME}")
    endif()
    target_include_directories(polyvec-flow SYSTEM PRIVATE "${GUROBI_INCLUDE_DIR}")
    target_link_libraries(polyvec-flow PRIVATE "${GUROBI_CXX_LIBRARY}" "${GUROBI_C_LIBRARY}")

  elseif(_solver STREQUAL "highs")
    find_package(highs CONFIG QUIET)
    if(NOT TARGET highs::highs)
      message(STATUS "HiGHS not installed - fetching ${POLYVEC_HIGHS_TAG}")
      include(FetchContent)
      FetchContent_Declare(highs_src
          GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
          GIT_TAG        ${POLYVEC_HIGHS_TAG}
          GIT_SHALLOW    TRUE)
      FetchContent_GetProperties(highs_src)
      if(NOT highs_src_POPULATED)
        FetchContent_Populate(highs_src)
        set(BUILD_TESTING     OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        # EXCLUDE_FROM_ALL: build only the library we link, install nothing.
        add_subdirectory("${highs_src_SOURCE_DIR}" "${highs_src_BINARY_DIR}" EXCLUDE_FROM_ALL)
      endif()
    endif()
    target_sources(polyvec-flow PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/compat/gurobi/gurobi_shim.cpp")
    # BEFORE: make sure upstream's `#include <gurobi_c++.h>` finds the shim.
    target_include_directories(polyvec-flow BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/compat/gurobi")
    target_link_libraries(polyvec-flow PRIVATE highs::highs)

  else()
    message(FATAL_ERROR "POLYVEC_SOLVER must be auto, gurobi or highs (got '${POLYVEC_SOLVER}')")
  endif()
endif()
