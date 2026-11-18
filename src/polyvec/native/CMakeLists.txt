# Builds the two upstream programs that polyvec wraps, in two forms:
#
#   Python extension modules (POLYVEC_BUILD_PYTHON, the default under pip):
#     polyvec._native._field   PolyVector Fields, called in-process
#     polyvec._native._flow    PolyVector Flow, called in-process (HiGHS solver)
#
#   Command-line programs (POLYVEC_BUILD_CLI, the default otherwise; used by
#   `python -m polyvec build --local` and the Docker image):
#     polyvec-field   "Vectorization of Line Drawings via PolyVector Fields"
#                     (Bessmeltsev & Solomon 2019)        sources: ../vendor/pv
#     polyvec-flow    "Keypoint-Driven Line Drawing Vectorization via PolyVector Flow"
#                     (Puhachov et al. 2021)              sources: ../vendor/pvflow
#
# The upstream sources are compiled exactly as vendored; nothing in ../vendor is
# patched.  This file only replaces the upstream CMakeLists, which hard-code
# machine-specific paths (Gurobi, OpenCV) and require Qt even for the CLI build.
# The extension modules need no Qt at all: python/*_module.cpp replace upstream's
# main() and compat/headless stands in for the Qt and highgui headers the
# algorithm files include (but do not really use).
#
# Typical CLI use (normally driven by `python -m polyvec build`):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel
#   cmake --install build --prefix <somewhere>
#
# The extension modules are built by `pip install .` (scikit-build-core, see
# pyproject.toml).  Release wheels get their dependencies from vcpkg
# (vcpkg.json next to this file), linked statically.
#
# Options:
#   POLYVEC_BUILD_PYTHON=ON|OFF        extension modules (default: ON under scikit-build)
#   POLYVEC_REQUIRE_PYTHON=ON|OFF      fail instead of skipping them when a
#                                      dependency is missing (CI sets this)
#   POLYVEC_BUILD_CLI=ON|OFF           command-line programs (default: OFF under scikit-build)
#   POLYVEC_BUILD_FIELD=ON|OFF
#   POLYVEC_BUILD_FLOW=ON|OFF
#   POLYVEC_SOLVER=auto|gurobi|highs   solver used by the polyvec-flow CLI (see below)
#   GUROBI_HOME=<path>                 Gurobi install, e.g. /opt/gurobi1103/linux64

cmake_minimum_required(VERSION 3.16)
project(polyvec_backends LANGUAGES CXX)

if(SKBUILD)
  set(_python_default ON)
  set(_cli_default OFF)
else()
  set(_python_default OFF)
  set(_cli_default ON)
endif()
option(POLYVEC_BUILD_PYTHON  "Build the polyvec._native extension modules" ${_python_default})
option(POLYVEC_REQUIRE_PYTHON "Fail if the extension modules cannot be built" OFF)
option(POLYVEC_BUILD_CLI     "Build the polyvec-field / polyvec-flow programs" ${_cli_default})
option(POLYVEC_BUILD_FIELD   "Build the PolyVector-field backend (2019)" ON)
option(POLYVEC_BUILD_FLOW    "Build the PolyVector-flow backend (2021)"  ON)
set(POLYVEC_SOLVER "auto" CACHE STRING
    "Optimisation library for polyvec-flow: auto, gurobi or highs")
set_property(CACHE POLYVEC_SOLVER PROPERTY STRINGS auto gurobi highs)
set(POLYVEC_HIGHS_TAG "v1.7.2" CACHE STRING
    "HiGHS release fetched when HiGHS is needed and no installed HiGHS is found")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(POLYVEC_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor" ABSOLUTE)
set(PV_DIR     "${POLYVEC_VENDOR_DIR}/pv")
set(PVF_DIR    "${POLYVEC_VENDOR_DIR}/pvflow")
set(PAAL_DIR   "${POLYVEC_VENDOR_DIR}/paal")
set(COMPAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/compat")

# ---------------------------------------------------------------------------
# Upstream sources
# ---------------------------------------------------------------------------
file(GLOB PV_SOURCES CONFIGURE_DEPENDS "${PV_DIR}/src/*.cpp")
# The Qt viewer is only compiled upstream when WITH_GUI is set.
list(FILTER PV_SOURCES EXCLUDE REGEX "/(imagequiverviewer|mainwindow|MyScrollArea)\\.cpp$")

file(GLOB PVF_SOURCES CONFIGURE_DEPENDS
     "${PVF_DIR}/src/*.cpp" "${PVF_DIR}/src/*.h" "${PVF_DIR}/src/*.hpp")
# Files upstream's own src/CMakeLists.txt leaves out of the build.
list(FILTER PVF_SOURCES EXCLUDE REGEX
     "/(chopThisCurve|GraphToTracing|OrientRoots)\\.(cpp|h)$|/hs071_nlp\\.hpp$")

# The extension modules replace main() (they #include it, see python/) and
# never compile the Qt viewer.
set(PV_ALGORITHM_SOURCES ${PV_SOURCES})
list(FILTER PV_ALGORITHM_SOURCES EXCLUDE REGEX "/main\\.cpp$")
set(PVF_ALGORITHM_SOURCES ${PVF_SOURCES})
list(FILTER PVF_ALGORITHM_SOURCES EXCLUDE REGEX
     "/(main|imagequiverviewer|mainwindow|MyScrollArea)\\.(cpp|h|hpp)$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function(polyvec_compiler_setup target)
  if(MSVC)
    target_compile_definitions(${target} PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
    # /w: research code, keep the build log readable (as -w below).
    target_compile_options(${target} PRIVATE /bigobj /w)
    # Upstream spells some operators `and`/`or` (e.g. pvflow Coverage.cpp).
    # That is standard C++, but MSVC only knows these alternative tokens in
    # /permissive- mode or through <iso646.h>; force-include the header rather
    # than editing the authors' code (or making MSVC stricter everywhere).
    target_compile_options(${target} PRIVATE /FIiso646.h)
  else()
    # Research code: keep the build log readable.
    target_compile_options(${target} PRIVATE -w)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      # Upstream brace-initialises ints from doubles (e.g. std::array<int, 2>
      # pi = { std::round(..), .. } in AlmostReebGraph.cpp).  GCC only warns
      # about that narrowing; Clang (macOS) makes it an error.  Treat it as
      # GCC does instead of editing the authors' code.
      target_compile_options(${target} PRIVATE -Wno-c++11-narrowing)
    endif()
  endif()
endfunction()

# HiGHS, preferring an installed copy (vcpkg, conda, ...), else fetched.
macro(polyvec_find_highs)
  if(NOT TARGET highs::highs)
    find_package(highs CONFIG QUIET)
  endif()
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
      set(CMAKE_POSITION_INDEPENDENT_CODE ON)
      # EXCLUDE_FROM_ALL: build only the library we link, install nothing.
      add_subdirectory("${highs_src_SOURCE_DIR}" "${highs_src_BINARY_DIR}" EXCLUDE_FROM_ALL)
    endif()
  endif()
endmacro()

# ===========================================================================
# Python extension modules: polyvec._native._field / ._flow
# ===========================================================================
if(POLYVEC_BUILD_PYTHON)
  if(POLYVEC_REQUIRE_PYTHON)
    set(_find REQUIRED)
  else()
    set(_find QUIET)
  endif()

  find_package(Python 3.10 COMPONENTS Interpreter Development.Module ${_find})
  find_package(nanobind CONFIG ${_find})
  # No highgui: compat/headless replaces it.  imgcodecs only because upstream's
  # (unused) main() calls imread and is compiled into the module.
  find_package(OpenCV ${_find} COMPONENTS core imgproc imgcodecs)
  find_package(Eigen3 ${_find} NO_MODULE)
  find_package(Boost 1.58 ${_find})          # header-only parts (graph, range)

  set(_missing "")
  foreach(_dep Python nanobind OpenCV Eigen3 Boost)
    if(NOT ${_dep}_FOUND)
      list(APPEND _missing ${_dep})
    endif()
  endforeach()

  if(_missing)
    message(WARNING
      "polyvec: not building the in-process extension modules (missing: ${_missing}).\n"
      "The package still installs and can run the backends through Docker or\n"
      "locally compiled programs.  Wheels from PyPI include the modules; to build\n"
      "them yourself see README.md (vcpkg).")
  else()
    find_package(OpenMP)

    function(polyvec_add_module name module_source upstream_dir)
      # NOMINSIZE: nanobind would otherwise build the whole target with -Os,
      # which makes the numerical code markedly slower.
      nanobind_add_module(${name} NB_STATIC NOMINSIZE ${module_source} ${ARGN})
      polyvec_compiler_setup(${name})
      target_compile_features(${name} PRIVATE cxx_std_17)
      set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden
                                               VISIBILITY_INLINES_HIDDEN ON)
      # BEFORE: the headless Qt/highgui stand-ins must win over real headers.
      target_include_directories(${name} BEFORE PRIVATE "${COMPAT_DIR}/headless")
      target_include_directories(${name} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/python" "${upstream_dir}")
      target_include_directories(${name} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
      target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
      if(TARGET Boost::headers)
        target_link_libraries(${name} PRIVATE Boost::headers)
      endif()
      # Upstream asserts are debugging aids that abort the process; never in a module.
      target_compile_definitions(${name} PRIVATE NDEBUG)

      # Force-included everywhere: printf -> captured log (python/polyvec_log.h;
      # inactive in the module's own file), and the OpenMP stand-in where
      # OpenMP is unavailable.
      set(_force "${CMAKE_CURRENT_SOURCE_DIR}/python/polyvec_log.h")
      if(OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
      else()
        list(APPEND _force "${COMPAT_DIR}/noomp/omp.h")
      endif()
      foreach(_header ${_force})
        if(MSVC)
          target_compile_options(${name} PRIVATE "/FI${_header}")
        else()
          target_compile_options(${name} PRIVATE "SHELL:-include \"${_header}\"")
        endif()
      endforeach()
      set_property(SOURCE ${module_source} APPEND PROPERTY COMPILE_DEFINITIONS POLYVEC_ENTRY_TU)

      install(TARGETS ${name} LIBRARY DESTINATION polyvec/_native)
    endfunction()

    if(POLYVEC_BUILD_FIELD)
      polyvec_add_module(_field "${CMAKE_CURRENT_SOURCE_DIR}/python/field_module.cpp"
                         "${PV_DIR}/src" ${PV_ALGORITHM_SOURCES})
      # Selects upstream's QElapsedTimer code path (headless stand-in) over
      # <sys/time.h>, which MSVC lacks.  Not WITH_GUI: no viewer.
      target_compile_definitions(_field PRIVATE WITH_QT=1)
    endif()

    if(POLYVEC_BUILD_FLOW)
      polyvec_find_highs()
      polyvec_add_module(_flow "${CMAKE_CURRENT_SOURCE_DIR}/python/flow_module.cpp"
                         "${PVF_DIR}/src" ${PVF_ALGORITHM_SOURCES}
                         "${COMPAT_DIR}/gurobi/gurobi_shim.cpp")
      target_include_directories(_flow PRIVATE "${PVF_DIR}" "${PAAL_DIR}")
      # BEFORE: upstream's `#include <gurobi_c++.h>` must find the HiGHS shim.
      target_include_directories(_flow BEFORE PRIVATE "${COMPAT_DIR}/gurobi")
      target_link_libraries(_flow PRIVATE highs::highs)
      # Solver progress into the captured per-call log, not the console.
      target_compile_definitions(_flow PRIVATE POLYVEC_SHIM_LOG_TO_COUT)
    endif()
  endif()
endif()

# ===========================================================================
# Command-line programs
# ===========================================================================
if(POLYVEC_BUILD_CLI)
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
    polyvec_compiler_setup(${target})
    install(TARGETS ${target} RUNTIME DESTINATION bin)
  endfunction()

  # -------------------------------------------------------------------------
  # polyvec-field  (Bessmeltsev & Solomon 2019) — plain CLI, no GUI
  # -------------------------------------------------------------------------
  if(POLYVEC_BUILD_FIELD)
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

  # -------------------------------------------------------------------------
  # polyvec-flow  (Puhachov et al. 2021)
  # -------------------------------------------------------------------------
  if(POLYVEC_BUILD_FLOW)
    # Upstream's main() always constructs its Qt debug window (it is simply not
    # shown when an output path is given), so Qt Widgets is a hard dependency.
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)

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

    # ---- solver -----------------------------------------------------------
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
      polyvec_find_highs()
      target_sources(polyvec-flow PRIVATE "${COMPAT_DIR}/gurobi/gurobi_shim.cpp")
      # BEFORE: make sure upstream's `#include <gurobi_c++.h>` finds the shim.
      target_include_directories(polyvec-flow BEFORE PRIVATE "${COMPAT_DIR}/gurobi")
      target_link_libraries(polyvec-flow PRIVATE highs::highs)

    else()
      message(FATAL_ERROR "POLYVEC_SOLVER must be auto, gurobi or highs (got '${POLYVEC_SOLVER}')")
    endif()
  endif()
endif()
