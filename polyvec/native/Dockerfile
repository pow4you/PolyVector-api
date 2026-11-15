# Image holding the two compiled backends (polyvec-field, polyvec-flow).
#
# Built and used automatically by polyvec.runners.DockerRunner; to build by hand,
# run from the polyvec/ package directory:
#
#   docker build -f native/Dockerfile -t polyvec-backends .
#
# The Python wrapper mounts a scratch directory at /work and runs one of the
# programs in it, e.g.
#
#   docker run --rm -v "$PWD":/work polyvec-backends polyvec-field drawing.png
#
# polyvec-flow is built with the open-source HiGHS solver here; to use Gurobi,
# build locally instead (`python -m polyvec build --local --solver gurobi`).
#
# Build arguments:
#   POLYVEC_BUILD_JOBS  parallel compile jobs (default 4, ~1.5 GB of RAM each)

# ---------------------------------------------------------------- build stage
FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        qtbase5-dev libopencv-dev libboost-graph-dev libeigen3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY vendor/pv/src       /src/vendor/pv/src
COPY vendor/pvflow/src   /src/vendor/pvflow/src
COPY vendor/pvflow/LBFGS /src/vendor/pvflow/LBFGS
COPY vendor/paal         /src/vendor/paal
COPY native              /src/native

# Each translation unit pulls in Eigen, Boost.Graph and OpenCV and needs well
# over 1 GB to compile; unlimited parallelism exhausts the Docker VM's memory.
ARG POLYVEC_BUILD_JOBS=4
RUN cmake -S /src/native -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/polyvec \
        -DPOLYVEC_SOLVER=highs \
    && cmake --build /build --parallel ${POLYVEC_BUILD_JOBS} \
    && cmake --install /build

# -------------------------------------------------------------- runtime stage
# Only the shared libraries the programs link against (same Ubuntu release as
# the build stage, hence the versioned package names).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d \
        libopencv-highgui4.5d libqt5widgets5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/polyvec /opt/polyvec

# Fail the build, not the first run, if a shared library is missing.
RUN ! ldd /opt/polyvec/bin/polyvec-field /opt/polyvec/bin/polyvec-flow | grep "not found"

# offscreen: polyvec-flow creates (but never shows) a Qt window, so it needs a
# Qt platform plugin that works without a display.
ENV PATH=/opt/polyvec/bin:$PATH \
    QT_QPA_PLATFORM=offscreen

WORKDIR /work
