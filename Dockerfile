# syntax=docker/dockerfile:1
#
# Build/test image for the online-VAD feature on branch
# feat/online-silero-vad-endpoint. CPU-only (Silero VAD + a streaming
# zipformer2 model are both small); no GPU/TensorRT needed to validate
# this feature. Uses a modern Ubuntu release (22.04, GCC 11) specifically
# because the prebuilt onnxruntime 1.28.2 release this repo downloads
# fails to link on Ubuntu 20.04's GCC 9/10 here (undefined reference to
# `std::__throw_bad_array_new_length()` -- a pre-existing, environment-
# only issue unrelated to this feature; see README.online-vad.md).
#
# Structured like ds-k2-sherpa-onnx's Dockerfile: deps that don't depend
# on the repo install above the source COPY so they stay cached across
# code edits, and the build uses a BuildKit cache mount on build/ for
# incremental recompiles. Build with BuildKit (default on Docker >= 23;
# else `DOCKER_BUILDKIT=1 docker build ...`).

FROM ubuntu:22.04
LABEL maintainer="manickavela1998@gmail.com"

WORKDIR /workspace
ARG DEBIAN_FRONTEND=noninteractive

# ---- Tier 1: OS deps (no repo input -> cached) ----
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        build-essential cmake git ca-certificates curl wget \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ---- Tier 2: source + incremental sherpa build (the only repo-dependent layer) ----
RUN mkdir -p /workspace/sherpa-onnx
COPY . /workspace/sherpa-onnx/
WORKDIR /workspace/sherpa-onnx

# CPU-only; WEBSOCKET on (the feature under test); TESTS on (existing unit
# tests, e.g. circular-buffer-test, still a cheap correctness signal);
# PORTAUDIO off (no system audio device/lib in a container); PYTHON off
# (not needed for this feature; keeps the build fast).
RUN --mount=type=cache,target=/workspace/sherpa-onnx/build \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=ON \
          -DSHERPA_ONNX_ENABLE_GPU=OFF \
          -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
          -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
          -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \
          -DSHERPA_ONNX_ENABLE_TESTS=ON && \
    cmake --build build -j"$(nproc)" && \
    mkdir -p /opt/sherpa/bin /opt/sherpa/lib && \
    cp -a build/bin/. /opt/sherpa/bin/ && \
    cp -a build/lib/. /opt/sherpa/lib/ && \
    cp -aL build/_deps/onnxruntime-src/lib/lib*.so* /opt/sherpa/lib/ 2>/dev/null || true

ENV LD_LIBRARY_PATH=/opt/sherpa/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/sherpa/bin:${PATH}

# Test assets (Silero VAD model, a small streaming ASR model, test wavs)
# are expected as a bind-mounted volume at /models -- see
# docker/run_vad_tests.sh and README.online-vad.md for the expected
# layout and how to fetch them. Not baked into the image: keeps the
# image small and avoids re-downloading on every rebuild.
COPY docker/run_vad_tests.sh /opt/sherpa/run_vad_tests.sh
COPY docker/start_websocket_server.sh /opt/sherpa/start_websocket_server.sh
RUN chmod +x /opt/sherpa/run_vad_tests.sh /opt/sherpa/start_websocket_server.sh

WORKDIR /opt/sherpa
CMD ["/opt/sherpa/run_vad_tests.sh"]
