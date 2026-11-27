# siphon-serve in a container. The NVIDIA driver comes from the host through the NVIDIA
# container toolkit (`docker run --gpus all`); the image holds no driver libraries.
#
#   docker build -t siphon .
#   docker run --gpus all -p 8000:8000 -v <model-dir>:/models/inkling-2-mini:ro siphon \
#     /models/inkling-2-mini --host 0.0.0.0 --port 8000

FROM nvidia/cuda:13.5.0-devel-ubuntu24.04@sha256:d3fa71793540a2b76ab227ee00016e754da2758b7a51b49bc8a72543af5a8bc2 AS build
ARG RUST_VERSION=1.100.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "$RUST_VERSION"
ENV PATH=/root/.cargo/bin:$PATH \
    NVCC=/usr/local/cuda/bin/nvcc
WORKDIR /src
COPY . .
# The commit siphon-serve logs at startup: the copy holds no .git.
ARG SIPHON_COMMIT=unknown
ENV SIPHON_COMMIT=$SIPHON_COMMIT
RUN cargo build --release --locked -p siphon-serve \
    && cp target/release/siphon-serve /siphon-serve

FROM nvidia/cuda:13.5.0-base-ubuntu24.04@sha256:8c30b31b13a97ee353c21e666b227dea44f4331d38a221f257de70fc04b437d0
COPY --from=build /siphon-serve /usr/local/bin/siphon-serve
EXPOSE 8000
ENTRYPOINT ["siphon-serve"]
