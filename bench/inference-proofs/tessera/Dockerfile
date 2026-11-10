# Tessera's adapter image (RFC-0005), draft for systems/tessera (#741), not built yet: the images
# workflow still has to add inputs.lock (B2), the network-less build stage (B3), inputs.json (B5)
# and the adapter commit (B6). Hosted development runs install the checkout at /opt/tessera in the
# container the provider started, and the harness reads this ENTRYPOINT (RFC-0006 'Hosted
# execution', RFC-0014 development install).
#   docker build --build-arg TESSERA_COMMIT=<40-hex> -f systems/tessera/Dockerfile systems
FROM docker.io/nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 24.04 packages at exact versions (PIN-5); the CUDA devel image carries the C toolchain.
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    git=1:2.43.0-1ubuntu7.4 pkg-config=1.8.1-2build1 python3=3.12.3-0ubuntu2.1 \
 && rm -rf /var/lib/apt/lists/*
ARG TESSERA_COMMIT
ADD --keep-git-dir=true https://github.com/AbdelStark/tessera.git#${TESSERA_COMMIT} /opt/tessera
# rust-toolchain.toml's 1.95.0.
ADD --checksum=sha256:2e0338f18ecbaa4a0f631b9e80e8b8e26bb6fe77dd5454fba8a70cf96c1e84a1 https://static.rust-lang.org/dist/rust-1.95.0-x86_64-unknown-linux-gnu.tar.xz /fetch/
COPY --from=ghcr.io/astral-sh/uv@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03 /uv /usr/local/bin/uv
ENV CARGO_HOME=/opt/cargo PATH=/opt/rust/bin:$PATH UV_PYTHON_DOWNLOADS=never
RUN tar -xJf /fetch/rust-1.95.0-x86_64-unknown-linux-gnu.tar.xz -C /tmp \
 && /tmp/rust-1.95.0-x86_64-unknown-linux-gnu/install.sh --prefix=/opt/rust \
    --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu \
 && rm -rf /tmp/rust-1.95.0-x86_64-unknown-linux-gnu /fetch \
 && cd /opt/tessera && cargo build --release --locked -p tessera-cli --features gpu \
 && cd verifier && cargo build --release --locked -p tv-cli \
 && cargo build --release --locked -p tv-cli --example statement_pin \
 && cd /opt/tessera/python && uv sync --frozen --extra cu130 \
 && chmod -R a+rX /opt
RUN useradd --uid 10001 --no-create-home adapter
USER 10001
ENTRYPOINT ["python3", "/opt/tessera/bench/inference-proofs/tessera/entrypoint"]
