# Tessera's adapter image (RFC-0005), draft for systems/tessera (#741). Not built yet: the RFC-0014
# rules (B1 every FROM by digest, B2 one fetch stage with every download in inputs.lock, B3
# --network=none after it, B5 /opt/iproofs/inputs.json, B6 the adapter commit) are still to be
# applied by the images workflow. Hosted development runs install the checkout at /opt/tessera
# in the container the provider started, and the harness reads this ENTRYPOINT (RFC-0006
# 'Hosted execution', RFC-0014 development install).
#   docker build --build-arg TESSERA_COMMIT=<40-hex> -f systems/tessera/Dockerfile systems
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git build-essential pkg-config python3 \
 && rm -rf /var/lib/apt/lists/*
ARG TESSERA_COMMIT
RUN git clone https://github.com/AbdelStark/tessera /opt/tessera \
 && git -C /opt/tessera checkout "$TESSERA_COMMIT"
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none \
 && cd /opt/tessera && cargo build --release --locked -p tessera-cli --features gpu \
 && cd verifier && cargo build --release --locked -p tv-cli && cargo build --release --locked -p tv-cli --example statement_pin
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
RUN cd /opt/tessera/python && uv sync --frozen --extra cu130 && chmod -R a+rX /opt
RUN useradd --uid 10001 --no-create-home adapter
USER 10001
ENTRYPOINT ["python3", "/opt/tessera/bench/inference-proofs/tessera/entrypoint"]
