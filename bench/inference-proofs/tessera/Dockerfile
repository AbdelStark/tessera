# Tessera's adapter (RFC-0005), built by the RFC-0014 rules: B1 every FROM by digest; B2 one
# `fetch` stage, every download in inputs.lock or in the pinned source's own lockfiles (Cargo.lock
# and verifier/Cargo.lock checksums, python/uv.lock hashes), fetched by the locked tools; B3
# `RUN --network=none` after it; B4 no patch or sed over upstream source; B5 /opt/iproofs/inputs.json;
# B6 the adapter commit and dirty flag as build arguments. No GPU is needed to build: the CUDA
# devel image carries nvcc for the prover's kernels. Build with the systems/ directory as context:
#   docker build --build-arg IPROOFS_ADAPTER_COMMIT=$(git rev-parse HEAD) \
#     --build-arg IPROOFS_ADAPTER_DIRTY=$(test -z "$(git status --porcelain)" && echo false || echo true) \
#     -f systems/tessera/Dockerfile systems
# CUDA 12.8.1 devel on Ubuntu 24.04.
FROM docker.io/nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS fetch
ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu packages at exact versions (PIN-5), checked by apt against the archive's signed index.
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    git=1:2.43.0-1ubuntu7.4 pkg-config=1.8.1-2build1 python3=3.12.3-0ubuntu2.1 \
 && rm -rf /var/lib/apt/lists/*
COPY tessera/inputs.lock /fetch/
ADD --keep-git-dir=true https://github.com/AbdelStark/tessera.git#3c75bdf78e4cc5c9ed2140900ff109d5933e3006 /opt/tessera
ADD --checksum=sha256:2e0338f18ecbaa4a0f631b9e80e8b8e26bb6fe77dd5454fba8a70cf96c1e84a1 https://static.rust-lang.org/dist/rust-1.95.0-x86_64-unknown-linux-gnu.tar.xz /fetch/
# The Hub's cache layout for a commit revision: offline, hf_hub_download returns the snapshot file.
ARG HF=/opt/hf/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
ADD --checksum=sha256:1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00000-of-00008.json.gz $HF/en/
ADD --checksum=sha256:db4c23d7a336032de37dfe2f3ed39653c8afd94004d4e8d9c73c03222299292c https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00001-of-00008.json.gz $HF/en/
ADD --checksum=sha256:4bf6b248b0f910dcde2cdf2118d6369d8208c8f9515ec29ab73e531f380b18e2 https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00002-of-00008.json.gz $HF/en/
ADD --checksum=sha256:082b72c1ec600502df2e4403df6884390c85627ffdf37a5b700397c9a1ac6b8e https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00003-of-00008.json.gz $HF/en/
ADD --checksum=sha256:37c2ce46864db0b6f1fe89220ebf4ef811ea10485c8336f1712227bbbdb530e1 https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00004-of-00008.json.gz $HF/en/
ADD --checksum=sha256:00b164e59df49df5dc1df170db99bb0d9207a6329cb9272ed8b6ef7a55058359 https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00005-of-00008.json.gz $HF/en/
ADD --checksum=sha256:86896a9baffe0ab9df39a2901f0e515009989ca1fec9a42583d718ce16166b5a https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00006-of-00008.json.gz $HF/en/
ADD --checksum=sha256:0bf87534c7050091c60466df50f174c3147d052ec7dc645da9574c902445beaf https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00007-of-00008.json.gz $HF/en/
COPY --from=ghcr.io/astral-sh/uv@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03 /uv /usr/local/bin/uv
ENV CARGO_HOME=/opt/cargo PATH=/opt/rust/bin:$PATH UV_PYTHON_DOWNLOADS=never UV_CACHE_DIR=/tmp/uv-cache
RUN test "$(git -C /opt/tessera rev-parse HEAD)" = 3c75bdf78e4cc5c9ed2140900ff109d5933e3006 \
 && test "sha256:$(git -C /opt/tessera archive --format=tar HEAD | sha256sum | cut -d' ' -f1)" \
         = "$(awk -F'\t' '$1 == "tessera" {print $3}' /fetch/inputs.lock)" \
 && tar -xJf /fetch/rust-1.95.0-x86_64-unknown-linux-gnu.tar.xz -C /tmp \
 && /tmp/rust-1.95.0-x86_64-unknown-linux-gnu/install.sh --prefix=/opt/rust \
    --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu \
 && rm -rf /tmp/rust-1.95.0-x86_64-unknown-linux-gnu /fetch/rust-1.95.0-x86_64-unknown-linux-gnu.tar.xz \
 && cd /opt/tessera && cargo fetch --locked \
 && cd verifier && cargo fetch --locked \
 && cd /opt/tessera/python && uv sync --frozen --extra cu130 \
 && rm -rf /tmp/uv-cache

FROM fetch
WORKDIR /opt/tessera
RUN --network=none cargo build --release --locked --offline -p tessera-cli --features gpu \
 && cd verifier && cargo build --release --locked --offline -p tv-cli \
 && cargo build --release --locked --offline -p tv-cli --example statement_pin \
 && rm -rf /opt/tessera/target/release/build /opt/tessera/target/release/deps \
    /opt/tessera/target/release/incremental /opt/tessera/verifier/target/release/build \
    /opt/tessera/verifier/target/release/deps /opt/tessera/verifier/target/release/incremental
COPY fake/write_inputs.py /opt/iproofs/write_inputs.py
COPY tessera/entrypoint tessera/calibration.json /opt/iproofs/tessera/
RUN --network=none useradd --uid 10001 --no-create-home adapter \
 && python3 /opt/iproofs/write_inputs.py /fetch/inputs.lock /opt/iproofs/inputs.json \
    rust="$(rustc --version)" cuda="$(nvcc --version | grep -o 'V[0-9.]*$')" \
    torch="$(/opt/tessera/python/.venv/bin/python -c 'import torch; print(torch.__version__)')" \
    tessera="$(/opt/tessera/target/release/tessera --version)" \
 && chmod -R a+rX /opt
ENV TESSERA_REPO=/opt/tessera HF_HUB_CACHE=/opt/hf HF_HUB_OFFLINE=1
ARG IPROOFS_ADAPTER_COMMIT=unknown
ARG IPROOFS_ADAPTER_DIRTY=true
ENV IPROOFS_ADAPTER_COMMIT=${IPROOFS_ADAPTER_COMMIT} IPROOFS_ADAPTER_DIRTY=${IPROOFS_ADAPTER_DIRTY}
USER 10001
ENTRYPOINT ["python3", "/opt/iproofs/tessera/entrypoint"]
