# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent stdlib/struct calibration audit for #633 (no torch decoder)."""

import argparse
import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path

import blake3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--calibration", type=Path, required=True)
args = parser.parse_args()
root = args.calibration
report = json.loads((root / "calibration.json").read_text())
assert report["schema"] == "tessera.calibration/1" and report["seed"] == 0
assert (report["sequences"], report["held_out"], report["tokens_per_sequence"]) == (
    512,
    64,
    2048,
)
assert report["dataset"]["revision"] == "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
assert (
    report["environment"]["device"] == "mps"
    and report["environment"]["deterministic_algorithms"]
)
config = json.loads((args.checkpoint / "config.json").read_text())
a = report["architecture"]
for field, key in [
    ("layers", "num_hidden_layers"),
    ("d_model", "hidden_size"),
    ("d_ff", "intermediate_size"),
    ("heads", "num_attention_heads"),
    ("kv_heads", "num_key_value_heads"),
    ("vocab", "vocab_size"),
]:
    assert a[field] == config[key]
assert (a["layers"], a["d_model"]) == (32, 4096)
dm, heads, kv, ff = (
    a["d_model"],
    a["heads"] * a["d_head"],
    a["kv_heads"] * a["d_head"],
    a["d_ff"],
)
per = {
    "attn_input": dm,
    "attn_norm": dm,
    "q": heads,
    "k": kv,
    "v": kv,
    "attn_out": heads,
    "mlp_input": dm,
    "mlp_norm": dm,
    "gate": ff,
    "up": ff,
    "silu": ff,
    "h": ff,
}
expected = {
    f"layers.{l}.{name}": n for l in range(a["layers"]) for name, n in per.items()
} | {"final.input": dm, "final.norm": dm}
assert {v["name"]: v["channels"] for v in report["points"]} == expected
path = root / "statistics.safetensors"
data = path.read_bytes()
digest = blake3.blake3(
    data, derive_key_context="tessera-quant 2026-10-17 calibration statistics v1"
).hexdigest()
assert digest == report["statistics_digest"]
length = struct.unpack("<Q", data[:8])[0]
assert length < len(data) - 8
header = json.loads(data[8 : 8 + length])
meta = json.loads(header.pop("__metadata__")["tessera"])
assert meta == {
    "schema": "tessera.calibration-statistics/1",
    "calibration_selection": report["selection_digest"],
    "seed": 0,
    "sequences": 512,
    "tokens_per_sequence": 2048,
}
assert set(header) == set(expected)
end = 0
for name, t in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
    assert t["dtype"] == "F64" and t["shape"] == [expected[name]]
    lo, hi = t["data_offsets"]
    assert lo == end and hi - lo == 8 * expected[name]
    end = hi
    positive = False
    for (value,) in struct.iter_unpack(
        "<d", memoryview(data)[8 + length + lo : 8 + length + hi]
    ):
        assert math.isfinite(value) and value >= 0, "invalid private statistics"
        positive |= value > 0
    assert positive, "degenerate private statistics"
assert 8 + length + end == len(data)
for filename in ["statistics.safetensors", "calibration.json"]:
    assert stat.S_IMODE((root / filename).stat().st_mode) == 0o600
out = {
    "state": "verified-complete",
    "source": "NousResearch/Meta-Llama-3.1-8B",
    "model_revision": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
    "dataset_revision": report["dataset"]["revision"],
    "sequences": 512,
    "tokens_per_sequence": 2048,
    "held_out": 64,
    "statistics_bytes": len(data),
    "statistics_digest": digest,
    "statistics_sha256": hashlib.sha256(data).hexdigest(),
    "report_sha256": hashlib.sha256(
        (root / "calibration.json").read_bytes()
    ).hexdigest(),
    "points": len(expected),
    "channels": sum(expected.values()),
    "private_file_modes": "0600",
    "scope": "complete BF16 calibration capture and artifact consistency; not alpha selection, fidelity or proof",
}
fd = os.open(root / "verification.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
print(json.dumps(out), flush=True)
