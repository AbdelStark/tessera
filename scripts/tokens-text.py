# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Print a TokenFile's one sequence as decimal ids separated by spaces (for tessera-verify --tokens)."""
import sys
from pathlib import Path

from tessera_quant.tce import MESSAGE_LIMITS, TOKEN_FILE, read_artifact

(sequence,) = read_artifact(Path(sys.argv[1]), TOKEN_FILE, limits=MESSAGE_LIMITS).sequences
print(" ".join(str(int(t)) for t in sequence))
