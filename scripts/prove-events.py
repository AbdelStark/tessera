# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Summarize a `tessera --log-format json` stderr: one line per event, time and message."""
import json
import sys

for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        e = json.loads(line)
    except ValueError:
        print(line.rstrip()[:200])
        continue
    print(e.get("ts", "")[11:23], e.get("event", ""), str(e.get("fields", {}).get("message", ""))[:100])
