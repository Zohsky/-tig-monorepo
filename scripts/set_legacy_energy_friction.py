#!/usr/bin/env python3

from pathlib import Path


path = Path("tig-algorithms/src/energy_arbitrage/titan_v6/mod.rs")
source = path.read_text()

for track in ("T51", "T52", "T53"):
    prefix = f"const {track}_FRICTION_ALPHA: f64 = "
    start = source.find(prefix)
    if start < 0:
        raise SystemExit(f"{track} friction alpha not found")
    value_start = start + len(prefix)
    value_end = source.find(";", value_start)
    source = source[:value_start] + "0.0" + source[value_end:]

path.write_text(source)
