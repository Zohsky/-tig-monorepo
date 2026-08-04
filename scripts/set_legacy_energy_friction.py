#!/usr/bin/env python3

from pathlib import Path


path = Path("tig-algorithms/src/energy_arbitrage/titan_v6/mod.rs")
source = path.read_text()
corrected = "(kappa * (1.0 + eta_rt), kappa * (1.0 + 1.0 / eta_rt))"
legacy = "(2.0 * kappa, 2.0 * kappa)"

if source.count(corrected) != 1:
    raise SystemExit("central friction expression not found exactly once")

path.write_text(source.replace(corrected, legacy))
