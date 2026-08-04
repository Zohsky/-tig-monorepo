#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
import subprocess
import time


def run(cmd):
    started = time.monotonic()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--nonces", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fuel", type=int, required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="energy-arbitrage-friction-paired")
    args = parser.parse_args()

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = json.dumps(
        {
            "algorithm_id": "",
            "challenge_id": "c008",
            "track_id": f"s={args.scenario}",
            "block_id": "",
            "player_id": "",
        },
        separators=(",", ":"),
    )

    rows = []
    for nonce in range(args.start, args.start + args.nonces):
        for variant, binary in (
            ("legacy", args.legacy),
            ("corrected", args.corrected),
        ):
            variant_dir = output / variant
            variant_dir.mkdir(exist_ok=True)
            runtime, runtime_ms = run(
                [
                    "tig-runtime",
                    settings,
                    args.seed,
                    str(nonce),
                    binary,
                    "--fuel",
                    str(args.fuel),
                    "--output",
                    str(variant_dir),
                ]
            )
            result_file = variant_dir / f"{nonce}.json"
            verifier, verifier_ms = run(
                [
                    "tig-verifier",
                    settings,
                    args.seed,
                    str(nonce),
                    str(result_file),
                ]
            )
            payload = json.loads(result_file.read_text())
            quality_line = next(
                line for line in verifier.stdout.splitlines() if line.startswith("quality: ")
            )
            rows.append(
                {
                    "track": args.track,
                    "scenario": args.scenario,
                    "nonce": nonce,
                    "variant": variant,
                    "quality": int(quality_line.removeprefix("quality: ")),
                    "fuel_consumed": int(payload["fuel_consumed"]),
                    "runtime_ms": runtime_ms,
                    "verifier_ms": verifier_ms,
                    "runtime_stdout": runtime.stdout.strip(),
                }
            )

    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
