#!/usr/bin/env python3
"""Generate PE-augmented EMData files.

Reads an existing EMData file, computes PEmax (type 41) observed values
from matching EyReal (type 3) / EyImag (type 4) pairs, and writes a new
data file with the PE data points appended.

Usage:
    python generate_pe_data.py <input.data> [output.data]

If output path is not given, writes to <input_pe.data>.
"""

import math
import os
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_pe{ext}"

    with open(input_path) as f:
        lines = f.readlines()

    # Find "# Data:" line
    data_header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("# data"):
            data_header_idx = i
            break

    if data_header_idx is None:
        sys.exit("Could not find '# Data:' line")

    # Parse existing data count
    parts = lines[data_header_idx].split(":")
    old_n_data = int(parts[1].strip())

    # Find start of data rows (skip comment line)
    data_start = data_header_idx + 1
    if lines[data_start].strip().startswith("!"):
        data_start += 1

    # Parse all data points
    data_points = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) >= 6:
            data_points.append(
                {
                    "type": int(tokens[0]),
                    "freq": int(tokens[1]),
                    "tx": int(tokens[2]),
                    "rx": int(tokens[3]),
                    "value": float(tokens[4]),
                    "error": float(tokens[5]),
                }
            )

    # Group by (freq, tx, rx) to find EyReal/EyImag pairs
    groups = defaultdict(dict)
    for dp in data_points:
        key = (dp["freq"], dp["tx"], dp["rx"])
        groups[key][dp["type"]] = dp

    # Generate PEmax data points
    pe_points = []
    for key in sorted(groups.keys()):
        types = groups[key]
        if 3 in types and 4 in types:
            ey_real = types[3]["value"]
            ey_imag = types[4]["value"]
            pe_max = math.sqrt(ey_real**2 + ey_imag**2)
            # Use same relative error (approx 2%), with noise floor
            err = max(0.02 * abs(pe_max), 1e-15)
            pe_points.append(
                {
                    "type": 41,
                    "freq": key[0],
                    "tx": key[1],
                    "rx": key[2],
                    "value": pe_max,
                    "error": err,
                }
            )

    if not pe_points:
        sys.exit("No EyReal/EyImag pairs found in data file")

    # Update data count
    new_n_data = old_n_data + len(pe_points)
    lines[data_header_idx] = f"# Data:           {new_n_data}\n"

    # Write output: original file + PE data points appended
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line)
        for dp in pe_points:
            f.write(
                f"{dp['type']:12d}{dp['freq']:12d}{dp['tx']:12d}"
                f"{dp['rx']:12d}{dp['value']:14.6e}{dp['error']:14.6e}\n"
            )

    print(f"Added {len(pe_points)} PEmax data points")
    print(f"Total data points: {new_n_data}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
