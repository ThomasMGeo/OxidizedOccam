#!/usr/bin/env python3
"""Generate synthetic tilted-transmitter CSEM data for polarization ellipse example.

Creates a data file with EyReal, EyImag, EzReal, EzImag and all four PE
types (PEmax, PEmin, PBmax, PBmin) using a transmitter tilted 20 deg off
inline and 20 deg below horizontal.  This breaks the inline symmetry so
that Ex != 0, producing genuinely elliptical polarization.

Workflow:
  1. Copy model file from canonical_reservoir
  2. Write dummy data file (obs=1) with desired types and geometry
  3. Run forward model with true (reservoir) parameters → true response
  4. Add 2 % Gaussian noise → write final data file
  5. Write starting-model startup (1 Ωm halfspace)
  6. Clean up temporaries
"""

import csv
import math
import os
import random
import shutil
import subprocess
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXAMPLE_DIR, "../.."))
OCCAM_BIN = os.path.join(REPO_ROOT, "target/release/occam1d")

# ── geometry ────────────────────────────────────────────────────────
TX_AZIMUTH = 70.0      # 90 = inline (Y); 70 = 20° toward X
TX_DIP = 20.0           # degrees below horizontal
TX_Z = 950.0
TX_Y = [i * 500.0 for i in range(41)]   # 0 … 20 000 m

RX = (0.0, 0.0, 1000.0)
FREQS = [0.25, 1.0]

# data types: E real/imag + all four PE
DATA_TYPES = [3, 4, 5, 6, 41, 42, 43, 44]

# ── noise ───────────────────────────────────────────────────────────
NOISE_PCT = 0.02
E_FLOOR = 1e-15          # V/m per Am
B_FLOOR = 1e-18          # T per Am
SEED = 42

# ── true model (canonical reservoir) ───────────────────────────────
# 75 free layers: 1000-1975 → 1 Ωm, 2000-2075 → 100 Ωm, 2100-4500 → 1 Ωm
TRUE_PARAMS = [0.0]*40 + [2.0]*4 + [0.0]*31      # log10(ρ)
STARTING_PARAMS = [0.0]*75                         # 1 Ωm halfspace


def noise_floor(dtype):
    return B_FLOOR if dtype in (43, 44) else E_FLOOR


def write_emdata(path, transmitters, freqs, rx, data_points):
    with open(path, "w") as f:
        f.write("Format:  EMData_1.1\n")
        f.write(f"! Tilted transmitter PE example (az={TX_AZIMUTH}, dip={TX_DIP})\n")
        f.write(f"! Enoisefloor: {E_FLOOR} V/m/Am\n")
        f.write(f"! Bnoisefloor: {B_FLOOR} T/Am\n")
        f.write(f"! Percent Noise added: {NOISE_PCT*100:.0f} %\n")
        f.write("! Data rotated (theta,alpha,beta): 0, 0, 0 degrees\n")
        f.write("\n")
        f.write(f"# Transmitters:   {len(transmitters)}\n")
        f.write("!            X            Y            Z      Azimuth"
                "          Dip\n")
        for tx in transmitters:
            f.write(f"{'':13s}{tx[0]:<13g}{tx[1]:<13g}{tx[2]:<13g}"
                    f"{tx[3]:<13g}{tx[4]:<13g}\n")
        f.write(f"# Frequencies:    {len(freqs)}\n")
        for freq in freqs:
            f.write(f"{freq:14.6e}\n")
        f.write("Phase Convention: lag\n")
        f.write("# Receivers:      1\n")
        f.write("!            X            Y            Z        Theta"
                "        Alpha         Beta\n")
        f.write(f"{rx[0]:14.6e}{rx[1]:14.6e}{rx[2]:14.6e}"
                f"{'0.00':>12s}{'0.00':>12s}{'0.00':>12s}\n")
        f.write(f"# Data:           {len(data_points)}\n")
        f.write("!         Type        Freq#        Tx#           Rx#"
                "       Data    Std_Error\n")
        for dp in data_points:
            f.write(f"{dp['type']:12d}{dp['freq']:12d}{dp['tx']:12d}"
                    f"{dp['rx']:12d}{dp['value']:14.6e}{dp['error']:14.6e}\n")


def write_startup(path, data_file, params):
    with open(path, "w") as f:
        f.write("Format:             OCCAMITER_FLEX\n")
        f.write("Description:        tilted transmitter PE example\n")
        f.write("Model File:         model\n")
        f.write(f"Data File:          {data_file}\n")
        f.write("Date/Time:          02/09/2026 00:00:00.000\n")
        f.write("Iterations to run:  100\n")
        f.write("Target Misfit:      1.000000\n")
        f.write("Roughness Type:     1\n")
        f.write("Stepsize Cut Count: 8\n")
        f.write("Debug Level:        1\n")
        f.write("Iteration:          0\n")
        f.write("Lagrange Value:     5.000000\n")
        f.write("Roughness Value:    1.0000000E+10\n")
        f.write("Misfit Value:       1000.000\n")
        f.write("Misfit Reached:     0\n")
        f.write(f"Param Count:        {len(params)}\n")
        for i in range(0, len(params), 4):
            chunk = params[i:i+4]
            f.write("     " + "       ".join(f"{p}" for p in chunk) + "\n")


def main():
    random.seed(SEED)

    # 1. Copy model file
    shutil.copy2(os.path.join(EXAMPLE_DIR, "../canonical_reservoir/model"),
                 os.path.join(EXAMPLE_DIR, "model"))
    print("Copied model file")

    # 2. Build transmitter list
    transmitters = [(0.0, y, TX_Z, TX_AZIMUTH, TX_DIP) for y in TX_Y]

    # 3. Dummy data file — obs=1 everywhere, just to run the forward model
    dummy = []
    for fi in range(1, len(FREQS) + 1):
        for ti in range(1, len(transmitters) + 1):
            for dt in DATA_TYPES:
                dummy.append(dict(type=dt, freq=fi, tx=ti, rx=1,
                                  value=1.0, error=1.0))
    write_emdata(os.path.join(EXAMPLE_DIR, "dummy.data"),
                 transmitters, FREQS, RX, dummy)
    write_startup(os.path.join(EXAMPLE_DIR, "startup_true"),
                  "dummy.data", TRUE_PARAMS)

    # 4. Build binary if needed, then run forward with true model
    print("Building occam1d (release)...")
    subprocess.run(["cargo", "build", "--release", "-p", "occam_cli"],
                   cwd=REPO_ROOT, capture_output=True, check=True,
                   env={**os.environ, "PATH": os.environ.get("PATH", "")
                        + ":" + os.path.expanduser("~/.cargo/bin")})

    print("Running forward model with true parameters...")
    env = {**os.environ,
           "PATH": os.environ.get("PATH", "") + ":"
                   + os.path.expanduser("~/.cargo/bin")}
    r = subprocess.run(
        [OCCAM_BIN, "forward", "--startup", "startup_true",
         "--output", "true_response", "-v"],
        cwd=EXAMPLE_DIR, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        sys.exit("Forward model with true parameters failed")
    print(r.stdout)

    # 5. Read true response, add noise
    freq_map = {f: i+1 for i, f in enumerate(FREQS)}
    noisy = []
    with open(os.path.join(EXAMPLE_DIR, "true_response.csv")) as f:
        for row in csv.DictReader(f):
            dtype = int(row["type"])
            true_val = float(row["computed"])
            freq_hz = float(row["freq_hz"])
            tx_idx = int(row["tx_idx"])

            floor = noise_floor(dtype)
            err = max(NOISE_PCT * abs(true_val), floor)
            obs = true_val + random.gauss(0, err)

            noisy.append(dict(type=dtype, freq=freq_map[freq_hz],
                              tx=tx_idx, rx=1, value=obs, error=err))

    # 6. Write final data + starting startup
    data_name = "tilted_pe.data"
    write_emdata(os.path.join(EXAMPLE_DIR, data_name),
                 transmitters, FREQS, RX, noisy)
    write_startup(os.path.join(EXAMPLE_DIR, "startup"),
                  data_name, STARTING_PARAMS)

    # 7. Clean up temporaries
    for tmp in ("dummy.data", "startup_true",
                "true_response.csv", "true_response.resp"):
        p = os.path.join(EXAMPLE_DIR, tmp)
        if os.path.exists(p):
            os.remove(p)

    print(f"\nGenerated {len(noisy)} data points in {data_name}")
    print(f"  {len(transmitters)} transmitters, {len(FREQS)} frequencies, "
          f"{len(DATA_TYPES)} types")
    print(f"  Transmitter: azimuth={TX_AZIMUTH}°, dip={TX_DIP}°")
    print(f"\nTo run forward:   occam1d forward --startup startup")
    print(f"To run inversion: occam1d invert  --startup startup")


if __name__ == "__main__":
    main()
