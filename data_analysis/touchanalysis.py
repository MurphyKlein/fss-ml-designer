"""
touchanalysis.py
----------------
Given a Touchstone S-parameter file, parses all frequency blocks and their associated
gamma arrays, and extracts key features: minimum reflection and maximum transmission
values and their corresponding frequencies.
"""

import numpy as np


def parse_s_block_fixed(lines):
    """
    Converts a block of string lines (from touchstone) into a 20x20 S-parameter matrix.
    Each line has magnitude and phase pairs; returns a complex-valued S-matrix.
    """
    data = []
    for line in lines:
        items = line.strip().split()
        # Sometimes E-notation comes as 'E', sometimes as 'e'
        try:
            floats = [float(x.replace('E', 'e')) for x in items]
        except Exception as err:
            print(f"Line parsing failed: {line}\n{err}")
            continue  # Just skip any bad lines
        # Each pair is (mag, phase) for one element
        for idx in range(0, len(floats), 2):
            mag = floats[idx]
            phase = floats[idx + 1]
            # Touchstone phase is degrees; convert to radians for np.exp
            phs_rad = np.deg2rad(phase)
            data.append(mag * np.exp(1j * phs_rad))
    # The order='C' is default, but just to be explicit
    try:
        s_matrix = np.array(data).reshape((20, 20), order='C')
    except Exception as e:
        print("Error in reshaping S-matrix. Maybe the block isn't 400 elements?", e)
        raise
    return s_matrix


def extract_all_freq_blocks_with_gamma(lines):
    """
    Loops through lines, grabbing each frequency block and its gamma array.
    Returns a dictionary of {frequency: S-matrix} and {frequency: gamma array}.
    """
    freq_to_s = {}
    gamma_dict = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip comments, headers, and empty lines
        if not line or line.startswith("!") or line.startswith("#"):
            i += 1
            continue

        try:
            freq = float(line.split()[0])
            s_lines = []
            # If this line also contains data, take that part too
            items = line.strip().split()
            if len(items) > 1:
                s_lines.append(' '.join(items[1:]))
            i += 1

            # Accumulate lines until we see a gamma marker
            while i < len(lines) and not lines[i].strip().startswith("! Gamma"):
                this_line = lines[i].strip()
                if this_line and not this_line.startswith("!"):
                    s_lines.append(this_line)
                i += 1

            # Gamma line: parse values
            if i < len(lines) and lines[i].strip().startswith("! Gamma"):
                gamma_line = lines[i].strip().replace("! Gamma", "").strip()
                gamma_vals = [float(val) for val in gamma_line.split()]
                gamma_dict[freq] = gamma_vals
                i += 1

            # Parse the S-matrix block for this frequency
            try:
                freq_to_s[freq] = parse_s_block_fixed(s_lines)
            except Exception as err:
                print(f"Problem parsing S block at {freq} GHz:", err)

        except ValueError:
            # Sometimes you'll get junk lines, just skip them
            i += 1

    return freq_to_s, gamma_dict


def analysis(file_path):
    """
    Given a file path to a Touchstone file, parses all S-matrix and gamma data,
    and extracts min reflection and max transmission values (and their freqs)
    for all frequencies in the file.
    Returns a list:
        [min_reflection_value, min_reflection_freq, max_transmission_value, max_transmission_freq]
    """
    # Example: file_path = r"C:\Users\mdklein\AEDTCode\data\FSS_FSS.s20p"
    with open(file_path, "r") as f:
        lines = f.readlines()

    s_blocks, gammas = extract_all_freq_blocks_with_gamma(lines)

    # Track the minimum reflected power and max transmitted power
    best_reflection = {"val": float("inf"), "freq": None}
    best_transmission = {"val": float("-inf"), "freq": None}

    # Loop over each frequency point
    for freq, s_mat in s_blocks.items():
        gamma_vals = gammas.get(freq)
        if gamma_vals is None or len(gamma_vals) < 40:
            # Can't reliably determine propagating modes
            continue

        # Propagating modes: gamma == 0 (real axis, no decay)
        propagating = []
        for k in range(20):
            if gamma_vals[k * 2] == 0:  # even index is Re(Gamma), odd is Im
                propagating.append(k)
        # Port1: 0-9, Port2: 10-19
        port1_modes = [k for k in propagating if k < 10]
        port2_modes = [k for k in propagating if k >= 10]

        for inc_idx in port1_modes:
            # Reflected: all power from port 1 back into port 1
            refl_power = 0.0
            for col in port1_modes:
                # |S_ij|^2: incident mode i, reflected into mode j (port 1)
                refl_power += abs(s_mat[inc_idx, col]) ** 2

            # Transmitted: incident port 1, transmitted to port 2
            tran_power = 0.0
            for col in port2_modes:
                tran_power += abs(s_mat[inc_idx, col]) ** 2

            # If this is a new minimum/maximum, store the value and freq
            if refl_power < best_reflection["val"]:
                best_reflection = {"val": refl_power, "freq": freq}
            if tran_power > best_transmission["val"]:
                best_transmission = {"val": tran_power, "freq": freq}

    # Prepare the output in the order requested
    return [
        best_reflection["val"],
        best_reflection["freq"],
        best_transmission["val"],
        best_transmission["freq"]
    ]
