import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d

def read_data(filename):
    """Reads CSV, auto-detects where numeric data starts, returns measurement and reference arrays."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    start_index = 0
    for i, line in enumerate(lines):
        try:
            float(line.strip().split(',')[0])
            start_index = i
            break
        except ValueError:
            continue

    df = pd.read_csv(filename, delimiter=",", usecols=[0,1], skiprows=start_index, header=None)
    return df.iloc[:, 0].values, df.iloc[:, 1].values

def interpolate_data(reference_array, measure_array, factor):
    """Performs cubic interpolation with given factor."""
    x = np.arange(reference_array.size)
    new_x = np.arange(0, reference_array.size - 1 + 1e-6, 1 / factor)
    interp_ref = interp1d(x, reference_array, kind="cubic")(new_x)
    interp_meas = interp1d(x, measure_array, kind="cubic")(new_x)
    return new_x, interp_ref, interp_meas

def save_interpolated_data(original_filename, new_x, interp_ref, interp_meas, factor):
    """Saves interpolated data to new CSV with _interp{factor} suffix."""
    base, ext = os.path.splitext(original_filename)
    output_filename = f"{base}_interp{factor}{ext}"
    df_out = pd.DataFrame({
        "Index": np.round(new_x, 6),
        "Interpolated Measurement": np.round(interp_meas, 6),
        "Interpolated Reference": np.round(interp_ref, 6)
    })
    df_out.to_csv(output_filename, index=False)
    print(f"Saved interpolated data to {output_filename}")

def main():
    filename = input("Enter CSV file path: ").strip('"')
    interp_factor = int(input("Interpolation factor (e.g., 10): ") or 10)

    measure_array, reference_array = read_data(filename)
    new_x, interp_ref, interp_meas = interpolate_data(reference_array, measure_array, interp_factor)
    save_interpolated_data(filename, new_x, interp_ref, interp_meas, interp_factor)

if __name__ == "__main__":
    main()
