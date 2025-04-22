import numpy as np
import pandas as pd
import os

def read_interpolated_csv(filename):
    """Reads interpolated CSV and returns index, measurement, and reference arrays."""
    df = pd.read_csv(filename)
    return df['Index'].values, df['Interpolated Measurement'].values, df['Interpolated Reference'].values

def find_zero_crossings(reference, measurement, index):
    """
    Find zero-crossings in the reference signal.
    Return original index values and corresponding measurement values at those crossings.
    """
    zero_crossing_indices = np.where(np.diff(np.sign(reference)))[0]
    refined_indices = []
    measurement_at_crossings = []
    original_indices = []

    for i in zero_crossing_indices:
        # Pick the point closest to zero between i and i+1
        if abs(reference[i]) <= abs(reference[i + 1]):
            refined_indices.append(i)
        else:
            refined_indices.append(i + 1)

        original_indices.append(index[refined_indices[-1]])
        measurement_at_crossings.append(measurement[refined_indices[-1]])

    return original_indices, measurement_at_crossings, refined_indices

def save_crossing_data(filename, index, interpolated_indices, measurements):
    """Save zero-crossing info to a CSV file."""
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_zc.csv"

    df_out = pd.DataFrame({
        "Original Index": np.round(index[:len(interpolated_indices)], 6),
        "Interpolated Array Index": np.round(interpolated_indices, 6),
        "Measurement at Zero-Crossing": np.round(measurements, 6)
    })

    df_out.to_csv(output_filename, index=False)
    print(f"Saved zero-crossing data to {output_filename}")

def main():
    filename = input("Enter interpolated CSV file path: ").strip('"')
    index, measurement, reference = read_interpolated_csv(filename)
    original_idx_vals, meas_vals, interpolated_indices = find_zero_crossings(reference, measurement, index)
    save_crossing_data(filename, original_idx_vals, interpolated_indices, meas_vals)

if __name__ == "__main__":
    main()
