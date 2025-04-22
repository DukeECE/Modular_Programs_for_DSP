import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """
    Load the CSV file, auto-detect where numeric data begins by skipping non-numeric header rows.
    """
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

    df = pd.read_csv(filename, skiprows=start_index, header=None)
    return df

def plot_and_select_truncation(df, filename, column_name, downsample_factor=1):
    y = df[column_name].values
    x = np.arange(len(y))
    y_downsampled = y[::downsample_factor]
    x_downsampled = x[::downsample_factor]
    data_length = len(y)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_downsampled, y_downsampled, '-', linewidth=0.5, color='b', label=f"Column {column_name}")
    ax.set_title("Zoom & Pan | Click to set LEFT (green) and RIGHT (red) edges | Press Enter to confirm")
    ax.set_xlabel("Index")
    ax.set_ylabel("Signal")
    ax.legend()

    left_point = [None]
    right_point = [None]
    left_line = [None]
    right_line = [None]
    selection_phase = ['left']

    def on_click(event):
        if event.xdata is not None and 0 <= int(event.xdata) < data_length:
            val = int(event.xdata)
            if selection_phase[0] == 'left':
                left_point[0] = val
                if left_line[0] is not None:
                    left_line[0].remove()
                left_line[0] = ax.axvline(val, color='g', linestyle='--', label='Left Edge')
                print(f"Left Edge selected at index: {val}")
            elif selection_phase[0] == 'right':
                right_point[0] = val
                if right_line[0] is not None:
                    right_line[0].remove()
                right_line[0] = ax.axvline(val, color='r', linestyle='--', label='Right Edge')
                print(f"Right Edge selected at index: {val}")
            fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            if selection_phase[0] == 'left' and left_point[0] is not None:
                selection_phase[0] = 'right'
                print("Now select the RIGHT edge and press Enter again.")
            elif selection_phase[0] == 'right' and right_point[0] is not None:
                plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if left_point[0] is None or right_point[0] is None:
        print("Error: Two valid boundaries not selected!")
        return

    start_idx, end_idx = sorted([left_point[0], right_point[0]])
    print(f"Selected truncation range: {start_idx} to {end_idx}")

    cropped_df = df.iloc[start_idx:end_idx + 1].copy()

    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_truncated{ext}"
    cropped_df.to_csv(output_filename, index=False, header=False)
    print(f"Truncated data saved as: {output_filename}")

    show_plot = input("\nWould you like to view the truncated result? (Y/N): ").strip().upper()
    if show_plot == 'Y':
        plt.figure(figsize=(10, 5))
        plt.plot(cropped_df[column_name].values, label=f"Column {column_name}")
        plt.title("Truncated Output")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return output_filename

def main():
    filename = input("Enter CSV file path: ").strip('"')
    df = load_data(filename)

    print("Available columns:")
    for i in range(df.shape[1]):
        print(f"{i}: Column {i}")

    col_idx = int(input("Enter the index of the column to view while truncating: "))
    downsample_factor = int(input("Enter downsample factor for display (1 = full res): ") or 1)

    plot_and_select_truncation(df, filename, col_idx, downsample_factor)

if __name__ == "__main__":
    main()
