import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def fft_unshift_index(index, n):
    return (index + n // 2) % n

def plot_and_collect_windows(df, column_name, num_windows, downsample_factor=1):
    y = df[column_name].values
    x = np.arange(len(y))
    n = len(y)
    fftshifted = False

    if {'IFFT Real', 'IFFT Imag', 'IFFT Magnitude'}.issubset(df.columns):
        print("IFFT data detected — applying fftshift for visualization only.")
        y = np.fft.fftshift(y)
        fftshifted = True

    y_downsampled = y[::downsample_factor]
    x_downsampled = x[::downsample_factor]
    data_length = len(y)

    window_ranges = []

    for i in range(num_windows):
        print(f"\nDefine window {i+1} of {num_windows}...")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_downsampled, y_downsampled, '-', linewidth=0.5, color='b', label=column_name)
        ax.set_title(f"Window {i+1} — Click LEFT (green), then RIGHT (red). Press Enter after each.")
        ax.set_xlabel("Index")
        ax.set_ylabel(column_name)
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

        if left_point[0] is not None and right_point[0] is not None:
            if fftshifted:
                start = fft_unshift_index(min(left_point[0], right_point[0]), n)
                end = fft_unshift_index(max(left_point[0], right_point[0]), n)
            else:
                start, end = sorted([left_point[0], right_point[0]])
            window_ranges.append((start, end))

    return window_ranges, fftshifted

def apply_zero_mask(df, cols_to_process, window_ranges):
    mask = np.zeros(len(df), dtype=bool)
    for start, end in window_ranges:
        if start <= end:
            mask[start:end+1] = True
        else:
            mask[:end+1] = True
            mask[start:] = True
    df_masked = df.copy()
    df_masked.loc[~mask, cols_to_process] = 0.0
    return df_masked

def main():
    filename = input("Enter CSV file path: ").strip('"')
    df = load_data(filename)
    downsample_factor = int(input("Enter downsample factor for display (1 = full res): ") or 1)
    num_windows = int(input("How many windows would you like to define? ") or 1)

    if {'IFFT Real', 'IFFT Imag', 'IFFT Magnitude'}.issubset(df.columns):
        column_name = 'IFFT Magnitude'
        cols_to_process = ['IFFT Real', 'IFFT Imag', 'IFFT Magnitude']
    else:
        print("Available columns:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        col_idx = int(input("Enter the index of the column to window: "))
        column_name = df.columns[col_idx]
        cols_to_process = [column_name]

    window_ranges, fftshifted = plot_and_collect_windows(df, column_name, num_windows, downsample_factor)

    masked_df = apply_zero_mask(df, cols_to_process, window_ranges)
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_zeroedout{ext}"
    masked_df.to_csv(output_filename, index=False)
    print(f"Windowed data saved as: {output_filename}")

    print("\nWould you like to view the resulting windowed signal?")
    print("A: Yes (shifted)")
    print("B: Yes (non-shifted)")
    print("C: No")
    show_plot = input("Enter choice (A/B/C): ").strip().upper()

    if show_plot in ['A', 'B']:
        plt.figure(figsize=(10, 5))
        for col in cols_to_process:
            values = masked_df[col].values
            if show_plot == 'A':
                values = np.fft.fftshift(values)
            plt.plot(values, label=col)
        title = "Windowed Signal (Shifted)" if show_plot == 'A' else "Windowed Signal (Non-Shifted)"
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
