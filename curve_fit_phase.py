import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def taylor_series(w, beta0, beta1, beta2, w0):
    return beta0 + beta1 * (w - w0) + 0.5 * beta2 * (w - w0)**2

def select_window(df, x_col, y_col):
    x = df[x_col].values
    y = np.unwrap(df[y_col].values)
    data_length = len(x)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, '-', linewidth=0.8, label='Unwrapped Phase')
    ax.set_title("Click to select LEFT (green) and RIGHT (red) edges. Press Enter to confirm each.")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col + " (unwrapped)")
    ax.grid(True)
    ax.legend()

    left_idx = [None]
    right_idx = [None]
    left_line = [None]
    right_line = [None]
    phase = ['left']

    def on_click(event):
        if event.xdata is not None:
            val = event.xdata
            if phase[0] == 'left':
                left_idx[0] = val
                if left_line[0] is not None:
                    left_line[0].remove()
                left_line[0] = ax.axvline(val, color='g', linestyle='--', label='Left Edge')
                print(f"Left edge selected at {val:.2f}")
            elif phase[0] == 'right':
                right_idx[0] = val
                if right_line[0] is not None:
                    right_line[0].remove()
                right_line[0] = ax.axvline(val, color='r', linestyle='--', label='Right Edge')
                print(f"Right edge selected at {val:.2f}")
            fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            if phase[0] == 'left' and left_idx[0] is not None:
                phase[0] = 'right'
                print("Now select the RIGHT edge and press Enter again.")
            elif phase[0] == 'right' and right_idx[0] is not None:
                plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if left_idx[0] is None or right_idx[0] is None:
        print("Window selection failed. Exiting.")
        return None

    x_min, x_max = sorted([left_idx[0], right_idx[0]])
    windowed_df = df[(df[x_col] >= x_min) & (df[x_col] <= x_max)].copy()
    windowed_df[y_col] = np.unwrap(windowed_df[y_col].values)
    print(f"Windowed range: {x_min:.2f} to {x_max:.2f}")
    return windowed_df

def fit_phase_curve(df, x_col, y_col):
    x = df[x_col].values
    y = df[y_col].values
    w0 = x.mean()
    popt, pcov = curve_fit(lambda w, b0, b1, b2: taylor_series(w, b0, b1, b2, w0), x, y)
    beta0, beta1, beta2 = popt
    print("\nFitted Coefficients:")
    print(f"Beta_0: {beta0:.6f}")
    print(f"Beta_1: {beta1:.6f}")
    print(f"Beta_2: {beta2:.6f}")
    print(f"w0 (center freq): {w0:.6f}")
    return beta0, beta1, beta2, w0

def plot_fit(df, x_col, y_col, beta0, beta1, beta2, w0):
    x = df[x_col].values
    y = df[y_col].values
    y_fit = taylor_series(x, beta0, beta1, beta2, w0)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Unwrapped Phase', marker='o', linestyle='-', alpha=0.6)
    plt.plot(x, y_fit, label='Fitted Curve', color='red', linewidth=2)
    plt.title("Taylor Series Fit to Unwrapped Phase Data")
    plt.xlabel(x_col)
    plt.ylabel(y_col + " (unwrapped)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    filename = input("Enter FFT CSV file path (e.g., *_fft.csv): ").strip('"')
    df = load_data(filename)

    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    x_idx = int(input("Select the index for the x-axis column (e.g., FFT Bin): "))
    y_idx = int(input("Select the index for the y-axis column (e.g., Phase): "))
    x_col = df.columns[x_idx]
    y_col = df.columns[y_idx]

    windowed_df = select_window(df, x_col, y_col)
    if windowed_df is None:
        return

    beta0, beta1, beta2, w0 = fit_phase_curve(windowed_df, x_col, y_col)

    print("\nWould you like to view the fit?")
    print("A: Yes")
    print("B: No")
    view_fit = input("Enter choice (A/B): ").strip().upper()

    if view_fit == 'A':
        plot_fit(windowed_df, x_col, y_col, beta0, beta1, beta2, w0)

    print("\nFitted Coefficients (repeated for reference):")
    print(f"Beta_0: {beta0:.6f}")
    print(f"Beta_1: {beta1:.6f}")
    print(f"Beta_2: {beta2:.6f}")
    print(f"w0 (center freq): {w0:.6f}")

if __name__ == "__main__":
    main()
