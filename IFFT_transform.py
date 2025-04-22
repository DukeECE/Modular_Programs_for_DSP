import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def read_crossing_csv(filename):
    """Read zero-crossing data and extract the measurement signal."""
    df = pd.read_csv(filename)
    return df['Measurement at Zero-Crossing'].values, df

def perform_ifft(signal):
    """
    Perform IFFT on the signal.
    Returns real, imaginary, and magnitude components.
    """
    transformed = np.fft.ifft(signal)
    real_part = np.real(transformed)
    imag_part = np.imag(transformed)
    magnitude = np.abs(transformed)
    return real_part, imag_part, magnitude

def save_ifft_output(filename, real, imag, magnitude):
    """Save IFFT output with consistent columns: Real, Imag, Magnitude."""
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_ifft.csv"

    df_out = pd.DataFrame({
        'IFFT Real': np.round(real, 6),
        'IFFT Imag': np.round(imag, 6),
        'IFFT Magnitude': np.round(magnitude, 6)
    })

    df_out.to_csv(output_filename, index=False)
    print(f"Saved IFFT results to {output_filename}")

def plot_results(original_signal, real, imag, magnitude, plot_mode, mode_label):
    # Apply fftshift for visual clarity only
    shifted_real = np.fft.fftshift(real)
    shifted_imag = np.fft.fftshift(imag)
    shifted_magnitude = np.fft.fftshift(magnitude)

    plt.figure(figsize=(10, 6))

    if plot_mode == 'A':
        plt.subplot(2, 1, 1)
        plt.plot(original_signal, label="Input Signal")
        plt.title("Original Measurement at Zero Crossings")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        if mode_label == 'Real Only':
            plt.plot(shifted_real, label="IFFT Output (Real Only)")
        elif mode_label == 'Magnitude Only':
            plt.plot(shifted_magnitude, label="IFFT Output (Magnitude Only)")
        elif mode_label == 'Complex (Real & Imag)':
            plt.plot(shifted_real, label="IFFT Real")
            plt.plot(shifted_imag, label="IFFT Imag")
        plt.title("IFFT Output")
        plt.grid(True)
        plt.legend()

    elif plot_mode == 'B':
        if mode_label == 'Real Only':
            plt.plot(shifted_real, label="IFFT Output (Real Only)")
        elif mode_label == 'Magnitude Only':
            plt.plot(shifted_magnitude, label="IFFT Output (Magnitude Only)")
        elif mode_label == 'Complex (Real & Imag)':
            plt.plot(shifted_real, label="IFFT Real")
            plt.plot(shifted_imag, label="IFFT Imag")
        plt.title("IFFT Output Only")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    filename = input("Enter CSV file path: ").strip('"')

    print("Choose imaginary data handling:")
    print("A: Keep complex values")
    print("B: Drop imaginary part")
    print("C: Use magnitude of complex result")
    mode_choice = input("Enter choice (A/B/C): ").strip().upper()

    if mode_choice == 'A':
        mode_label = 'Complex (Real & Imag)'
    elif mode_choice == 'B':
        mode_label = 'Real Only'
    elif mode_choice == 'C':
        mode_label = 'Magnitude Only'
    else:
        print("Invalid choice. Defaulting to Real Only.")
        mode_choice = 'B'
        mode_label = 'Real Only'

    print("Choose plot option:")
    print("A: Plot before and after processing")
    print("B: Plot only IFFT output")
    plot_mode = input("Enter choice (A/B): ").strip().upper()

    signal, full_df = read_crossing_csv(filename)
    real, imag, magnitude = perform_ifft(signal)
    save_ifft_output(filename, real, imag, magnitude)

    plot_results(signal, real, imag, magnitude, plot_mode, mode_label)

if __name__ == "__main__":
    main()
