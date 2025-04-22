import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def compute_fft(signal):
    fft_result = np.fft.fft(signal)
    power = np.abs(fft_result)
    phase = np.angle(fft_result)
    return power, phase

def save_fft_output(filename, power, phase):
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_fft.csv"
    fft_bin = np.arange(len(power))
    df_out = pd.DataFrame({
        'FFT Bin': fft_bin,
        'Power': np.round(power, 6),
        'Phase': np.round(phase, 6)
    })
    df_out.to_csv(output_filename, index=False)
    print(f"FFT results saved to: {output_filename}")
    return output_filename

def plot_fft(index, power, phase):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(index, power, label='Power Spectrum')
    plt.title("FFT Power Spectrum")
    plt.xlabel("FFT Bin")
    plt.ylabel("Power")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(index, phase, label='Phase Spectrum', color='orange')
    plt.title("FFT Phase Spectrum")
    plt.xlabel("FFT Bin")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    filename = input("Enter windowed CSV file path: ").strip('"')
    df = load_data(filename)

    print("\nFFT Input Modes:")
    print("A: Use IFFT Real + IFFT Imag (complex signal)")
    print("B: Use IFFT Real only")
    print("C: Use IFFT Imag only")
    print("D: Use IFFT Magnitude only")

    print("\nReminder:")
    print("- A: Full complex FFT (recommended) — accurate phase and frequency")
    print("- B/C: Real or Imag only — partial signal, experimental only")
    print("- D: Magnitude — shape only, no phase info")

    choice = input("\nEnter choice (A/B/C/D): ").strip().upper()

    if choice == 'A':
        if 'IFFT Real' in df.columns and 'IFFT Imag' in df.columns:
            signal = df['IFFT Real'].values + 1j * df['IFFT Imag'].values
        else:
            print("Error: Both 'IFFT Real' and 'IFFT Imag' columns are required for this option.")
            return
    elif choice == 'B':
        if 'IFFT Real' in df.columns:
            signal = df['IFFT Real'].values
        else:
            print("Error: 'IFFT Real' column not found.")
            return
    elif choice == 'C':
        if 'IFFT Imag' in df.columns:
            signal = df['IFFT Imag'].values
        else:
            print("Error: 'IFFT Imag' column not found.")
            return
    elif choice == 'D':
        if 'IFFT Magnitude' in df.columns:
            signal = df['IFFT Magnitude'].values
        else:
            print("Error: 'IFFT Magnitude' column not found.")
            return
    else:
        print("Invalid choice. Please select A, B, C, or D.")
        return

    power, phase = compute_fft(signal)
    output_file = save_fft_output(filename, power, phase)

    print("\nWould you like to plot the FFT results?")
    print("A: Yes")
    print("B: No")
    show_plot = input("Enter choice (A/B): ").strip().upper()

    if show_plot == 'A':
        fft_bin = np.arange(len(power))
        plot_fft(fft_bin, power, phase)

if __name__ == "__main__":
    main()
