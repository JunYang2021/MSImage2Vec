from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
import numpy as np


def read_file(file_path):
    p = ImzMLParser(file_path)
    return p


def display_tic(p):
    x_ = []
    y_ = []
    total_i = []
    idx = 0
    for idx, (x, y, _) in enumerate(p.coordinates):
        x_.append(x)
        y_.append(y)
        _, intensities = p.getspectrum(idx)
        total_i.append(sum(intensities))
    print("Total number of pixels: ", idx + 1)
    print("x axis range: ", min(x_), max(x_))
    print("y axis range: ", min(y_), max(y_))
    x_range = max(x_) - min(x_) + 1
    y_range = max(y_) - min(y_) + 1
    tic_matrix = np.zeros((y_range, x_range))

    # Populate the matrix with intensities
    for x, y, intensity in zip(x_, y_, total_i):
        tic_matrix[y - min(y_), x - min(x_)] = intensity

    # Plotting the matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(tic_matrix, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label="Total Intensity (TIC)")
    plt.title("TIC Image")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.show()


def display_spectrum(p, x, y):
    for idx, (xx, yy, _) in enumerate(p.coordinates):
        if xx == x and yy == y:
            mz, i = p.getspectrum(idx)
            print("Length of spectrum point: ", mz.shape)
            plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
            plt.stem(mz, i, linefmt='r-', markerfmt='ro', basefmt='r-')  # Plot with red lines and circles
            plt.title('Mass Spectrum')
            plt.xlabel('m/z')
            plt.ylabel('Intensity')
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    p = read_file(r'E:\yangjun\msi\msi_open_data\metaspace-mcf\2025_03_18_mcf_pos-total ion count.imzML')
    # display_tic(p)
    display_spectrum(p, 25, 51)
    display_spectrum(p, 38, 51)
    display_spectrum(p, 100, 100)