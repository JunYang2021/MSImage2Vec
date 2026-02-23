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


def display_xic(p, mz):
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


def display_spectrum(p, x, y, mzmin=None, mzmax=None):
    for idx, (xx, yy, _) in enumerate(p.coordinates):
        if xx == x and yy == y:
            mz, i = p.getspectrum(idx)
            print("Length of spectrum point:", mz.shape)

            # Apply m/z filtering if needed
            if mzmin is not None or mzmax is not None:
                mask = np.ones_like(mz, dtype=bool)
                if mzmin is not None:
                    mask &= mz >= mzmin
                if mzmax is not None:
                    mask &= mz <= mzmax
                mz_plot = mz[mask]
                i_plot = i[mask]
            else:
                mz_plot = mz
                i_plot = i

            plt.figure(figsize=(8, 5))
            plt.stem(
                mz_plot,
                i_plot,
                linefmt='r-',
                markerfmt='ro',
                basefmt='r-'
            )
            plt.title('Mass Spectrum')
            plt.xlabel('m/z')
            plt.ylabel('Intensity')
            plt.grid(True)
            plt.show()
            # return  # stop after finding the matching (x, y)



if __name__ == '__main__':
    r'''p = read_file(r'E:\yangjun\msi\msi_open_data\metaspace-mcf\2025_03_18_mcf_pos-total ion count.imzML')
    # display_tic(p)
    display_spectrum(p, 25, 51)
    display_spectrum(p, 38, 51)
    display_spectrum(p, 100, 100)'''

    # p = read_file(r'F:\\msi_data\\yj-20250605-chca-dut\\yj-20250605-chca-dut.imzML')
    # display_tic(p)
    # display_spectrum(p, 1358, 263)

    # p = read_file(r'F:\\msi_data\\yj-20250605-chca-dut\\yj-20250605-chca-dut-from-disk.imzML')
    # # display_tic(p)
    # display_spectrum(p, 1358, 263)

    # p = read_file(r'F:\\msi_data\\yj-20250605-chca-dicp\\yj-20250605-chca-dicp.imzML')
    # # display_tic(p)
    # display_spectrum(p, 1366, 291)
    # display_spectrum(p, 1343, 368)

    # p = read_file(r'E:\yangjun\msi\maldi_pre_exp_20250709\brain_tissue_chca_maldift_20250710_centroid.imzML')
    # display_tic(p)
    # display_spectrum(p, 40, 50)

    # p = read_file(r'E:\yangjun\msi\maldi_pre_exp_20250709\brain_tissue_chca_maldift_20250710_feature_list.imzML')
    # display_tic(p)
    # display_spectrum(p, 40, 50)

    p = read_file(r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m3-brain-complete.imzML')
    display_tic(p)
    # display_spectrum(p, 75, 140)