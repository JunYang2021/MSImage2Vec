from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def mean_spectrum(msi_data, match_ppm=10):
    mean_mz = []
    mean_i = []
    num_spec = 0

    for idx, _ in enumerate(msi_data.coordinates):
        mzs, intensities = msi_data.getspectrum(idx)
        num_spec += 1
        for mz, intensity in zip(mzs, intensities):
            matched = False
            for j, mean_mz_val in enumerate(mean_mz):
                tolerance = mean_mz_val * match_ppm / 1e6
                if abs(mz - mean_mz_val) <= tolerance:
                    mean_i[j].append(intensity)
                    matched = True
                    break

            if not matched:
                mean_mz.append(mz)
                mean_i.append([intensity])

    mean_mz = np.array(mean_mz)

    i_to_remove = []
    for j, i_array in enumerate(mean_i):
        # delete the up 95% and below 5% value and then compute mean value
        i_array = np.array(i_array)
        lower_percentile = np.percentile(i_array, 5)
        upper_percentile = np.percentile(i_array, 95)

        # Filter out the extreme values
        filtered_i_array = i_array[(i_array >= lower_percentile) & (i_array <= upper_percentile)]

        if len(filtered_i_array) == 0:
            i_to_remove.append(j)
            mean_i[j] = 0
        else:
            mean_i[j] = np.mean(filtered_i_array)

    mean_mz = np.delete(mean_mz, i_to_remove)
    mean_i = np.delete(mean_i, i_to_remove)

    mean_i = np.array(mean_i)
    mean_i = mean_i / num_spec

    return mean_mz, mean_i


def correction_func(ref_file, tar_file, match_ppm=10):
    ref_data = ImzMLParser(ref_file)
    tar_data = ImzMLParser(tar_file)

    # 1. Get mean spectrum for two files
    ref_mz_array, ref_i_array = mean_spectrum(ref_data, match_ppm)  # return two numpy.array
    tar_mz_array, tar_i_array = mean_spectrum(tar_data, match_ppm)
    ref_mz_array, ref_i_array = zip(*sorted(zip(ref_mz_array, ref_i_array)))
    tar_mz_array, tar_i_array = zip(*sorted(zip(tar_mz_array, tar_i_array)))

    ref_mz_array = np.array(ref_mz_array)
    ref_i_array = np.array(ref_i_array)
    tar_mz_array = np.array(tar_mz_array)
    tar_i_array = np.array(tar_i_array)

    # 2. Compute correction function ref/tar
    matched_mz = []
    matched_i_ref = []
    matched_i_tar = []
    t = 0

    for i in range(len(ref_mz_array)):
        r_mz, r_i = ref_mz_array[i], ref_i_array[i]
        tolerance = r_mz * match_ppm / 1e6  # ppm tolerance based on reference m/z

        # Find matching m/z in target spectrum
        for j in range(t, len(tar_mz_array)):
            t_mz, t_i = tar_mz_array[j], tar_i_array[j]
            if abs(r_mz - t_mz) < tolerance:
                matched_mz.append(r_mz)
                matched_i_ref.append(r_i)
                matched_i_tar.append(t_i)
                t = j + 1  # Keep track of the last index from target data to avoid redundant matching
                break

    # Convert matched data to numpy arrays for easy manipulation
    matched_mz = np.array(matched_mz)
    matched_i_ref = np.array(matched_i_ref)
    matched_i_tar = np.array(matched_i_tar)

    return matched_mz, matched_i_ref, matched_i_tar


def non_linear_fit(mz, ref_i, tar_i, d=4):
    correction_lamda = ref_i / tar_i
    x = mz.reshape(-1, 1)
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(X_poly, correction_lamda)
    return model  # 多项式回归最稳健


def polynomial_pred(model, mz, d=4):
    x = mz.reshape(-1, 1)

    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(x)
    prediction = model.predict(X_poly)
    return prediction


if __name__ == '__main__':
    ref_file = r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m3-1-before.imzML'
    tar_file_1 = r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m3-1-end.imzML'
    tar_file_2 = r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m1-1-before.imzML'
    tar_file_3 = r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m1-1-end.imzML'
    tar_file_4 = r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m9-1-before.imzML'
    tar_file_5 = r'E:\yangjun\msi\MSI_IIE_article\aging\imzml files\m9-1-end.imzML'

    mz_pred = np.linspace(92, 1000, 2000).reshape(-1, 1)

    mz1, r1, t1 = correction_func(ref_file, tar_file_1)
    lamda1 = r1 / t1
    model1 = non_linear_fit(mz1, r1, t1)
    # lamda1_pred = model1.predict(mz_pred)
    lamda1_pred = polynomial_pred(model1, mz_pred, 4)

    mz2, r2, t2 = correction_func(ref_file, tar_file_2)
    lamda2 = r2 / t2
    model2 = non_linear_fit(mz2, r2, t2)
    # lamda2_pred = model2.predict(mz_pred)
    lamda2_pred = polynomial_pred(model2, mz_pred, 4)

    mz3, r3, t3 = correction_func(ref_file, tar_file_3)
    lamda3 = r3 / t3
    model3 = non_linear_fit(mz3, r3, t3)
    # lamda3_pred = model3.predict(mz_pred)
    lamda3_pred = polynomial_pred(model3, mz_pred, 4)

    mz4, r4, t4 = correction_func(ref_file, tar_file_4)
    lamda4 = r4 / t4
    model4 = non_linear_fit(mz4, r4, t4)
    # lamda4_pred = model4.predict(mz_pred)
    lamda4_pred = polynomial_pred(model4, mz_pred, 4)

    mz5, r5, t5 = correction_func(ref_file, tar_file_5)
    lamda5 = r5 / t5
    model5 = non_linear_fit(mz5, r5, t5)
    # lamda5_pred = model5.predict(mz_pred)
    lamda5_pred = polynomial_pred(model5, mz_pred, 4)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    plt.scatter(mz1, lamda1, label='m3-1-end', color=colors[0])
    plt.plot(mz_pred, lamda1_pred, label='m3-1-end-fit', color=colors[0], alpha=0.7)

    plt.scatter(mz2, lamda2, label='m1-1-before', color=colors[1])
    plt.plot(mz_pred, lamda2_pred, label='m1-1-before-fit', color=colors[1], alpha=0.7)

    plt.scatter(mz3, lamda3, label='m1-1-end', color=colors[2])
    plt.plot(mz_pred, lamda3_pred, label='m1-1-end-fit', color=colors[2], alpha=0.7)

    plt.scatter(mz4, lamda4, label='m9-1-before', color=colors[3])
    plt.plot(mz_pred, lamda4_pred, label='m9-1-before-fit', color=colors[3], alpha=0.7)

    plt.scatter(mz5, lamda5, label='m9-1-end', color=colors[4])
    plt.plot(mz_pred, lamda5_pred, label='m9-1-end-fit', color=colors[4], alpha=0.7)

    plt.xlabel('m/z')
    plt.ylabel('Correction factor')
    plt.legend()
    plt.grid(True)
    plt.show()
