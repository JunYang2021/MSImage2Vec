import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from BTrees.OOBTree import OOBTree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})
import os
import pickle
from tqdm import tqdm


class IonImage:
    def __init__(self, mz, width, height):
        self.mzimage = np.full((height, width), mz, dtype=np.float32)
        self.iimage = np.zeros((height, width), dtype=np.float32)
        self.mzmean = mz
        self.real_points = 1
        self.compound = None

    def plot(self, ax=None):
        if ax is None:
            plt.imshow(self.iimage, cmap='magma')
            plt.title('m/z: {0:.4f}'.format(self.mzmean))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar()
            # plt.gca().invert_yaxis()
            plt.show()
        else:
            im = ax.imshow(self.iimage, cmap='magma')
            ax.set_title('m/z: {0:.4f}'.format(self.mzmean))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)


def get_ion_images(msi_parser: ImzMLParser, resolution, noise_threshold, blank_pixels_percent, target_mz_list=None,
                   target_mz_name=None):
    # Ensure the length of 'target_mz_list' and 'target_mz_name' is identical
    # Max intensity of an image should be greater than noise_threshold
    coord = np.array(msi_parser.coordinates)
    width = coord[:, 0].max() - coord[:, 0].min() + 1
    height = coord[:, 1].max() - coord[:, 1].min() + 1
    x_minus = coord[:, 0].min()
    y_minus = coord[:, 1].min()
    mask = np.zeros((height, width), dtype=bool)

    L = len(msi_parser.coordinates)

    ions = OOBTree()
    for idx, (x, y, _) in enumerate(tqdm(msi_parser.coordinates, total=L, desc="Processing coordinates")):
        for m, i in zip(*msi_parser.getspectrum(idx)):
            if i > 0:
                mask[y - y_minus, x - x_minus] = True
                delta_mz = resolution / (10 ** 6) * m
                closest_mz, closest_item = None, None
                for candidate_mz, candidate_item in ions.items(min=m - delta_mz, max=m + delta_mz):
                    if closest_mz is None or abs(candidate_mz - m) < abs(closest_mz - m):
                        closest_mz, closest_item = candidate_mz, candidate_item

                if closest_mz is None:
                    ions[m] = IonImage(m, width, height)
                    ions[m].iimage[y - y_minus, x - x_minus] = i
                else:
                    if closest_item.iimage[y - y_minus, x - x_minus] == 0:
                        closest_item.mzimage[y - y_minus, x - x_minus] = m
                        closest_item.iimage[y - y_minus, x - x_minus] = i
                        closest_item.real_points += 1
                    else:
                        closest_item.mzimage[y - y_minus, x - x_minus] = (closest_item.mzimage[
                                                                              y - y_minus, x - x_minus] *
                                                                          closest_item.iimage[
                                                                              y - y_minus, x - x_minus] + m * i) / (
                                                                                 closest_item.iimage[
                                                                                     y - y_minus, x - x_minus] + i)
                        closest_item.iimage[y - y_minus, x - x_minus] = closest_item.iimage[
                                                                            y - y_minus, x - x_minus] + i

    total_pixels = idx + 1
    final_ions = []
    for mz, ion in ions.items():
        if ion.real_points >= total_pixels * blank_pixels_percent and np.max(ion.iimage) >= noise_threshold:
            ion.mzmean = np.sum(ion.mzimage * ion.iimage) / np.sum(ion.iimage)
            final_ions.append(ion)

    if target_mz_list:
        for target_mz, target_name in zip(target_mz_list, target_mz_name):
            delta_mz = resolution / 1e6 * target_mz
            matched = False
            for ion in final_ions:
                if abs(ion.mzmean - target_mz) <= delta_mz:
                    ion.compound = target_name
                    matched = True
                    break
            if not matched:
                tar_ion_image = IonImage(target_mz, width, height)
                for idx, (x, y, _) in enumerate(msi_parser.coordinates):
                    for m, i in zip(*msi_parser.getspectrum(idx)):
                        if target_mz - delta_mz < m < target_mz + delta_mz:
                            if tar_ion_image.iimage[y - y_minus, x - x_minus] == 0:
                                tar_ion_image.mzimage[y - y_minus, x - x_minus] = m
                                tar_ion_image.iimage[y - y_minus, x - x_minus] = i
                                tar_ion_image.real_points += 1
                            else:
                                tar_ion_image.mzimage[y - y_minus, x - x_minus] = (tar_ion_image.mzimage[
                                                                                       y - y_minus, x - x_minus] *
                                                                                   tar_ion_image.iimage[
                                                                                       y - y_minus, x - x_minus] + m * i) / (
                                                                                          tar_ion_image.iimage[
                                                                                              y - y_minus, x - x_minus] + i)
                                tar_ion_image.iimage[y - y_minus, x - x_minus] = tar_ion_image.iimage[
                                                                                     y - y_minus, x - x_minus] + i
                tar_ion_image.mzmean = np.sum(tar_ion_image.mzimage * tar_ion_image.iimage) / np.sum(
                    tar_ion_image.iimage)
                tar_ion_image.compound = target_name
                print(tar_ion_image.iimage.max())
                if tar_ion_image.iimage.max() > 0:
                    final_ions.append(tar_ion_image)

    return final_ions, mask


def extract_ion_images(msi_parser: ImzMLParser, target_mz, resolution):
    # Max intensity of an image should be greater than noise_threshold
    coord = np.array(msi_parser.coordinates)
    width = coord[:, 0].max() - coord[:, 0].min() + 1
    height = coord[:, 1].max() - coord[:, 1].min() + 1
    x_minus = coord[:, 0].min()
    y_minus = coord[:, 1].min()
    mask = np.zeros((height, width), dtype=bool)

    L = len(msi_parser.coordinates)

    tar_ion_image = IonImage(target_mz, width, height)
    delta_mz = resolution / (10 ** 6) * target_mz
    for idx, (x, y, _) in enumerate(tqdm(msi_parser.coordinates, total=L, desc="Processing coordinates")):
        for m, i in zip(*msi_parser.getspectrum(idx)):
            if target_mz - delta_mz < m < target_mz + delta_mz:
                if tar_ion_image.iimage[y - y_minus, x - x_minus] == 0:
                    tar_ion_image.mzimage[y - y_minus, x - x_minus] = m
                    tar_ion_image.iimage[y - y_minus, x - x_minus] = i
                    tar_ion_image.real_points += 1
                else:
                    tar_ion_image.mzimage[y - y_minus, x - x_minus] = (tar_ion_image.mzimage[y - y_minus, x - x_minus] *
                                                                       tar_ion_image.iimage[
                                                                           y - y_minus, x - x_minus] + m * i) / (
                                                                              tar_ion_image.iimage[
                                                                                  y - y_minus, x - x_minus] + i)
                    tar_ion_image.iimage[y - y_minus, x - x_minus] = tar_ion_image.iimage[y - y_minus, x - x_minus] + i
    tar_ion_image.mzmean = np.sum(tar_ion_image.mzimage * tar_ion_image.iimage) / np.sum(tar_ion_image.iimage)
    return tar_ion_image


def get_samples_ion_images(sample_path_list, sample_id_list, ppm_torelance, noise_threshold, blank_pixels_percent,
                           output_directory, target_mz_list=None, target_mz_name=None):
    """

    :param output_directory: Directory to save the detected ion images and display the previous five images
    :param sample_path_list: list of samples files
    :param sample_id_list:  list of samples ids
    :param ppm_torelance: mass tolerance in parts per million
    :param noise_threshold: intensity threshold for noise filtering
    :param blank_pixels_percent: percentage of blank pixels allowed
    :return:
    """
    os.makedirs(output_directory, exist_ok=True)

    n_samples = len(sample_path_list)
    if isinstance(noise_threshold, (list, tuple, np.ndarray)):
        if len(noise_threshold) != n_samples:
            raise ValueError(
                f"Length of noise_threshold ({len(noise_threshold)}) must match "
                f"number of samples ({n_samples})."
            )
        noise_threshold_list = list(noise_threshold)
    else:
        noise_threshold_list = [noise_threshold] * n_samples
    inputs = []
    for k, (sample_id, sample_path) in enumerate(zip(sample_id_list, sample_path_list)):
        print(f'Obtaining ion images from {sample_id}...')
        msi_data = ImzMLParser(sample_path)
        if target_mz_list:
            ions, msi_mask = get_ion_images(msi_data, resolution=ppm_torelance,
                                            noise_threshold=noise_threshold_list[k], blank_pixels_percent=blank_pixels_percent,
                                            target_mz_list=target_mz_list[k], target_mz_name=target_mz_name[k])
        else:
            ions, msi_mask = get_ion_images(msi_data, resolution=ppm_torelance,
                                            noise_threshold=noise_threshold_list[k], blank_pixels_percent=blank_pixels_percent)

        # Export previous five pictures for each sample
        for i in range(min(5, len(ions))):
            fig, ax = plt.subplots()
            im = ax.imshow(ions[i].iimage, cmap='magma')
            ax.set_title(f'm/z: {ions[i].mzmean:.4f} name: {ions[i].compound}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)

            # Save the plot to output directory
            output_path = os.path.join(output_directory, f"{sample_id}_{i}.png")
            fig.savefig(output_path)
            plt.close(fig)

        # Export mask for each sample
        fig, ax = plt.subplots()
        im = ax.imshow(msi_mask)
        ax.set_title(f'MSI mask for {sample_id}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        output_path = os.path.join(output_directory, f"{sample_id}_image_mask.png")
        fig.savefig(output_path)
        plt.close(fig)

        # Save inputs to a file in output_directory
        mz_array = np.array([i.mzmean for i in ions])
        intensity_array = np.array([i.iimage for i in ions])  # shape: # images, height, width
        print(msi_mask.shape, mz_array.shape, intensity_array.shape)
        inputs.append([sample_id, msi_mask, mz_array, intensity_array])

    inputs_output_path = os.path.join(output_directory, "input_data.pkl")
    with open(inputs_output_path, 'wb') as f:
        pickle.dump(inputs, f)


def extract_sample_ion_image(sample_path_list, sample_id_list, target_mz, ppm_torelance, shape_mask_list,
                             correction_list=[1, 1, 1], rot_time_list=[0, 0, 0], output_path=None):
    num_samples = len(sample_path_list)
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))

    # Flatten axes array for easy iteration
    if num_samples > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    cmap = plt.cm.magma
    cmap = cmap.copy()
    cmap.set_bad(color='black')
    for i, (sample_id, sample_path) in enumerate(zip(sample_id_list, sample_path_list)):
        print(f'Obtaining ion images from {sample_id}...')
        msi_data = ImzMLParser(sample_path)
        tar_image = extract_ion_images(msi_data, target_mz, ppm_torelance)
        mat = np.where(shape_mask_list[i] == 0, np.nan, tar_image.iimage)
        mat = np.rot90(mat * correction_list[i], k=rot_time_list[i])
        print(np.nanmax(mat))
        threshold = np.percentile(mat[mat > 0], 95)
        print(threshold)
        mat_clipped = np.clip(mat, None, threshold)
        im = axes[i].imshow(mat_clipped, cmap=cmap, vmin=np.nanmin(mat_clipped), vmax=threshold)
        # axes[i].set_title(f'{sample_id} m/z: {target_mz:.4f}')
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.026, pad=0.04)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(formatter)
        axes[i].axis('off')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    r"""sample_path_list = [r'D:\Experiments\MSI\kunming\合作项目-张登峰-空间代谢组-项目报告\合作项目-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M14d-pos.imzML',
                        r'D:\Experiments\MSI\kunming\合作项目-张登峰-空间代谢组-项目报告\合作项目-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M14d-neg.imzML',
                        r'D:\Experiments\MSI\kunming\合作项目-张登峰-空间代谢组-项目报告\合作项目-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M3m-pos.imzML',
                        r'D:\Experiments\MSI\kunming\合作项目-张登峰-空间代谢组-项目报告\合作项目-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M3m-neg.imzML'
                        ]
    sample_id_list = ['14d_pos', '14d_neg', '3m_pos', '3m_neg']
    output_dir = './test_mice_brain_aging'

    get_samples_ion_images(sample_path_list=sample_path_list,
                           sample_id_list=sample_id_list,
                           ppm_torelance=10,
                           noise_threshold=100,
                           blank_pixels_percent=0.2,
                           output_directory=output_dir)"""

    # Evaluation 1: Integration of different samples
    # sample_path_list = [
    #     r'E:\yangjun\msi\ad_msi\M14d-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\M1m-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\M2m-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\M3m-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\M1_5m-pos.imzML'
    # ]
    # sample_path_list = [
    #     r'E:\yangjun\msi\ad_msi\raw_data\M14d-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\raw_data\M1m-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\raw_data\M2m-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\raw_data\M3m-pos.imzML',
    #     r'E:\yangjun\msi\ad_msi\raw_data\M1_5m-pos.imzML'
    # ]
    # sample_id_list = ['14d_pos', '1m_pos', '2m_pos', '3m_pos', '5m_pos']
    # output_dir = r'E:\yangjun\msi\MSI_IIE_article\test_five_samples'

    # get_samples_ion_images(sample_path_list=sample_path_list,
    #                        sample_id_list=sample_id_list,
    #                        ppm_torelance=10,
    #                        noise_threshold=100,
    #                        blank_pixels_percent=0.3,
    #                        output_directory=output_dir)
    # extract_sample_ion_image(sample_path_list=sample_path_list,
    #                          sample_id_list=sample_id_list,
    #                          target_mz=369.34424,
    #                          ppm_torelance=10)

    # Evaluation 2: Integration of different ion modes
    # sample_path_list = [
    #     r'E:\yangjun\msi\msi_open_data\metaspace-mcf\2025_03_18_mcf_pos-total ion count.imzML',
    #     r'E:\yangjun\msi\msi_open_data\metaspace-mcf\2025_03_17_mcf_neg_lpval-total ion count.imzML'
    # ]
    # sample_id_list = ['mcf-pos', 'mcf-neg']
    # output_dir = r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg'

    # get_samples_ion_images(sample_path_list=sample_path_list,
    #                        sample_id_list=sample_id_list,
    #                        ppm_torelance=10,
    #                        noise_threshold=500,
    #                        blank_pixels_percent=0.3,
    #                        output_directory=output_dir)
    # extract_sample_ion_image(sample_path_list=sample_path_list,
    #                          sample_id_list=sample_id_list,
    #                          target_mz=516.2848,
    #                          ppm_torelance=5)

    # Evaluation 3: Application on large-scale imaging experiments
    r"""sample_path_list = [
        r'E:\yangjun\msi\ad_msi\raw_data\M14d-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M2m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M3m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1_5m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P14d-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P2m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P3m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1_5m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M14d-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M2m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M3m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1_5m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P14d-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P2m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P3m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1_5m-neg.imzML'
    ]

    sample_id_list = ['M_14d_pos', 'M_1m_pos', 'M_2m_pos', 'M_3m_pos', 'M_5m_pos',
                      'P_14d_pos', 'P_1m_pos', 'P_2m_pos', 'P_3m_pos', 'P_5m_pos',
                      'M_14d_neg', 'M_1m_neg', 'M_2m_neg', 'M_3m_neg', 'M_5m_neg',
                      'P_14d_neg', 'P_1m_neg', 'P_2m_neg', 'P_3m_neg', 'P_5m_neg'
                      ]
    output_dir = r'E:\yangjun\msi\MSI_IIE_article\ad_study'

    import pandas as pd
    identified_pos = pd.read_excel(r'E:\yangjun\msi\ad_msi\Qualitative.xlsx', sheet_name='pos')
    pos_mz = identified_pos['mz'].astype(float).tolist()
    pos_name = identified_pos['Metabolites'].astype(str).tolist()
    identified_neg = pd.read_excel(r'E:\yangjun\msi\ad_msi\Qualitative.xlsx', sheet_name='neg')
    neg_mz = identified_neg['mz'].astype(float).tolist()
    neg_name = identified_neg['Metabolites'].astype(str).tolist()
    get_samples_ion_images(sample_path_list=sample_path_list,
                           sample_id_list=sample_id_list,
                           ppm_torelance=10,
                           noise_threshold=100,
                           blank_pixels_percent=0.3,
                           output_directory=output_dir,
                           target_mz_list=[pos_mz] * 10 + [neg_mz] * 10,
                           target_mz_name=[pos_name] * 10 + [neg_name] * 10)"""

    r"""# Extraction of specific ions in positive mode
    sample_path_list = [
        r'E:\yangjun\msi\ad_msi\raw_data\M14d-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M2m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M3m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1_5m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P14d-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P2m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P3m-pos.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1_5m-pos.imzML']

    sample_id_list = ['M_14d_pos', 'M_1m_pos', 'M_2m_pos', 'M_3m_pos', 'M_5m_pos',
                      'P_14d_pos', 'P_1m_pos', 'P_2m_pos', 'P_3m_pos', 'P_5m_pos']

    extract_sample_ion_image(sample_path_list=sample_path_list,
                             sample_id_list=sample_id_list,
                             target_mz=169.98393,
                             ppm_torelance=10)"""

    r"""# Extraction of specific ions in negative mode
    sample_path_list = [
        r'E:\yangjun\msi\ad_msi\raw_data\M14d-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M2m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M3m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\M1_5m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P14d-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P2m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P3m-neg.imzML',
        r'E:\yangjun\msi\ad_msi\raw_data\P1_5m-neg.imzML']

    sample_id_list = ['M_14d_neg', 'M_1m_neg', 'M_2m_neg', 'M_3m_neg', 'M_5m_neg',
                      'P_14d_neg', 'P_1m_neg', 'P_2m_neg', 'P_3m_neg', 'P_5m_neg']

    extract_sample_ion_image(sample_path_list=sample_path_list,
                             sample_id_list=sample_id_list,
                             target_mz=733.4954,
                             ppm_torelance=10)"""

    r"""# Evaluation 4: Application on maldi imaging experiments
    sample_path_list = [
        r'E:\yangjun\msi\maldi_pre_exp_20250709\brain_tissue_chca_maldift_20250710_centroid.imzML',
        r'E:\yangjun\msi\maldi_pre_exp_20250709\brain_tissue_chca_maldift_20250710_feature_list.imzML',
        r'E:\yangjun\msi\maldi_pre_exp_20250709\brain_tissue_nedc_maldift_20250710_centroid.imzML',
        r'E:\yangjun\msi\maldi_pre_exp_20250709\brain_tissue_nedc_maldift_20250710_feature_list.imzML'
    ]

    sample_id_list = ['chca-centroid', 'chca-features', 'nedc-centroid', 'nedc-features']
    output_dir = r'E:\yangjun\msi\MSI_IIE_article\maldi_pre_experiment'

    get_samples_ion_images(sample_path_list=sample_path_list,
                           sample_id_list=sample_id_list,
                           ppm_torelance=10,
                           noise_threshold=100,
                           blank_pixels_percent=0.3,
                           output_directory=output_dir)"""

    # Evaluation 5: Application on Ouyi whole brain 2 months
    sample_path_list = [
        r'G:\ad_msi_data_ouyi\LM2025M1005W-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M2M-neg.imzML',
        r'G:\ad_msi_data_ouyi\LM2025M1005W-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M2M-pos.imzML',
        r'G:\ad_msi_data_ouyi\LM2025M1005W-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\P2M-neg.imzML',
        r'G:\ad_msi_data_ouyi\LM2025M1005W-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\P2M-pos.imzML'
    ]

    sample_id_list = ['M2M-neg', 'M2M-pos', 'P2M-neg', 'P2M-pos']
    output_dir = r'E:\yangjun\msi\MSI_IIE_article\ad_whole_brain_pre'

    import pandas as pd
    identified_pos = pd.read_excel(r'G:\ad_msi_data_ouyi\LM2025M1005W-张登峰-空间代谢组-项目报告\2.定性结果\Qualitative.xlsx', sheet_name='pos')
    pos_mz = identified_pos['mz'].astype(float).tolist()
    pos_name = identified_pos['Metabolites'].astype(str).tolist()
    identified_neg = pd.read_excel(r'G:\ad_msi_data_ouyi\LM2025M1005W-张登峰-空间代谢组-项目报告\2.定性结果\Qualitative.xlsx', sheet_name='neg')
    neg_mz = identified_neg['mz'].astype(float).tolist()
    neg_name = identified_neg['Metabolites'].astype(str).tolist()
    get_samples_ion_images(sample_path_list=sample_path_list,
                           sample_id_list=sample_id_list,
                           ppm_torelance=10,
                           noise_threshold=50,
                           blank_pixels_percent=0.3,
                           output_directory=output_dir,
                           target_mz_list=[neg_mz, pos_mz, neg_mz, pos_mz],
                           target_mz_name=[neg_name, pos_name, neg_name, pos_name])

