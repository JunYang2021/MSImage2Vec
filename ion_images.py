import numpy as np
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from BTrees.OOBTree import OOBTree
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm


class IonImage:
    def __init__(self, mz, width, height):
        self.mzimage = np.full((height, width), mz, dtype=np.float32)
        self.iimage = np.zeros((height, width), dtype=np.float32)
        self.mzmean = mz
        self.real_points = 1

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


def get_ion_images(msi_parser: ImzMLParser, resolution, noise_threshold, blank_pixels_percent):
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
                        closest_item.mzimage[y - y_minus, x - x_minus] = (closest_item.mzimage[y - y_minus, x - x_minus] * closest_item.iimage[
                            y - y_minus, x - x_minus] + m * i) / (closest_item.iimage[y - y_minus, x - x_minus] + i)
                        closest_item.iimage[y - y_minus, x - x_minus] = closest_item.iimage[y - y_minus, x - x_minus] + i

    total_pixels = idx + 1
    final_ions = []
    for mz, ion in ions.items():
        if ion.real_points >= total_pixels * blank_pixels_percent and np.max(ion.iimage) >= noise_threshold:
            ion.mzmean = np.sum(ion.mzimage * ion.iimage) / np.sum(ion.iimage)
            final_ions.append(ion)
    return final_ions, mask


def get_samples_ion_images(sample_path_list, sample_id_list, ppm_torelance, noise_threshold, blank_pixels_percent,
                           output_directory):
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
    inputs = []
    for sample_id, sample_path in zip(sample_id_list, sample_path_list):
        print(f'Obtaining ion images from {sample_id}...')
        msi_data = ImzMLParser(sample_path)
        ions, msi_mask = get_ion_images(msi_data, resolution=ppm_torelance,
                                        noise_threshold=noise_threshold, blank_pixels_percent=blank_pixels_percent)

        # Export previous five pictures for each sample
        for i in range(min(5, len(ions))):
            fig, ax = plt.subplots()
            im = ax.imshow(ions[i].iimage, cmap='magma')
            ax.set_title(f'm/z: {ions[i].mzmean:.4f}')
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
        intensity_array = np.array([i.iimage for i in ions])   # shape: # images, height, width
        print(msi_mask.shape, mz_array.shape, intensity_array.shape)
        inputs.append([sample_id, msi_mask, mz_array, intensity_array])

    inputs_output_path = os.path.join(output_directory, "input_data.pkl")
    with open(inputs_output_path, 'wb') as f:
        pickle.dump(inputs, f)


if __name__ == '__main__':
    sample_path_list = [r'D:\Experiments\MSI\kunming\合作项目-张登峰-空间代谢组-项目报告\合作项目-张登峰-空间代谢组-项目报告\4.选区分析文件\MSiReader选区文件\M14d-pos.imzML',
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
                           output_directory=output_dir)

