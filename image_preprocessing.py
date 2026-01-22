import numpy as np
import copy
import cv2
import os
import matplotlib.pyplot as plt


def pre_alignment(inputs, sample_transform, output_path=None):
    """

    :param output_path:
    :param inputs: list of data of all samples.
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :param sample_transform: Transformations for each sample, optional:
        'i'    : No transform
        '90'    : Rotate counterclockwise by 90 degrees
        '180'   : Rotate counterclockwise by 180 degrees
        '270'   : Rotate counterclockwise by 270 degrees
        'f': Flip horizontally
        'f90'   : Flip horizontally and rotate counterclockwise by 90 degrees
        'f180'  : Flip horizontally and rotate counterclockwise by 180 degrees
        'f270'  : Flip horizontally and rotate counterclockwise by 270 degrees
    :return: inputs after transformation
    :return:
    """
    assert len(inputs) == len(sample_transform), "Length of transformation should be same with the number of samples."
    inputs = copy.deepcopy(inputs)
    transform_ops = {
        'i': lambda x: x,
        '90': lambda x: np.rot90(x, k=1, axes=(1, 2)),
        '180': lambda x: np.rot90(x, k=2, axes=(1, 2)),
        '270': lambda x: np.rot90(x, k=3, axes=(1, 2)),
        'f': lambda x: x[:, :, ::-1],
        'f90': lambda x: np.rot90(x[:, :, ::-1], k=1, axes=(1, 2)),
        'f180': lambda x: np.rot90(x[:, :, ::-1], k=2, axes=(1, 2)),
        'f270': lambda x: np.rot90(x[:, :, ::-1], k=3, axes=(1, 2))
    }
    mask_transform_ops = {
        'i': lambda x: x,
        '90': lambda x: np.rot90(x, k=1),
        '180': lambda x: np.rot90(x, k=2),
        '270': lambda x: np.rot90(x, k=3),
        'f': lambda x: x[:, ::-1],
        'f90': lambda x: np.rot90(x[:, ::-1], k=1),
        'f180': lambda x: np.rot90(x[:, ::-1], k=2),
        'f270': lambda x: np.rot90(x[:, ::-1], k=3)
    }

    for s_input, s_transform in zip(inputs, sample_transform):
        if s_transform not in transform_ops:
            raise ValueError(f"Incorrect transformation: {s_transform} (Sample ID: {s_input[0]})")
        # intensity array (shape: # images, height, width)
        s_input[3] = transform_ops[s_transform](s_input[3])

        # shape mask (shape: height, width)
        s_input[1] = mask_transform_ops[s_transform](s_input[1])
        # save s_input[1] to file 'output_path/{s_input[0]}_after_transformation.png'
        # Export mask for each sample
        if output_path:
            fig, ax = plt.subplots()
            im = ax.imshow(s_input[1])
            ax.set_title(f'MSI mask (transformed) for {s_input[0]}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)
            o_path = os.path.join(output_path, f"{s_input[0]}_image_mask_transformed.png")
            fig.savefig(o_path)
            plt.close(fig)

    return inputs


def input_normalization(inputs):
    inputs = copy.deepcopy(inputs)
    for s_input in inputs:
        intensity_images = s_input[3]  # shape: # ion images, height, width
        quantiles = np.percentile(intensity_images, 99, axis=(1, 2), keepdims=True)
        clipped = np.minimum(intensity_images, quantiles)

        max_vals = np.max(clipped, axis=(1, 2), keepdims=True)
        normalized = clipped /(max_vals + 0.000001)

        s_input[3] = normalized.astype(np.float32)
    return inputs


def get_input_size(inputs, patch_size=None):
    """

    :param inputs: list of data of all samples.
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :return:
    """
    input_height, input_width = 0, 0
    for s_input in inputs:
        input_height = max(input_height, s_input[3].shape[1])
        input_width = max(input_width, s_input[3].shape[2])

    if patch_size is not None:
        def round_up(x, size):
            return ((x + size - 1) // size) * size

        input_height = round_up(input_height, patch_size)
        input_width = round_up(input_width, patch_size)
    return input_height, input_width


def resize_images(inputs, input_height, input_width):
    inputs = copy.deepcopy(inputs)

    for s_input in inputs:
        orig_intensity = s_input[3]
        N, H, W = orig_intensity.shape   # shape: # ion images, height, width
        # Use Opencv
        resized = np.empty((N, input_height, input_width), dtype=np.float32)
        for i in range(N):
            resized[i] = cv2.resize(orig_intensity[i],
                                    (input_width, input_height),
                                    interpolation=cv2.INTER_LINEAR)

        s_input[3] = resized

        s_input[1] = cv2.resize(s_input[1].astype(np.uint8),
                                (input_width, input_height),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

    return inputs


def display_mean_spectra_from_inputs(sample_input, mz_min=None, mz_max=None):
    """
    Display the mean spectrum for an input of a sample.

    :param sample_input: list
    [sample_id,
     mask: ndarray of shape (w, h), dtype=bool,
     mz: ndarray of shape (n,),
     intensity: ndarray of shape (n,w, h)]
    :param mz_min: float, optional
    :param mz_max: float, optional
    :return:
    """

    sample_id, mask, mz, intensity = sample_input
    mz_filter = np.ones_like(mz, dtype=bool)
    if mz_min is not None:
        mz_filter &= mz >= mz_min
    if mz_max is not None:
        mz_filter &= mz <= mz_max

    mz_filtered = mz[mz_filter]
    intensity_filtered = intensity[mz_filter]

    mean_intensity = []
    for i in range(intensity_filtered.shape[0]):
        masked_values = intensity_filtered[i][mask]
        mean_intensity.append(masked_values.mean())

    mean_intensity = np.asarray(mean_intensity)

    plt.figure(figsize=(8, 5))
    plt.stem(mz_filtered, mean_intensity, linefmt="k-", markerfmt=" ", basefmt=" ")
    plt.xlabel("m/z")
    plt.ylabel("Average Intensity")
    plt.title(f"Mean Spectrum (Sample: {sample_id})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_ion_image_from_inputs(sample_input, mz_min=None, mz_max=None):
    """
    Display the ion image for an input of a sample.

    :param sample_input: list
    [sample_id,
     mask: ndarray of shape (w, h), dtype=bool,
     mz: ndarray of shape (n,),
     intensity: ndarray of shape (n,w, h)]
    :param mz_min: float, optional
    :param mz_max: float, optional
    :return:
    """

    sample_id, mask, mz, intensity = sample_input

    mz_filter = np.ones_like(mz, dtype=bool)
    if mz_min is not None:
        mz_filter &= mz >= mz_min
    if mz_max is not None:
        mz_filter &= mz <= mz_max

    # mz_filtered = mz[mz_filter]
    intensity_filtered = intensity[mz_filter]
    ion_image = intensity_filtered.sum(axis=0)
    return ion_image, mask


def display_ion_image_from_inputs(sample_input, mz_min=None, mz_max=None):
    """
    Display the ion image for an input of a sample.

    :param sample_input: list
    [sample_id,
     mask: ndarray of shape (w, h), dtype=bool,
     mz: ndarray of shape (n,),
     intensity: ndarray of shape (n,w, h)]
    :param mz_min: float, optional
    :param mz_max: float, optional
    :return:
    """

    sample_id, mask, mz, intensity = sample_input

    mz_filter = np.ones_like(mz, dtype=bool)
    if mz_min is not None:
        mz_filter &= mz >= mz_min
    if mz_max is not None:
        mz_filter &= mz <= mz_max

    # mz_filtered = mz[mz_filter]
    intensity_filtered = intensity[mz_filter]
    ion_image = intensity_filtered.sum(axis=0)
    title_str = f"Summed Intensity from m/z {mz_min:.4f} to m/z {mz_max:.4f}"

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(mask,  cmap='viridis')
    axes[0].set_title("Mask")
    axes[0].axis("Off")

    im1 = axes[1].imshow(ion_image, cmap='viridis')
    axes[1].set_title(title_str)
    axes[1].axis("Off")

    fig.suptitle(f"Ion Image (Sample: {sample_id})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def rotate_image_clockwise(sample_input, angle):
    """

    :param sample_input: sample_input: list
    [sample_id,
     mask: ndarray of shape (w, h), dtype=bool,
     mz: ndarray of shape (n,),
     intensity: ndarray of shape (n,w, h)]
    :param angle: float, optional
    Clockwise rotation angle in degrees
    :return:
    """
    # Update mask and intensity array
    sample_id, mask, mz, intensity = sample_input
    n, h, w = intensity.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_mask = cv2.warpAffine(
        mask,
        M,
        (w, h),  # output size, inappropriate
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    rotated_intensity = np.empty_like(intensity)
    for i in range(n):
        rotated_intensity[i] = cv2.warpAffine(
        intensity[i],
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
        )
    return [sample_id, rotated_mask, mz, rotated_intensity]


def save_ion_image(sample_input, outpath, mz_min=None, mz_max=None):
    """
    Save ion image of specified range to numpy file

    :param mz_min: float, optional
    :param mz_max: float, optional
    :param sample_input: list
    [sample_id,
     mask: ndarray of shape (w, h), dtype=bool,
     mz: ndarray of shape (n,),
     intensity: ndarray of shape (n,w, h)]
    :return:
    """
    # 1. Compute the ion image in specified mz range
    sample_id, mask, mz, intensity = sample_input
    mz_filter = np.ones_like(mz, dtype=bool)
    if mz_min is not None:
        mz_filter &= mz >= mz_min
    if mz_max is not None:
        mz_filter &= mz <= mz_max

    mz_filtered = mz[mz_filter]
    intensity_filtered = intensity[mz_filter]
    ion_image = intensity_filtered.sum(axis=0)

    np.save(outpath, ion_image)


if __name__ == '__main__':
    import pickle
    with open('./test_mice_brain_aging/input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    inputs_after = pre_alignment(inputs, sample_transform=['i', 'i', 'i', 'i'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    for s, s_after in zip(inputs, inputs_after):
        print(s[0], s[1].shape, s[2].shape, s[3].shape, s[3].dtype)
        print(s_after[0], s_after[1].shape, s_after[2].shape, s_after[3].shape, s_after[3].dtype)

        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(s[3][100], cmap='magma')
        plt.colorbar(im1, ax=ax1)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(s_after[3][100], cmap='magma')
        plt.colorbar(im2, ax=ax2)
        plt.show()