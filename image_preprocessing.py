import numpy as np
import copy
import cv2


def pre_alignment(inputs, sample_transform):
    """

    :param inputs: list of data of all samples.
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :param sample_transform: Transformations for each sample, optional:
        'i'    : No transform
        '90'    : Rotate counterclockwise by 90 degrees
        '180'   : Rotate counterclockwise by 180 degrees
        '270'   : Rotate counterclockwise by 270 degrees
        'f': Flip
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

    return inputs


def input_normalization(inputs):
    inputs = copy.deepcopy(inputs)
    for s_input in inputs:
        intensity_images = s_input[3]  # shape: # ion images, height, width
        quantiles = np.percentile(intensity_images, 99, axis=(1, 2), keepdims=True)
        clipped = np.minimum(intensity_images, quantiles)

        max_vals = np.max(clipped, axis=(1, 2), keepdims=True)
        normalized = clipped /max_vals

        s_input[3] = normalized.astype(np.float32)
    return inputs


def get_input_size(inputs):
    """

    :param inputs: list of data of all samples.
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :return:
    """
    input_height, input_width = 0, 0
    for s_input in inputs:
        input_height = max(input_height, s_input[3].shape[1])
        input_width = max(input_width, s_input[3].shape[2])
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