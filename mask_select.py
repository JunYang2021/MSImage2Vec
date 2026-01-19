# Interactive Polygon Mask Tool
import argparse
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from skimage.draw import polygon


class InteractiveMask:
    def __init__(self, image):
        self.image = image
        self.mask = np.zeros(image.shape, dtype=bool)
        self.verts = None
        self.completed = False

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(image, cmap='viridis')
        self.ax.set_title("Draw polygon (double-click to close), then close window")

        self.selector = PolygonSelector(self.ax, self.on_select, useblit=True)

    def on_select(self, verts):
        self.verts = np.asarray(verts)
        print(self.verts)
        self.completed = True

    def run(self):
        plt.show()
        if not self.completed:
            raise RuntimeError("Region selection was not completed.")

        rr, cc = polygon(self.verts[:, 1], self.verts[:, 0], self.image.shape)
        self.mask[rr, cc] = True
        return self.mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to ion image (.npy)")
    parser.add_argument("--out", required=True, help="Output mask path (.npy)")
    args = parser.parse_args()

    image = np.load(args.image)
    selector = InteractiveMask(image)
    mask = selector.run()

    np.save(args.out, mask)

if __name__ == "__main__":
    main()



# def interactive_remask(sample_input, mz_min=None, mz_max=None):
#     """
#     Mask and rotate the sample ion images manually to get better correlation label.
#
#     :param mz_min: float, optional
#     :param mz_max: float, optional
#     :param sample_input: list
#     [sample_id,
#      mask: ndarray of shape (w, h), dtype=bool,
#      mz: ndarray of shape (n,),
#      intensity: ndarray of shape (n,w, h)]
#     :return:
#     """
#     # 1. Compute the ion image in specified mz range
#     sample_id, mask, mz, intensity = sample_input
#     mz_filter = np.ones_like(mz, dtype=bool)
#     if mz_min is not None:
#         mz_filter &= mz >= mz_min
#     if mz_max is not None:
#         mz_filter &= mz <= mz_max
#
#     mz_filtered = mz[mz_filter]
#     intensity_filtered = intensity[mz_filter]
#     ion_image = intensity_filtered.sum(axis=0)
#     # 2. Interactive selection
#     selector = InteractiveMask(ion_image)
#     # plt.show()
#     # return selector
#     new_mask = selector.run()
#     #
#     return [sample_id, new_mask, mz, intensity]