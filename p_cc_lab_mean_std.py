import numpy as np

from p_cc_rgb_to_lab import rgb_to_lab


def lab_mean_std(im_input, mask_out=None):
    im_lab = rgb_to_lab(im_input)

    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))

    mean_lab = np.array([im_lab[..., i].mean() for i in range(3)])
    std_lab = np.array([im_lab[..., i].std() for i in range(3)])

    return mean_lab, std_lab
