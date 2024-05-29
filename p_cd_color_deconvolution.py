"""Placeholder."""
import collections

import numpy as np

from p_cd_linalg import normalize
from p_cd_complement_stain_matrix import complement_stain_matrix
import u_convert_image_to_matrix
import u_convert_matrix_to_image
import p_cc_rgb_to_sda
import p_cc_sda_to_rgb

def color_deconvolution(im_rgb, w, I_0=None):
    # complement stain matrix if needed
    if np.linalg.norm(w[:, 2]) <= 1e-16:
        wc = complement_stain_matrix(w)
    else:
        wc = w

    # normalize stains to unit-norm
    wc = normalize(wc)

    # invert stain matrix
    Q = np.linalg.pinv(wc)

    # transform 3D input image to 2D RGB matrix format
    m = u_convert_image_to_matrix.convert_image_to_matrix(im_rgb)[:3]

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = p_cc_rgb_to_sda.rgb_to_sda(m, I_0)
    sda_deconv = np.dot(Q, sda_fwd)
    sda_inv = p_cc_sda_to_rgb.sda_to_rgb(sda_deconv,
                                          255 if I_0 is not None else None)

    # reshape output
    StainsFloat = u_convert_matrix_to_image.convert_matrix_to_image(sda_inv, im_rgb.shape)

    # transform type
    Stains = StainsFloat.clip(0, 255).astype(np.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output
