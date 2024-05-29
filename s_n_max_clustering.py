import numpy as np

import s_n_trysomething


def max_clustering(im_response, im_fgnd_mask, r=10):
    import skimage.measure

    # find local maxima of all foreground pixels
    mval, mind = s_n_trysomething._max_clustering_cython(
        im_response, im_fgnd_mask, r
    )

    # identify connected regions of local maxima and define their seeds
    im_label = skimage.measure.label(im_fgnd_mask & (im_response == mval))

    if not np.any(im_label):
        return im_label, None, None

    # compute normalized response
    min_resp = im_response.min()
    max_resp = im_response.max()
    resp_range = max_resp - min_resp

    if resp_range == 0:
        return np.zeros_like(im_label), None, None

    im_response_nmzd = (im_response - min_resp) / resp_range

    # compute object properties
    obj_props = skimage.measure.regionprops(im_label, im_response_nmzd)

    obj_props = [prop for prop in obj_props if np.isfinite(prop.weighted_centroid).all()]

    num_labels = len(obj_props)

    if num_labels == 0:
        return im_label, None, None

    # extract object seeds
    seeds = np.array(
        [obj_props[i].weighted_centroid for i in range(num_labels)])
    seeds = np.round(seeds).astype(int)

    # fix seeds outside the object region - happens for non-convex objects
    for i in range(num_labels):

        sy = seeds[i, 0]
        sx = seeds[i, 1]

        if im_label[sy, sx] == obj_props[i].label:
            continue

        # find object point with closest manhattan distance to center of mass
        pts = obj_props[i].coords

        ydist = np.abs(pts[:, 0] - sy)
        xdist = np.abs(pts[:, 1] - sx)

        seeds[i, :] = pts[np.argmin(xdist + ydist), :]

        assert im_label[seeds[i, 0], seeds[i, 1]] == obj_props[i].label

    # get seed responses
    max_response = im_response[seeds[:, 0], seeds[:, 1]]

    # set label of each foreground pixel to the label of its nearest peak
    im_label_flat = im_label.ravel()

    pind = np.flatnonzero(im_fgnd_mask)

    mind_flat = mind.ravel()

    im_label_flat[pind] = im_label_flat[mind_flat[pind]]

    # return
    return im_label, seeds, max_response
