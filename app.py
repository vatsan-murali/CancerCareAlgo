from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy as sp
import skimage.io
import skimage.measure
import skimage.color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

titlesize = 24

app = Flask(__name__)
CORS(app, origins="https://cancerclient.onrender.com")

@app.route('/api/reinhardt-normalization', methods=['POST'])
def reinhardt_normalization():
    # Retrieve the input image from the request
    file = request.files['input_image']
    image = skimage.io.imread(file)[:, :, :3]

    import p_cc_lab_mean_std
    import p_cn_rein

    # Load reference image for normalization
    ref_image_file = 'endoref.png'  # L1.png

    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = p_cc_lab_mean_std.lab_mean_std(im_reference)

    # perform Reinhardt color normalization
    im_nmzd = p_cn_rein.rein(image, mean_ref, std_ref)

    # Convert the normalized image to base64 string
    import io
    import base64

    buffered = io.BytesIO()
    skimage.io.imsave(buffered, im_nmzd, format='PNG')
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the base64 encoded normalized image as JSON response
    return jsonify({'normalized_image': encoded_image})

@app.route('/api/execute-notebook', methods=['POST'])
def execute_notebook():
    # Retrieve the input image from the request
    file = request.files['input_image']
    image = skimage.io.imread(file)[:, :, :3]

    plt.imshow(image)
    _ = plt.title('Input Image', fontsize=16)

    import p_cc_lab_mean_std
    import p_cn_rein

    # Load reference image for normalization
    ref_image_file = 'endoref.png'  # L1.png

    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = p_cc_lab_mean_std.lab_mean_std(im_reference)

    # perform reinhard color normalization
    im_nmzd = p_cn_rein.rein(image, mean_ref, std_ref)

    # Display results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(im_reference)
    _ = plt.title('Reference Image', fontsize=titlesize)

    plt.subplot(1, 2, 2)
    plt.imshow(im_nmzd)
    _ = plt.title('Normalized Input Image', fontsize=titlesize)

    skimage.io.imsave('normalized_image.png', im_nmzd)

    import p_cd_color_deconvolution

    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }

    # specify stains of input image
    stain_1 = 'hematoxylin'
    stain_2 = 'eosin'
    stain_3 = 'null'

    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    im_stains = p_cd_color_deconvolution.color_deconvolution(im_nmzd, W).Stains

    # Display results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(im_stains[:, :, 0])
    plt.title(stain_1, fontsize=titlesize)

    plt.subplot(1, 2, 2)
    plt.imshow(im_stains[:, :, 1])
    _ = plt.title(stain_2, fontsize=titlesize)

    import f_s_cdog

    # get nuclei/hematoxylin channel
    im_nuclei_stain = im_stains[:, :, 0]

    # segment foreground
    foreground_threshold = 50

    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < foreground_threshold)

    # run adaptive multi-scale LoG filter
    min_radius = 5
    max_radius = 20

    im_log_max, im_sigma_max = f_s_cdog.cdog(
        im_nuclei_stain, im_fgnd_mask,
        sigma_min=min_radius * np.sqrt(2),
        sigma_max=max_radius * np.sqrt(2)
    )

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 9

    import s_n_max_clustering

    im_nuclei_seg_mask, seeds, maxima = s_n_max_clustering.max_clustering(
        im_log_max, im_fgnd_mask, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 100

    import s_l_area_open

    im_nuclei_seg_mask = s_l_area_open.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

    results = {
        'number_of_nuclei': len(objProps),
        # Include any other relevant results you want to send
    }

    # Save the mask overlayand bounding box images
    mask_overlay = (skimage.color.label2rgb(im_nuclei_seg_mask, image, bg_label=0) * 255).astype(np.uint8)
    bbox_image = (image.copy()).astype(np.uint8)

    for i in range(len(objProps)):
        c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
        width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
        height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1
        cur_bbox = {
            "type": "rectangle",
            "center": c,
            "width": width,
            "height": height,
        }
        plt.plot(c[0], c[1], 'g+')
        mrect = mpatches.Rectangle(
            [c[0] - 0.5 * width, c[1] - 0.5 * height],
            width, height, fill=False, ec='g', linewidth=2
        )
        plt.gca().add_patch(mrect)
        cv2.rectangle(bbox_image, (objProps[i].bbox[1], objProps[i].bbox[0]),
                      (objProps[i].bbox[3], objProps[i].bbox[2]), (0, 255, 0), 2)

    # Save the mask overlay and bounding box images
    skimage.io.imsave('mask_overlay.png', mask_overlay)
    skimage.io.imsave('bounding_boxes.png', bbox_image)

    # Convert the images to base64 strings
    import io
    import base64

    with open('mask_overlay.png', 'rb') as file:
        encoded_mask_overlay = base64.b64encode(file.read()).decode('utf-8')

    with open('bounding_boxes.png', 'rb') as file:
        encoded_bounding_boxes = base64.b64encode(file.read()).decode('utf-8')

    # Return the results and encoded images as JSON
    return jsonify({
        'number_of_nuclei': len(objProps),
        'mask_overlay': encoded_mask_overlay,
        'bounding_boxes': encoded_bounding_boxes
    })

@app.route('/api/python-files/<filename>')
def serve_python_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Return the content as the response
    return content

if __name__ == '__main__':
    app.run(debug=True, port=5002)
