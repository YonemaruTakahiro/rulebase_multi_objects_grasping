import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import skimage.io
import cv2
import colorsys
# from mrcnn import utils
# from mrcnn import visualize
# from mrcnn.visualize import display_images

import mrcnn.model as modellib
from mrcnn.model import log
import soyjoy


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

# 推論結果を表示用にレンダリング
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = random_colors(N)

    masked_image = image.copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        camera_color = (color[0] * 255, color[1] * 255, color[2] * 255)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), camera_color, 1)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        camera_font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(masked_image, caption, (x1, y1), camera_font, 1, camera_color)

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

    return masked_image.astype(np.uint8)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def estimate(ROOT_DIR, IMAGE_DIR, filename):
    """Runs the detection pipeline.

            input
            ROOT_DIR: Mask R-CNN directory
            imgpath: path of img to estimate

            Return
            roi: (y1, x1, y2, x2) detection bounding boxes
            class_id: N int class IDs
            score: N float probability scores for the class IDs
            mask: H, W, N instance binary masks
            """

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    sys.path.append(os.path.join(ROOT_DIR, "samples/soyjoy/"))

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to soyjoy trained weights
    # You can download this file from the Releases page
    # https://github.com/matterport/Mask_RCNN/releases
    SOYJOY_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_soyjoy_0010.h5")
    print(SOYJOY_WEIGHTS_PATH)

    config = soyjoy.soyjoyConfig()
    SOYJOY_DIR = os.path.join(ROOT_DIR, "datasets/soyjoy")
    # Override the training configurations with a few
    # changes for inferencing.

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    # Set path to soyjoy weights file

    # Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    weights_path = os.path.join(ROOT_DIR, "mask_rcnn_soyjoy_0010.h5")

    # Or, load the last model you trained

    print(weights_path)

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    image = cv2.imread(os.path.join(IMAGE_DIR, filename))

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]

    out = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            ['BG', 'soyjoy'], r['scores'], ax=ax,
                            title="Predictions")

    outputname = 'result_' + filename
    print(outputname)
    cv2.imwrite(os.path.join(IMAGE_DIR, outputname), out)
    cv2.imshow('out', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return r

if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("../..")
    IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/soyjoy/test")
    filename = '20200616082049.jpg'
    estimate(ROOT_DIR, IMAGE_DIR, filename)
