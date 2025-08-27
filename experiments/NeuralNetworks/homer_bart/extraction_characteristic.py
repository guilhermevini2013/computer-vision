import cv2
import numpy as np
import tensorflow as tf
import numpy
import os

characteristics = {
    "mouth_homer": {
        "range": ([95, 160, 175], [140, 185, 205]),
    },
    "pants_homer": {
        "range": ([150, 95, 0], [180, 120, 90]),
    },
    "shoes_homer": {
        "range": ([24, 23, 24], [45, 45, 45]),
    },
    "shirt_bart": {
        "range": ([11, 85, 240], [50, 105, 255]),
    },
    "pants_bart": {
        "range": ([125, 0, 0], [170, 12, 20]),
    },
    "shoes_bart": {
        "range": ([125, 0, 0], [170, 12, 20]),
    }
}

images_url = os.listdir("./images")
images = []
classes = []
characteristics_values = []

for image_url in images_url:
    image = cv2.imread(f"./images/{image_url}")
    h, w = image.shape[:2]
    if image_url.startswith("b"):
        classe = 0
    else:
        classe = 1
    classes.append(classe)

    image_characteristics = []

    for name, information in characteristics.items():
        range_color = information["range"]
        min_range = np.array(range_color[0])
        max_range = np.array(range_color[1])

        mask = cv2.inRange(image, min_range, max_range)
        quantity_pixels = cv2.countNonZero(mask)
        image_characteristics.append({
            "property":name,
            "value": round((quantity_pixels / (h*w)) * 100, 9)
        })

    characteristics_values.append(image_characteristics)


