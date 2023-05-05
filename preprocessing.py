import tensorflow as tf
import numpy as np
import os


def load_faces_class(path):
    faces = []
    # select 20 image
    for img, i in zip(os.listdir(path), range(20)):
        faces.append(os.path.join(path, img))
    return faces


def load_faces_classes(path):
    faces = []
    for dir in os.listdir(path):
        faces.append((load_faces_class(os.path.join(path, dir))))
    return faces


def make_pairs(data):
    pairs = []
    labels = []
    i = 0
    for cls1 in data:
        for img1 in cls1:
            j = 0
            for cls2 in data:
                for img2 in cls2:
                    # prevent same images in a pair
                    if not (img1 == img2).all():
                        pairs.append((img1, img2))
                        labels.append(1 if i == j else 0)
                j += 1
        i += 1
    return pairs, labels


def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(
            img, max_delta=0.02, seed=(1, 2))
        img = tf.image.stateless_random_contrast(
            img, lower=0.6, upper=1, seed=(1, 3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(
            img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(
            np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(
            img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))

        data.append(img)

    return data
