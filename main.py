from PIL import Image
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

MEAN_DISTANCE = 25
MIN_DIM_PCT = 0.15
FILL_INSTEAD_OF_STRETCH = True


def try_detect_image_splits(image_path):
    try:
        detect_image_splits(image_path)
    except:
        print("Error: " + image_path)


def detect_image_splits(image_path):
    image = Image.open(image_path)
    pixels = image.load()
    h_splits = detect_horizontal_splits(image, pixels)
    v_splits = detect_vertical_splits(image, pixels)

    split_ranges = compute_ranges(image, h_splits, v_splits)
    image_name = os.path.basename(image_path)
    image_name = base_name = os.path.splitext(image_name)[0]
    if not os.path.exists('splitted'):
        os.mkdir('splitted')

    for index, split_range in enumerate(split_ranges):
        output_name = image_name + '_' + str(index + 1) + '.jpg'
        output_name = os.path.join('splitted', output_name)
        save_image(output_name, image, split_range)

    # image.show()


def detect_vertical_splits(image, pixels):
    return detect_splits(pixels, image.width, image.height, compute_mean_col_diff)


def detect_horizontal_splits(image, pixels):
    return detect_splits(pixels, image.height, image.width, compute_mean_line_diff)


def detect_splits(pixels, main_dim, diff_dim, diff_method):
    means = []
    start = int((main_dim * 0.1) / 2)
    end = main_dim - start

    for i in range(0, main_dim - 1):
        if i < start or i > end:
            means.append(0)
            continue

        means.append(diff_method(diff_dim, pixels, i))

    means_mean = statistics.mean(means)

    distances = []
    for i, mean in enumerate(means):
        if i < start or i > end:
            distances.append(0)
            continue

        distance = abs(means_mean - mean)
        distances.append(distance)

    rtn = []
    for col, distance in enumerate(distances):
        if distance > MEAN_DISTANCE:
            rtn.append(col)

    return rtn


def compute_mean_col_diff(image_height, pixels, x):
    total = 0
    for y in range(0, image_height):
        total = total + pixel_diff(pixels[x, y], pixels[x+1, y])

    total = total / image_height
    return total


def compute_mean_line_diff(image_width, pixels, y):
    total = 0
    for x in range(0, image_width):
        total = total + pixel_diff(pixels[x, y], pixels[x, y+1])

    total = total / image_width
    return total


def pixel_diff(a, b):
    return abs(grayscale(a) - grayscale(b))


def grayscale(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3


def plot_diff(col_means, col_means_mean):
    x = range(0, len(col_means))

    plt.plot(x, col_means, label='column diff mean')

    plt.xlabel('column index')
    plt.ylabel('value')

    plt.hlines(col_means_mean, 0, len(col_means))

    plt.title("Mean")

    plt.legend()

    plt.show()
    pass


def compute_ranges(image, h_splits, v_splits):
    rtn = []
    if len(h_splits) == 0 and len(v_splits) == 0:
        return rtn
    h_splits.insert(0, 0)
    h_splits.append(image.height-1)
    v_splits.insert(0, 0)
    v_splits.append(image.width-1)

    for v in range(0, len(v_splits) - 1):
        for h in range(0, len(h_splits) - 1):
            start = (v_splits[v], h_splits[h])
            end = (v_splits[v+1], h_splits[h+1])
            rtn.append((start, end))

    rtn = filter_ranges(rtn, image.width, image.height)

    return rtn


def filter_ranges(split_ranges, image_width, image_height):
    rtn = []
    for split_range in split_ranges:
        start = split_range[0]
        end = split_range[1]

        if end[0] - start[0] <= image_width * MIN_DIM_PCT:
            continue
        if end[1]-start[1] <= image_height * MIN_DIM_PCT:
            continue

        rtn.append(split_range)
    return rtn


def print_range(pixels, split_range):
    start = split_range[0]
    end = split_range[1]
    for x in range(start[0], end[0]):
        pixels[x, start[1]] = (0, 255, 0)

    for x in range(start[0], end[0]):
        pixels[x, end[1]] = (0, 255, 0)

    for y in range(start[1], end[1]):
        pixels[start[0], y] = (0, 255, 0)

    for y in range(start[1], end[1]):
        pixels[end[0], y] = (0, 255, 0)


def save_image(image_name, image, split_range):
    start = split_range[0]
    end = split_range[1]
    new_image = image.crop((start[0], start[1], end[0], end[1]))
    resize_to_size(new_image, 512).save(image_name)


def resize_to_size(im, desired_size):

    if not FILL_INSTEAD_OF_STRETCH:
        return im.resize((desired_size, desired_size))

    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im.thumbnail(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))
    return new_im


def main():
    files = os.listdir('input_to_split')
    Parallel(n_jobs=10)(delayed(try_detect_image_splits)(
        os.path.join('input_to_split', image)) for image in tqdm(files))


if __name__ == '__main__':
    main()
