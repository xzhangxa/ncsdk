
# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import PIL
from PIL import Image
import skimage
import skimage.io
import skimage.transform
import numpy as np
import re
from mvnctools.Models.EnumDeclarations import SeedData


def preprocess(img, args, new_size):
    """
        Prepare an input for processing
    """

    if img is None:
        print("Warning: Randomly initialized input")
    elif args.image == "Debug":
        return SeedData.random
    elif args.image == "Debug_ones":
        return SeedData.all_ones
    elif args.image == "Debug_zeros":
        return SeedData.all_zeros
    elif args.image == "Debug_int":
        return SeedData.random_int
    else:
        # Should be an image path.
        data = input_to_npy(img, new_size)
        print(data.shape)
        data = standardize(data, new_size)
        print(data.shape)
        data = scale(data, args.raw_scale)
        print(data.shape)
        data = channel_swap(data, args.channel_swap)
        print(data.shape)
        data = mean(data, args.mean)
        print(data.shape)
        return data


def input_to_npy(path, new_size):
    """
        Convert from the input format to a numpy array.

        Currently Handles (in varying levels):
        - Images (PNG, JPG, BMP, GIF)
        - Numpy Files (saved with .save, rather than .tofile)   # TODO: Do tofile
        - Mat files, from MatLab.
    """

    # Image Parsing
    if path.split(".")[-1].lower() in ["png", "jpeg", "jpg", "bmp", "gif"]:
        greyscale = True if new_size[2] == 1 else False
        data = skimage.img_as_float(
            skimage.io.imread(
                path, as_grey=greyscale)).astype(
            np.float32)

    # Numpy Array
    elif path.split(".")[-1] in ["npy"]:
        im = np.load(path)

        if (len(im.shape) == 2):
            if(im.shape[0] != new_size[2] or im.shape[1] != new_size[3]):
                throw_error(ErrorTable.InvalidInputFile)
        elif (len(im.shape) == 3):
            if(im.shape[0] != new_size[2] or im.shape[1] != new_size[3]):
                throw_error(ErrorTable.InvalidInputFile)
        else:
            throw_error(ErrorTable.InvalidInputFile)
        data = np.asarray(im)

    # MAT File
    elif path.split(".")[-1] in ["mat"]:
        print("Filetype not officially supported use at your own peril: MAT File")
        import scipy.io
        im = scipy.io.loadmat(path)
        data = np.asarray(im)

    else:
        print("Unsupported Data File:", path)
        throw_error(ErrorTable.InputFileUnsupported)

    return data


def standardize(data, new_size):
    """
        Make sure that all inputs are treated the same.
    """
    # Add axis for greyscale images (size 1)
    if (len(data.shape) == 2):
        data = data[:, :, np.newaxis]

    # Transposed and reshaped. TODO: Make more obvious what is going on here.
    data = skimage.transform.resize(data, new_size[2:])
    data = np.transpose(data, (2, 0, 1))
    data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
    return data


def scale(data, raw_scale=1):
    return data * raw_scale


def channel_swap(data, channel_swap=None):
    if channel_swap is not None:
        data[0] = data[0][np.argsort(channel_swap), :, :]
    return data


def mean(data, mean=None):
    if mean is not None:
        #  Try loading mean from .npy file
        if re.search('[a-zA-Z]+', mean):
            try:
                mean = np.load(mean)
            except:
                throw_error(ErrorTable.InvalidNpyFile, mean)

            mean = mean.mean(1).mean(1)
            mean_arr = np.zeros(data.shape[1:])

            for x in range(mean.shape[0]):
                mean_arr[x].fill(mean[x])

            data[0] -= mean_arr

        # Else, try loading mean as tuple
        elif re.search('[,]+', mean):
            try:
                (R,G,B) = mean.split(',')
            except:
                throw_error(ErrorTable.InvalidTuple, mean)

            mean = np.asarray([float(R), float(G), float(B)])
            mean_arr = np.zeros(data.shape[1:])

            for x in range(mean.shape[0]):
                mean_arr[x].fill(mean[x])

            data[0] -= mean_arr

        # Else, load mean as single number
        elif re.search(r'\d+', mean):
            try:
                data = data - float(mean)
            except:
                throw_error(ErrorTable.InvalidMean, mean)

        # Else, invalid mean input
        else:
            throw_error(ErrorTable.InvalidMean, mean)

    return data
