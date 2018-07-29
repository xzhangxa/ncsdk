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

import numpy as np
from mvnctools.Models.EnumDeclarations import *
from mvnctools.Controllers.EnumController import *
from ctypes import *
import math


def zyx_to_yxz_dimension_only(data, z=0, y=0, x=0):
    """
    Creates a tuple containing the shape of a conversion if it were to happen.
    :param data:
    :param z:
    :param y:
    :param x:
    :return:
    """
    z = data[0] if z == 0 else z
    y = data[1] if y == 0 else y
    x = data[2] if x == 0 else x

    return (y, x, z)


def zyx_to_yxz(data, data_type=np.float16):
    """
    Converts from Channel-Major (Column-Minor) to Row-Major (Channel-Minor)

    :param data: data to convert. Assumed flattened unless other parameters populated
    :param data_type: DataType Enum
    :return:
    """
    return np.moveaxis(
        data,
        0,
        2).ravel().reshape(
        data.shape[1],
        data.shape[2],
        data.shape[0])


def yxz_to_zyx(data, y=0, x=0, z=0):
    """
     Converts from Row-Major (Channel-Minor) to Channel-Major (Column-Minor)

     Assumes data is not flattened. If it is flattened, pass in y, x, z parameters
     :param data:
     :param y:
     :param x:
     :param z:
    """

    y = data.shape[0] if y == 0 else y
    x = data.shape[1] if x == 0 else x
    z = data.shape[2] if z == 0 else z

    # print(data)

    yxz = data.reshape((y * x, z))
    trans = yxz.transpose()
    trans = np.reshape(trans, (z * x, y))
    trans = np.reshape(trans, (z, y, x))

    return trans


def xyz_to_zyx(data):
    """
     Reverses dimensions of matrix.
    """
    trans = data.swapaxes(0, 2)

    return trans


def xyz_to_yxz(data):
    """
     Converts from Row-Major (Channel-Minor) to Channel-Major (Column-Minor)
    """
    trans = data.swapaxes(0, 1)
    return trans


def yxz_to_xyz(data):
    trans = data.swapaxes(0, 1)
    return trans

def yzx_to_zyx(data):
    trans = data.swapaxes(0, 1)
    return trans

def yzx_to_yxz(data):
    trans = data.swapaxes(1, 2)

    return trans

def kchw_to_hwck(data, k=0, c=0, fh=0, fw=0):
    """
    Needed for Convolutions converted to YXZ format.
    Assumes non-flattened data. If it is flattened, pass in appropiate parameters
    :param data:
    :param k:
    :param c:
    :param fh:
    :param fw:
    :return:
    """
    k = data.shape[0] if k == 0 else k
    c = data.shape[1] if c == 0 else c
    fh = data.shape[2] if fh == 0 else fh
    fw = data.shape[3] if fw == 0 else fw
    data = data.reshape((k, c, fh, fw))

    data = np.swapaxes(data, 0, 2)  # kchw -> hckw
    data = np.swapaxes(data, 1, 3)  # hckw -> hwkc
    data = np.swapaxes(data, 2, 3)  # hckw -> hwkc

    return data


def hwck_transpose_correction(data, fh=0, fw=0, c=0, k=0):
    fw = data.shape[0] if fw == 0 else fw
    fh = data.shape[1] if fh == 0 else fh
    c = data.shape[2] if c == 0 else c
    k = data.shape[3] if k == 0 else k

    return data


def merge_buffers_zyz(data):
    return


def storage_order_reshape(shape, layout, newLayout):
    newShape = (0, 0, 0)
    if ((layout == StorageOrder.orderZYX and newLayout == StorageOrder.orderYXZ) or 
        (layout == StorageOrder.orderYXZ and newLayout == StorageOrder.orderZYX)):
        newShape = (shape[1], shape[2], shape[0])
    elif ((layout == StorageOrder.orderYXZ and newLayout == StorageOrder.orderYZX) or 
        (layout == StorageOrder.orderYZX and newLayout == StorageOrder.orderYXZ)):
        newShape = (shape[0], shape[2], shape[1])
    elif (layout == newLayout):
        newShape = shape
    else:
        throw_error(ErrorTable.ConversionNotSupported, newLayout)
    return newShape

def storage_order_convert(data, layout, newLayout):
    """
    Converts the 'data' volume assumed to be in the StorageOrder specified 
    by 'layout' into a new volume with layout specified by 'newLayout'
    """
    newData = (0, 0, 0)
    if (layout == StorageOrder.orderZYX and newLayout == StorageOrder.orderYXZ):
        newData = zyx_to_yxz(data)
    elif (layout == StorageOrder.orderXYZ and newLayout == StorageOrder.orderYXZ):
        newData = xyz_to_yxz(data)
    elif (layout == StorageOrder.orderXYZ and newLayout == StorageOrder.orderZYX):
        newData = xyz_to_zyx(data)
    elif (layout == StorageOrder.orderYXZ and newLayout == StorageOrder.orderZYX):
        newData = yxz_to_zyx(data)
    elif (layout == StorageOrder.orderYXZ and newLayout == StorageOrder.orderXYZ):
        newData = yxz_to_xyz(data)
    elif (layout == StorageOrder.orderYZX and newLayout == StorageOrder.orderZYX):
        newData = yzx_to_zyx(data)
    elif (layout == StorageOrder.orderYZX and newLayout == StorageOrder.orderYXZ):
        newData = yzx_to_yxz(data)
    elif (layout == newLayout):
        newData = data
    else: 
        print("storage_order_convert:  %s -> %s" % (layout, newLayout))
        throw_error(ErrorTable.ConversionNotSupported, newLayout)

    return newData
