#!/usr/bin/env python3

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


from mvnctools.Controllers.Parsers.Parser.Pooling import Pooling
from mvnctools.Controllers.Parsers.CaffeParser.Helpers import getBatchedTensorHeight, getBatchedTensorWidth

def load(obj, parsedNetworkObj):
    # This operation supports only one inputTensor and one
    # outputTensor
    assert(len(obj.bottom) == 1)
    assert(len(obj.top) == 1)

    x = Pooling(obj.name, obj.bottom, obj.top)

    # Get the parameters from Caffe:
    kernel_size = obj.pooling_param.kernel_size
    kernel_h = obj.pooling_param.kernel_h
    kernel_w = obj.pooling_param.kernel_w

    # In Caffe, you can either have kernel_size xor
    # kernel_h and kernel_w defined.
    if kernel_size > 0:
        x.loadKernelSize(kernel_size, kernel_size)
    else:
        x.loadKernelSize(kernel_h, kernel_w)

    stride = obj.pooling_param.stride
    stride_h = obj.pooling_param.stride_h
    stride_w = obj.pooling_param.stride_w

    # In Caffe, you can either have stride xor
    # stride_h and stride_w defined.
    if stride > 0:
        x.loadStrideSize(stride, stride)
    else:
        x.loadStrideSize(stride_h, stride_w)

    pad = obj.pooling_param.pad
    pad_h = obj.pooling_param.pad_h
    pad_w = obj.pooling_param.pad_w

    # In Caffe, you can either have pad xor
    # pad_h and pad_w defined.
    if pad > 0:
        x.loadPadding(pad, pad)
    else:
        x.loadPadding(pad_h, pad_w)

    pool_type = obj.pooling_param.pool
    if pool_type == 0:
        x.loadType(Pooling.Type.MAX)
    elif pool_type == 1:
        x.loadType(Pooling.Type.AVE)
    else:
        assert(False)

    global_pooling = obj.pooling_param.global_pooling
    x.loadGlobal(global_pooling)

    # Set the padding when global pooling is used
    if x.isGlobal():
        bottomName = obj.bottom[0]
        inputTensorShape = parsedNetworkObj.blobs[bottomName].data.shape
        kernel_h = getBatchedTensorHeight(inputTensorShape)
        kernel_w = getBatchedTensorWidth(inputTensorShape)
        x.loadKernelSize(kernel_h, kernel_w)

    return [x]