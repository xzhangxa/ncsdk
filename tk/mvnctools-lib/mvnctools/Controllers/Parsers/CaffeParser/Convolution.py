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


from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D

from .Helpers import getBatchedTensorDepth

def load(obj, parsedNetworkObj):
    def unitListOrEmpty(l, default):
        if not l:
            return default
        else:
            assert(len(l) == 1)
            return l[0]

    # This operation supports only one inputTensor and one
    # outputTensor
    assert(len(obj.bottom) == 1)
    assert(len(obj.top) == 1)

    # Only 2D convolution is supported
    assert(obj.convolution_param.axis == 1)

    bottomName = obj.bottom[0]
    inputTensorShape = parsedNetworkObj.blobs[bottomName].data.shape

    topName = obj.top[0]
    outputTensorShape = parsedNetworkObj.blobs[topName].data.shape

    inputTensorDepth = getBatchedTensorDepth(inputTensorShape)
    outputTensorDepth = getBatchedTensorDepth(outputTensorShape)
    groupSize = obj.convolution_param.group
    if len(set([inputTensorDepth, outputTensorDepth, groupSize])) == 1:
        x = ConvolutionDepthWise2D(obj.name, obj.bottom, obj.top)
    else:
        x = Convolution2D(obj.name, obj.bottom, obj.top)
    x.loadGroupSize(groupSize)

    outputChannels = obj.convolution_param.num_output
    x.loadOutputChannels(outputChannels)

    dilationFactor = unitListOrEmpty(obj.convolution_param.dilation, 1)
    x.loadDilation(dilationFactor)

    # Get the parameters from Caffe:
    kernel_size = unitListOrEmpty(obj.convolution_param.kernel_size, 0)
    kernel_h = obj.convolution_param.kernel_h
    kernel_w = obj.convolution_param.kernel_w

    # In Caffe, you can either have kernel_size xor
    # kernel_h and kernel_w defined.
    if kernel_size > 0:
        x.loadKernelSize(kernel_size, kernel_size)
    else:
        x.loadKernelSize(kernel_h, kernel_w)

    stride = unitListOrEmpty(obj.convolution_param.stride, 1)
    stride_h = obj.convolution_param.stride_h
    stride_w = obj.convolution_param.stride_w

    # In Caffe, you can either have stride xor
    # stride_h and stride_w defined.
    if stride_h > 0 or stride_w > 0:
        x.loadStrideSize(stride_h, stride_w)
    else:
        x.loadStrideSize(stride, stride)

    pad = unitListOrEmpty(obj.convolution_param.pad, 0)
    pad_h = obj.convolution_param.pad_h
    pad_w = obj.convolution_param.pad_w

    # In Caffe, you can either have pad xor
    # pad_h and pad_w defined.
    if pad > 0:
        x.loadPadding(pad, pad)
    else:
        x.loadPadding(pad_h, pad_w)

    biasTerm = obj.convolution_param.bias_term
    x.setBiasEnabled(biasTerm)

    # Load trained parameters
    trainedParameters = tuple(parsedNetworkObj.params[obj.name])

    # When bias exists, Caffe ignores the `bias_term` field.
    if len(trainedParameters) > 1:
        x.setBiasEnabled(True)

    weights = trainedParameters[0].data
    bias = trainedParameters[1].data if len(trainedParameters) > 1 else None

    x.loadTrainedParameters(weights=weights, bias=bias)

    return [x]
