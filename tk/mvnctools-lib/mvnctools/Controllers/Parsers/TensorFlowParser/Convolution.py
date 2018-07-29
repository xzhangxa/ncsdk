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

import numpy as np
from mvnctools.Controllers.Parsers.Parser.Convolution2D import (Convolution2D,
    ConvolutionDepthWise2D)

from .Helpers import (fillTensorDimensions, getPadding, findTensor,
    findTensorValue, getInputNames, getOutputNames)

def load(obj, operations):
    # operation only supports one input Tensor and output Tensor
    assert(len(obj.inputs) == 2) # input and weights
    assert(len(obj.outputs) == 1)
    
    # first input always assumed to be conv input
    conv_inputs = [getInputNames(obj)[0]]

    inputTensorShape = fillTensorDimensions(obj.inputs[0].shape, True)
    outputTensorShape = fillTensorDimensions(obj.outputs[0].shape, True)

    if obj.type == 'Conv2D':
        x = Convolution2D(obj.name, conv_inputs, getOutputNames(obj))
        groupSize = 1 # Conv2D
        x.loadGroupSize(groupSize)
    elif obj.type == 'DepthwiseConv2dNative':
        x = ConvolutionDepthWise2D(obj.name, conv_inputs, getOutputNames(obj))
    else: # else not supported layer
        assert(False, "Layer type not supported by Convolution: " + obj.type)

    # NCHW
    outputChannels = outputTensorShape[1]
    x.loadOutputChannels(outputChannels)

    # ensure dilation is equal - do we support otherwise?
    try:
        # NHWC
        dilationFactor = obj.get_attr('dilations')
        assert(len(set(dilationFactor[1:3])) == 1)
        
        # dilation in batch and channels are always 1
        x.loadDilation(dilationFactor[1])
    except ValueError:
        # assume that dilation is default - 1
        x.loadDilation(1)

    # assume second input kernel
    kernel = obj.inputs[1]

    # The kernel is stored as HWCO
    kernelSize = fillTensorDimensions(kernel.shape, False)[0:2]
    x.loadKernelSize(kernelSize[0], kernelSize[1])

    # NHWC
    stride = obj.get_attr('strides')[1:3]
    x.loadStrideSize(stride[0], stride[1])

    padding = getPadding(inputTensorShape[0:2], kernelSize,
        stride, obj.get_attr('padding'))

    x.loadPadding(padding[0], padding[1])

    x.setPadStyle(obj.get_attr('padding').decode('ascii'))

    # no built in bias - ignore here?
    x.setBiasEnabled(False)

    weightName = kernel.name
    weights = findTensorValue(weightName, operations)
    weights = np.transpose(weights, (3, 2, 0, 1))

    x.loadTrainedParameters(weights=weights)

    return [x]
