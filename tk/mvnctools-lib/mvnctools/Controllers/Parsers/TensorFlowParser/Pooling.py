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
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getInputNames, getOutputNames, getPadding, fillTensorDimensions

def load(obj, operations):
    # one input/output
    assert(len(obj.inputs) == 1)
    assert(len(obj.outputs) == 1)

    x = Pooling(obj.name, getInputNames(obj), getOutputNames(obj))

    kernelSize = obj.get_attr('ksize')[1:3]
    x.loadKernelSize(kernelSize[0], kernelSize[1])

    stride = obj.get_attr('strides')[1:3]
    x.loadStrideSize(stride[0], stride[1])

    inputTensorShape = fillTensorDimensions(obj.inputs[0].shape)[1:3]
    padding = getPadding(inputTensorShape, kernelSize, stride, obj.get_attr('padding'))

    x.loadPadding(padding[0], padding[1])

    x.setPadStyle(obj.get_attr('padding'))

    pool_type = {
        'MaxPool': Pooling.Type.MAX,
        'AvgPool': Pooling.Type.AVE
    }.get(obj.type, False)

    assert(pool_type) # make sure it was a known type

    x.loadType(pool_type)

    x.loadGlobal(kernelSize == inputTensorShape)

    return [x]