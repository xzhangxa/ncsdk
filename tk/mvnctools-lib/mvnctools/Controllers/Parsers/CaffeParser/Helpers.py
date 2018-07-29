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


def getBatchedTensorDepth(numpy_shape):
    assert(len(numpy_shape) == 4)
    return numpy_shape[1]

def getBatchedTensorHeight(numpy_shape):
    assert(len(numpy_shape) == 4)
    return numpy_shape[2]

def getBatchedTensorWidth(numpy_shape):
    assert(len(numpy_shape) == 4)
    return numpy_shape[3]

def loadTensorSizes(parsedLayers, parsedNetworkObj):
    tensorDict = {}
    for tensorName, tensorObj in parsedNetworkObj.blobs.items():
        # Ensure only 4D Tensors are in rotation.
        data = list(tensorObj.shape)
        for x in range(4 - len(list(tensorObj.shape))):
            # data.insert(0, 1)
            data.append(1)
        tensorDict[tensorName] = tuple(data)
    for layer in parsedLayers:
        layer.loadTensorSizes(tensorDict)
