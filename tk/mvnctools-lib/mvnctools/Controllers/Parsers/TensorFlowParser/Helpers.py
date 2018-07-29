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

def getBatchedTensorChannels(numpy_shape):
    assert(len(numpy_shape) == 4)
    return numpy_shape[3]

def getBatchedTensorHeight(numpy_shape):
    assert(len(numpy_shape) == 4)
    return numpy_shape[1]

def getBatchedTensorWidth(numpy_shape):
    assert(len(numpy_shape) == 4)
    return numpy_shape[2]


def fillTensorDimensions(numpy_shape, convertToNCHW=True):
    # Fill out Tensor to hold four dimensions
    from tensorflow import TensorShape

    if numpy_shape == None:
        data = []
    elif isinstance(numpy_shape, TensorShape):
        data = numpy_shape.as_list()
    else:
        data = list(numpy_shape)

    for x in range(4 - len(data)):
        data.insert(0, 1)

    # Convert all tensors to NCHW
    if convertToNCHW:
        data = [data[0], data[3], data[1], data[2]]

    return tuple(data)

def loadTensorSizes(parsedLayers, parsedModelObj):
    tensorDict = {}

    for operation in parsedModelObj.get_operations():
        for x in operation.inputs:
            tensorDict[x.name] = fillTensorDimensions(x.get_shape())

        for x in operation.outputs:
            tensorDict[x.name] = fillTensorDimensions(x.get_shape())

    for layer in parsedLayers:
        layer.loadTensorSizes(tensorDict)

def getPadding(in_dim, kernel_dim, stride_dim, padding_type):
    def same_padding(in_dim, kernel_dim, stride_dim):
        import numpy as np
        """
        Calculates the output dimension and also the padding required for that dimension.
        :param in_dim: Width/Height of Input
        :param kernel_dim: Width/Height of Kernel
        :param stride_dim: Vertical/Horizontal Stride
        """
        in_dim = np.array(in_dim)
        in_dim[in_dim == None] = 1

        kernel_dim = np.array(kernel_dim)
        stride_dim = np.array(stride_dim)
        
        output_dim = np.ceil(np.float_(in_dim) / np.float_(stride_dim))
        pad = ((output_dim - 1) * stride_dim + kernel_dim - in_dim) / 2
        
        return list(np.int_(pad))

    def valid_padding(in_dim, kernel_dim, stride_dim):
        # output_dim = np.ceil(np.float_(in_dim - kernel_dim + 1) / np.float_(stride_dim))
        pad = [0] * len(in_dim) # as many zeros as there are dimensions

        return pad

    if padding_type == b'VALID':
        return valid_padding(in_dim, kernel_dim, stride_dim)
    elif padding_type == b'SAME':
        return same_padding(in_dim, kernel_dim, stride_dim)
    else:
        return None

def stripTensorName(tensorName):
    if tensorName[-2:] == ':0':
        return tensorName[0:-2]
    else:
        return tensorName

# find tensor from given name and graph operations
def findTensor(tensorName, operations):
    # ensure the tensor name does not have a :0
    tensorName = stripTensorName(tensorName)

    tensor = [opx for opx in operations 
        if tensorName.lower() == opx.name.lower()]
        
    assert(len(tensor) == 1)

    return tensor[0]

# find tensor value
def findTensorValue(tensorName, operations):
    tensor = findTensor(tensorName, operations)

    values = tensor.values()
    assert(len(values) == 1)

    # if tensor only has one set of values, evaluate with default session
    # and return as numpy array
    values_npy = values[0].eval()
    
    return values_npy

def getInputNames(tensor):
    return [x.name for x in tensor.inputs]

def getOutputNames(tensor):
    return [x.name for x in tensor.outputs]
