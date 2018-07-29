#!/usr/bin/env python3

from mvnctools.Controllers.Parsers.Parser.Convolution2D import Deconvolution
import numpy as np

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

    # Only 2D deconvolution is supported
    assert(obj.convolution_param.axis == 1)

    bottomName = obj.bottom[0]
    inputTensorShape = parsedNetworkObj.blobs[bottomName].data.shape

    topName = obj.top[0]
    outputTensorShape = parsedNetworkObj.blobs[topName].data.shape

    inputTensorDepth = getBatchedTensorDepth(inputTensorShape)
    outputTensorDepth = getBatchedTensorDepth(outputTensorShape)
    groupSize = obj.convolution_param.group
    if groupSize > 1:
        raise ValueError('Deconvolutions with group size > 1 are not supported')
    else:
        x = Deconvolution(obj.name, obj.bottom, obj.top)
        x.loadGroupSize(groupSize)

    outputChannels = obj.convolution_param.num_output
    x.loadOutputChannels(outputChannels)

    dilationFactor = unitListOrEmpty(obj.convolution_param.dilation, 1)
    if dilationFactor != 1:
        raise ValueError('Unsupported Deconvolution with dilation != 1')
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
    # For Deconv the wheights are in CKHW format.
    # Transform to KCWH
    weights = np.swapaxes(weights, 0, 1)
    # Taps need to be roated in the HW plane because caffe
    # implements the deconvolution via convolution backward pass
    # which does an 180deg rotation.
    weights = weights[:,:,::-1,::-1]

    bias = trainedParameters[1].data if len(trainedParameters) > 1 else None

    x.loadTrainedParameters(weights=weights, bias=bias)

    return [x]
