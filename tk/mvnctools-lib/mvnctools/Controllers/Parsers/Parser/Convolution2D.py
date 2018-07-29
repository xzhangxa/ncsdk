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

from .Layer import Layer
from mvnctools.Controllers.Tensor import PopulatedTensor
import mvnctools.Models.Layouts as Layouts
from mvnctools.Controllers.TensorFormat import TensorFormat
from mvnctools.Models.EnumDeclarations import PadStyle
import numpy as np

class Convolution2DUniversal(Layer):
    def __init__(self, *args):
        super().__init__(*args)
        self.padStyle = PadStyle.caffe

    def loadKernelSize(self, kernelHeight, kernelWidth):
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

    def getKernelSize(self):
        return self.kernelHeight, self.kernelWidth

    def loadStrideSize(self, strideHeight, strideWidth):
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth

    def getStrideSize(self):
        return (self.strideHeight, self.strideWidth)

    def loadPadding(self, paddingHeight, paddingWidth):
        self.paddingHeight = paddingHeight
        self.paddingWidth = paddingWidth

    def setPadStyle(self, padStyle):
        if padStyle == 'VALID':
            self.padStyle = PadStyle.tfvalid
        elif padStyle == 'SAME':
            self.padStyle = PadStyle.tfsame

    def getPadding(self):
        return (self.paddingHeight, self.paddingWidth)

    def loadDilation(self, dilationFactor):
        self.dilationFactor = dilationFactor

    def getDilation(self):
        return self.dilationFactor

    def setBiasEnabled(self, flag):
        self.hasBias = flag

    def biasEnabled(self):
        return self.hasBias

    def getBias(self):
        return self.bias

    def setBias(self, data):
        self.bias = PopulatedTensor(data)

    def getWeights(self):
        return self.weights

    def setWeights(self, data):
        self.weights = PopulatedTensor(data)

    def loadOutputChannels(self, outputChannels):
        self.outputChannels = outputChannels

    def loadTrainedParameters(self, **kwargs):
        self.weights = PopulatedTensor(kwargs['weights'])
        try:
            self.bias = PopulatedTensor(kwargs['bias'])
        except:
            #print("No Bias")
            pass

class Deconvolution(Convolution2DUniversal):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]

    def loadGroupSize(self, groupSize):
        self.groupSize = groupSize

    def convert2Conv(self):
        # See http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution

        # Equivalent convolution has padding p' = k - p - 1
        paddingHeight = self.kernelHeight - self.paddingHeight - 1
        paddingWidth = self.kernelWidth - self.paddingWidth - 1

        conv = Convolution2D(self.name, self.inputTensorNames, self.outputTensorNames)
        for attr_name in self.__dict__:
            setattr(conv, attr_name, getattr(self, attr_name))

        conv.loadPadding(paddingHeight, paddingWidth)

        # weights = conv.weights.data[:,:,::-1,::-1]
        # weights = np.swapaxes(weights, 0, 1)
        # conv.loadTrainedParameters(weights=weights)

        return conv


class Convolution2D(Convolution2DUniversal):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, ())
        self.formatPool = [(tfCM, tfCM)]

    def loadGroupSize(self, groupSize):
        self.groupSize = groupSize

    def getGroupSize(self):
        return self.groupSize


class ConvolutionDepthWise2D(Convolution2DUniversal):
    def __init__(self, *args):
        super().__init__(*args)
        # TODO: Merge layout representations
        self.addCompatibleLayout(Layouts.NCHW)    # Planar
        self.addCompatibleLayout(Layouts.NHCW)    # Row Interleaved

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        tfIV = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))

        # In mobilenet ssd, priorbox forces convertion layers. If type 'any' of TensorFormat
        # is implemented, this should not be necessary
        from mvnctools.Controllers.Globals import USING_MA2480
        if USING_MA2480:
            self.formatPool = [(tfIV, tfIV), (tfCM, tfCM)]
        else:
            self.formatPool = [(tfCM, tfCM), (tfIV, tfIV)]

    def loadGroupSize(self, groupSize):
        self.groupSize = groupSize

    def getGroupSize(self):
        return self.groupSize
