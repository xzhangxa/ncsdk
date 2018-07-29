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

class InnerProduct(Layer):

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]

    def loadKernelSize(self, kernelHeight, kernelWidth):
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

    def loadStrideSize(self, strideHeight, strideWidth):
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth

    def loadPadding(self, paddingHeight, paddingWidth):
        self.paddingHeight = paddingHeight
        self.paddingWidth = paddingWidth

    def loadDilation(self, dilationFactor):
        self.dilationFactor = dilationFactor

    def setBiasEnabled(self, flag):
        self.hasBias = flag

    def biasEnabled(self):
        return self.hasBias

    def getBias(self):
        return self.bias

    def getWeights(self):
        return self.weights

    def loadOutputChannels(self, outputChannels):
        self.outputChannels = outputChannels

    def loadTrainedParameters(self, **kwargs):
        # print("KW ARGSSSS", kwargs)
        self.weights = PopulatedTensor(kwargs['weights'])
        self.weights.setLayout((0, 3, 2, 1))
        try:
            self.bias = PopulatedTensor(kwargs['bias'])
        except:
            print("No Bias")

    def flatten(self):
        """
            By flattening an Inner Product, we can operate on Vectors rather
            than Matricies.
        """

        w = self.getWeights().data

        w = w.flatten()

        self.loadTrainedParameters(weights=w)

    def isInput3D(self):
        s = self.inputTensors[0].getTopEncloserRecursive().getShape()
        return len(s) - 1 != s.count(1)

    def canBeConvolution(self):
        # Check if input volume is 3D
        return self.isInput3D()

    def convert2Conv(self):
        from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D

        layer = Convolution2D(self.getStringifiedName(), self.inputTensorNames, self.outputTensorNames)

        # Copy the attributes
        for attr_name in self.__dict__:
            setattr(layer, attr_name, getattr(self, attr_name))

        _, __, out_ch, ___ = layer.weights.shape
        _, in_ch, height, width = self.inputTensors[0].getTopEncloserRecursive().getShape()
        layer.weights.reshape((out_ch, in_ch, height, width))

        # Set kernel parameter
        layer.loadKernelSize(height, width)
        layer.loadStrideSize(1, 1)
        layer.loadPadding(0,0)
        layer.loadGroupSize(1)
        layer.loadDilation(1)

        return layer