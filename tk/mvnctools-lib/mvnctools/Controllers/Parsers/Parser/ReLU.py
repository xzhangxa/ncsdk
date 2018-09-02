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
import mvnctools.Models.Layouts as Layouts
from mvnctools.Controllers.TensorFormat import TensorFormat


class ReLU(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (2,))
        tfIV = TensorFormat(Layouts.NHCW, (1, 3))
        self.formatPool = [(tfCM, tfCM), (tfIV, tfIV)]
        self.reluX = 0

        self.negativeSlope = 0.0

    def loadReluX(self, X):
        self.reluX = X

    def loadNegativeSlope(self, nSlope):
        self.negativeSlope = nSlope

class LeakyReLU(ReLU):

    def loadNegativeSlope(self, nSlope):
        self.negativeSlope = nSlope