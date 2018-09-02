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
from mvnctools.Controllers.TensorFormat import TensorFormat
import mvnctools.Models.Layouts as Layouts
import numpy as np

class Crop(Layer):

    def __init__(self, *args):
        super().__init__(*args)

        # No stride support on myriad
        chmin = TensorFormat(Layouts.NHWC, ())
        row_inter = TensorFormat(Layouts.NHCW, ())
        planar = TensorFormat(Layouts.NCHW, ())

        self.formatPool = [
            (chmin, chmin),
            (planar, planar),
            (row_inter, row_inter),
        ]

        # Default to no offset on any axis.
        self.crop_offset = np.array([0, 0, 0], np.dtype("<u4"))

    def setOffset(self, arr):
        assert len(arr) == 3
        self.crop_offset = arr

    def getOffset(self):
        return self.crop_offset
