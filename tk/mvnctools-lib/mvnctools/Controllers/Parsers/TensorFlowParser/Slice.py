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


from mvnctools.Controllers.Parsers.Parser.crop import Crop
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getOutputNames, findTensorValue, fillTensorDimensions
import numpy as np

def load(obj, operations):
    # 3 inputs (input/begin/size) 1 output
    assert(len(obj.inputs) == 3)
    assert(len(obj.outputs) == 1)

    def getSliceBegin(obj):
        begin = [x.name for x in obj.inputs if '/begin:0' in x.name]

        assert(len(begin) == 1)

        return begin[0]

    def getSliceSize(obj):
        size = [x for x in obj.inputs if '/size:0' in x.name]

        assert(len(size) == 1)

        return size

    def getSliceInputNames(obj):
        return [x.name for x in obj.inputs if '/size:0' not in x.name
            and '/begin:0' not in x.name]

    x = Crop(obj.name, getSliceInputNames(obj), getOutputNames(obj))
    
    begin = findTensorValue(getSliceBegin(obj), operations)

    # ensure it is four dimensional
    begin = np.array(fillTensorDimensions(begin, convertToNCHW=False))

    # print("Setting slice offset to: " + str(begin))
    x.setOffset(begin[1:])

    return [x]