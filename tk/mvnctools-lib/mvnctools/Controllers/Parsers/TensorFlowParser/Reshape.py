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

from mvnctools.Controllers.Parsers.Parser.Reshape import Reshape
from mvnctools.Controllers.Parsers.Parser.NoOp import NoOp
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getOutputNames, findTensorValue

def load(obj, operations):
    # two input/one output

    assert(len(obj.inputs) == 2)
    assert(len(obj.outputs) == 1)

    def getReshapeInputs(obj, shape=False):
        if shape:
            return [x.name for x in obj.inputs if '/shape:0' in x.name.lower()]
        else:
            return [x.name for x in obj.inputs if '/shape:0' not in x.name.lower()]

    inpts = [obj.inputs[0].name]  # getReshapeInputs(obj)
    assert(len(inpts) == 1)

    x = Reshape(obj.name, inpts, getOutputNames(obj))

    return[x]
