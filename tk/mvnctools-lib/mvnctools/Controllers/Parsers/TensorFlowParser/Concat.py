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

from mvnctools.Controllers.Parsers.Parser.Concat import Concat
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getOutputNames, findTensorValue
from mvnctools.Models import Layouts

def load(obj, operations):
    def getConcatInputs(obj, axis_str):
        inputs = obj.inputs

        return [inp.name for inp in inputs
            if axis_str not in inp.name.lower()]

    # many inputs, one output
    assert(len(obj.outputs) == 1)

    # the axis dimension input name changes depending on version
    axis_str = {
        'Concat': 'concat_dim:0',
        'ConcatV2': 'axis:0'
    }.get(obj.type, False)

    assert(axis_str) # ensure type is supported

    # make sure we only got rid of one input
    inpts = getConcatInputs(obj, axis_str)
    assert(len(inpts) == (len(obj.inputs) - 1))

    x = Concat(obj.name, inpts, getOutputNames(obj))

    # find value of the axis
    axis = findTensorValue(obj.name + '/' + axis_str, operations)
    axis = [0, 2, 3, 1][axis]

    # print("Axis found in " + obj.name + " : " + str(axis))
    x.loadConcatAxis(axis)

    return [x]
