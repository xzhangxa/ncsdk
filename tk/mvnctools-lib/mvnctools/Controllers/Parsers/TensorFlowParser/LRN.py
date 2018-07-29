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

from mvnctools.Controllers.Parsers.Parser.LRN import LRN
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getInputNames, getOutputNames

def load(obj, operations):
    # one input/output
    assert(len(obj.inputs) == 1)
    assert(len(obj.outputs) == 1)

    x = LRN(obj.name, getInputNames(obj), getOutputNames(obj))

    # tensorflow LRN is across
    x.loadType(LRN.Type.ACROSS)

    alpha = obj.get_attr('alpha')
    x.loadAlpha(alpha)

    beta = obj.get_attr('beta')
    x.loadBeta(beta)

    bias = obj.get_attr('bias')
    x.loadK(bias)

    channel_depth = obj.get_attr('depth_radius')
    x.loadSquareKernelSize(channel_depth)

    return [x]