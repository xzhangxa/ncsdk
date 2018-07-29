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


from mvnctools.Controllers.Parsers.Parser.ReLU import ReLU, LeakyReLU

def load(obj, parsedNetworkObj):
    # This operation supports only one inputTensor and one
    # outputTensor
    assert(len(obj.bottom) == 1)
    assert(len(obj.top) == 1)

    if obj.relu_param.negative_slope == 0.0:
        x = ReLU(obj.name, obj.bottom, obj.top)
    else:
        x = LeakyReLU(obj.name, obj.bottom, obj.top)

        negativeSlope = obj.relu_param.negative_slope
        x.loadNegativeSlope(negativeSlope)

    # When choosing an output-node, Caffe considers ReLUs after a chosen node.
    x.setInPlace()

    return [x]