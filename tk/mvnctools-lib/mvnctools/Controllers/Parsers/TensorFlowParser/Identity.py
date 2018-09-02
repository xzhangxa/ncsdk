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

from mvnctools.Controllers.Parsers.Parser.NoOp import NoOp
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getInputNames, getOutputNames

def load(obj, operations):
    # This operation supports only one inputTensor and one
    # outputTensor
    assert(len(obj.inputs) == 1)
    assert(len(obj.outputs) == 1)

    x = NoOp(obj.name, getInputNames(obj), getOutputNames(obj))

    return [x]
