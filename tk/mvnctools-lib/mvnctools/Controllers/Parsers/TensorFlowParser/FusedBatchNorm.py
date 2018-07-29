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

import numpy as np
from mvnctools.Controllers.Parsers.Parser.BatchNorm import BatchNorm
from mvnctools.Controllers.Parsers.TensorFlowParser.Helpers import getInputNames, getOutputNames, findTensorValue

def load(obj, operations):
    inpts = getInputNames(obj)

    def getBatchNormInputNames(inputs):
        return [x for x in inputs
            if 'beta/read:0' not in x
            and 'moving_mean/read:0' not in x
            and 'moving_variance/read:0' not in x]    

    x = BatchNorm(obj.name, getBatchNormInputNames(inpts), [getOutputNames(obj)[0]])

    eps = obj.get_attr('epsilon')

    scale_param = findTensorValue([x for x in inpts if 'Const:0' in x][0], operations)
    offset = findTensorValue([x for x in inpts if 'beta/read:0' in x][0], operations)
    mean = findTensorValue([x for x in inpts if 'moving_mean/read:0' in x][0], operations)
    var = findTensorValue([x for x in inpts if 'moving_variance/read:0' in x][0], operations)

    variance = var + eps
    scale = np.reciprocal(np.sqrt(variance)) * scale_param
    bias = offset - (mean * scale)

    x.assignMultiplier(scale)
    x.assignBiasBeta(bias)

    return [x]
