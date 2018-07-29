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


def load(obj, parsedNetworkObj):

    x = BatchNorm(obj.name, obj.bottom, obj.top)

    print(parsedNetworkObj.params[obj.name])
    trained_vals = tuple(parsedNetworkObj.params[obj.name])

    global_stats = obj.batch_norm_param.use_global_stats
    epsilon = obj.batch_norm_param.eps

    _mean = trained_vals[0].data
    _variance = trained_vals[1].data
    moving_avg_factor = trained_vals[2].data

    if moving_avg_factor == 0:
        mean_ = np.zeros(_mean.shape)
        var = _variance + epsilon
    else:
        mean_ = _mean * (1 / moving_avg_factor[0])
        var = _variance * (1 / moving_avg_factor[0]) + epsilon

    multiplier = np.reciprocal(np.sqrt(var))
    bias_beta = - mean_ * multiplier

    x.assignMultiplier(multiplier)
    x.assignBiasBeta(bias_beta)

    return [x]
