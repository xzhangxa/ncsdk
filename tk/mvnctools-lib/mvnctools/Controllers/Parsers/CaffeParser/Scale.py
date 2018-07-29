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
from mvnctools.Controllers.Parsers.Parser.Scale import Scale


def load(obj, parsedNetworkObj):

    x = Scale(obj.name, obj.bottom, obj.top)

    trainedParameters = tuple(parsedNetworkObj.params[obj.name])

    axis = obj.scale_param.axis
    num_axes = obj.scale_param.num_axes
    filler = obj.scale_param.filler

    # Bias
    # biasTerm = obj.scale_param.bias_term

    # When bias exists, Caffe ignores the `bias_term` field.
    if len(trainedParameters) > 1:
        x.assignBiasBeta(trainedParameters[1].data.astype(dtype=np.float16))

    x.assignMultiplier(trainedParameters[0].data.astype(dtype=np.float16))

    return [x]
