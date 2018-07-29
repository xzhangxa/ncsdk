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
from mvnctools.Models import Layouts

def load(obj, parsedNetworkObj):
    # Determine the concat axis:
    # axis == 1 -> concat channels
    # axis == 2 -> concat height
    # axis == 3 -> concat width
    # Note: When we concat over an axis, the dimensions
    # of the remaining axes stay the same. Only the
    # dimension of the axis over which we concatenate changes.

    x = Concat(obj.name, obj.bottom, obj.top)

    axis = obj.concat_param.axis
    concat_dim = obj.concat_param.concat_dim

    if axis >= 0:
        if concat_dim != 1 and axis == 1:
            axis = concat_dim
        elif concat_dim != 1 and axis != 1:
            raise Exception("Mismatch in deprecated concat_dim and current field axis")
    else:
        # Find the explicit axis
        bottomName = obj.bottom[0]
        shape = parsedNetworkObj.blobs[bottomName].data.shape
        axis = len(shape) - axis

    x.loadConcatAxis(axis)

    return [x]
