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


from mvnctools.Controllers.Parsers.Parser.Slice import Slice

def load(obj, parsedNetworkObj):
    # Determine the slice axis:
    # axis == 1 -> slice channels
    # axis == 2 -> slice height
    # axis == 3 -> slice width
    # Note: When we slice over an axis, the dimensions
    # of the remaining axes stay the same. Only the
    # dimension of the axis over which we slice changes.

    x = Slice(obj.name, obj.bottom, obj.top)

    axis = obj.slice_param.axis
    slice_dim = obj.slice_param.slice_dim

    if axis >= 0:
        # If axis is non negative, the deprecated field and
        # the current field must match.
        assert(axis == slice_dim)
    else:
        # Find the explicit axis
        topName = obj.top[0]
        shape = parsedNetworkObj.blobs[topName].data.shape
        axis = len(shape) - axis

    x.loadSliceAxis(axis)

    # Determine the ranges where slice happens
    portions = []
    for name in x.getOutputTensorNames():
        shape = parsedNetworkObj.blobs[name].data.shape
        portions.append(shape[axis])

    x.loadSlicedPortions(tuple(portions))

    return [x]