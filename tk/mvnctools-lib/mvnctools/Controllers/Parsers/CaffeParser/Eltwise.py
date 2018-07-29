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


from mvnctools.Controllers.Parsers.Parser.Eltwise import Eltwise

def load(obj, parsedNetworkObj):
    # This operation at least two input tensors
    assert(len(obj.bottom) > 1)
    assert(len(obj.top) == 1)

    x = Eltwise(obj.name, obj.bottom, obj.top)

    # Determine the operation
    eltwise_type = obj.eltwise_param.operation
    if eltwise_type == 0:
        x.loadType(Eltwise.Type.WPROD)
    elif eltwise_type == 1:
        x.loadType(Eltwise.Type.WSUM)
    elif eltwise_type == 2:
        x.loadType(Eltwise.Type.WMAX)
    else:
        assert(False)

    # Caffe supports weights only in Eltwise SUM. If the
    # operation is SUM, the weights for all the blobs are
    # needed.
    try:
        if x.getType() == Eltwise.Type.WSUM:
            coeffs = list(obj.eltwise_param.coeff)
            x.loadCoefficients(coeffs)
    except:
        print("Eltwise {} does not have coefficient. Use the default one".format(obj.name))  # TODO: Properly handle


    return [x]