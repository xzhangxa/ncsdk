
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

from mvnctools.Controllers.Parsers.Parser.Normalize import Normalize

def load(obj, parsedNetworkObj):
    # This operation supports only one inputTensor and one
    # outputTensor
    assert(len(obj.bottom) == 1)
    assert(len(obj.top) == 1)

    normalize_layer = Normalize(obj.name, obj.bottom, obj.top)
    normalize_layer.load_parameters(obj.norm_param)
    normalize_layer.load_scales(parsedNetworkObj.params[obj.name][0].data)

    return [normalize_layer]
