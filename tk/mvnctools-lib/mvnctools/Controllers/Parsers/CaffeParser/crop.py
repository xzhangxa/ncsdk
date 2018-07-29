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


from mvnctools.Controllers.Parsers.Parser.crop import Crop
import numpy as np

def load(obj, parsedNetworkObj):
    x = Crop(obj.name, obj.bottom, obj.top)

    crop_axis = obj.crop_param.axis
    if crop_axis < 0:   # Minus indexing valid in caffe.
        crop_axis += 4

    assert crop_axis != 0, "Cannot crop on batch"

    crop_offset = np.array([0, 0, 0], np.dtype("<u4"))  # Initialize to default Offset

    for offset_i in range(0, 3):
        if offset_i >= crop_axis - 1:
            if len(obj.crop_param.offset) == 1:
                crop_offset[offset_i] = obj.crop_param.offset[0]
            else:
                crop_offset[offset_i] = \
                    obj.crop_param.offset[offset_i - (crop_axis - 1)]

    # Caffe specifies Offset as (N)CHW
    # Myriad expects (N)HWC
    # So we do a quick conversion
    crop_offset = np.array([crop_offset[1], crop_offset[2],
                           crop_offset[0]], dtype=np.dtype("<u4"))

    x.setOffset(crop_offset)

    return [x]