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

from ctypes import *

from mvnctools.Models.StageDefinitions.Op import *
from mvnctools.Models.Blob import helper_parseBuffer
from mvnctools.Controllers.BlobBuilder import *


class Relu(Op):

    def __init__(self):
        super().__init__("Relu")

    def specific_details_push(self, target_container, instance):
        target_container.push("opX",  Value(c_float(instance.post_param1)))
        target_container.push("post_strideX",  Value(c_uint32(instance.post_strideX)))
        target_container.push("post_strideY",  Value(c_uint32(instance.post_strideY)))

    def adapt_fields(self, emulator, layer):

        from mvnctools.Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.

        emulator.post_param1 = layer.negativeSlope
        emulator.post_strideX = 0
        emulator.post_strideY = 0
