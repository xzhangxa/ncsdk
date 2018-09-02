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


class LRN(Op):

    def __init__(self):
        super().__init__("LRN")

    def specific_details_push(self, target_container, instance):
        target_container.push("size", Value(c_uint32(instance.radixX)))
        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)
        helper_parseBuffer("bias", target_container, instance.biasBUF)


    def adapt_fields(self, emulator, layer):

        from mvnctools.Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from mvnctools.Controllers.Tensor import PopulatedTensor  # TODO: Fix Imports.

        emulator.radixX = layer.getSquareKernelSize()

        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF    = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF  = BufferEmulator(o.resolve())

        emulator.tapsBUF    = BufferEmulator(None)

        b = PopulatedTensor(np.array((
            layer.getK(),
            layer.getAlpha(),
            layer.getBeta(),
            0
        )).astype(np.float16))
        b.setLayout((0, 1, 2, 3))
        b.setDatatype(np.float16)
        emulator.biasBUF    = BufferEmulator(b.resolve())
