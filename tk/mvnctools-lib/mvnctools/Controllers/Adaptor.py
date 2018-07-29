
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

import sys
import numpy as np
from mvnctools.Models.NetworkStage import *
from mvnctools.Models.Network import *
from mvnctools.Models.Blob import Blob
from mvnctools.Models.MyriadParam import MyriadParam
from mvnctools.Models.EnumDeclarations import StageType, StorageOrder, MemoryIndex, ErrorTable
from mvnctools.Models.StageDefinitions.OpManager import get_op_definition
from mvnctools.Controllers.Tensor import Tensor, UnpopulatedTensor
from mvnctools.Controllers.FileIO import override_global_buffer, override_global_bssbuffer
from mvnctools.Controllers.MiscIO import readOptimisationMask
from mvnctools.Controllers.Globals import BLOB_MAJOR_VERSION, BLOB_MINOR_VERSION, BLOB_PATCH_VERSION
from mvnctools.Controllers.EnumController import throw_warning
from mvnctools.Controllers.Parsers.Parser.Input import Input
from mvnctools.Controllers.Parsers.Parser.ReLU import ReLU, LeakyReLU
from mvnctools.Controllers.Parsers.Parser.ELU import ELU
from mvnctools.Controllers.Parsers.Parser.PReLU import PReLU
from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D, Deconvolution
from mvnctools.Controllers.Parsers.Parser.Pooling import Pooling
from mvnctools.Controllers.Parsers.Parser.LRN import LRN
from mvnctools.Controllers.Parsers.Parser.Eltwise import Eltwise
from mvnctools.Controllers.Parsers.Parser.Concat import Concat
from mvnctools.Controllers.Parsers.Parser.Conversion import Conversion
from mvnctools.Controllers.Parsers.Parser.Hw import HwConvolution, HwPooling, HwConvolutionPooling, HwFC
from mvnctools.Controllers.Parsers.Parser.InnerProduct import InnerProduct
from mvnctools.Controllers.Parsers.Parser.Softmax import Softmax
from mvnctools.Controllers.Parsers.Parser.Sigmoid import Sigmoid
from mvnctools.Controllers.Parsers.Parser.Permute import Permute, PermuteFlatten
from mvnctools.Controllers.Parsers.Parser.Normalize import Normalize
from mvnctools.Controllers.Parsers.Parser.PriorBox import PriorBox
from mvnctools.Controllers.Parsers.Parser.DetectionOutput import DetectionOutput
from mvnctools.Controllers.Parsers.Parser.Flatten import Flatten
from mvnctools.Controllers.Parsers.Parser.Reshape import Reshape
from mvnctools.Controllers.Parsers.Parser.NoOp import Identity
from mvnctools.Controllers.Parsers.Parser.Bias import Bias
from mvnctools.Controllers.Parsers.Parser.tan_h import TanH
from mvnctools.Controllers.Parsers.Parser.Scale import Scale
from mvnctools.Controllers.Parsers.Parser.Square import Square
from mvnctools.Controllers.Parsers.Parser.crop import Crop

import mvnctools.Models.Layouts as Layouts

from functools import reduce
from operator import mul
from collections import OrderedDict


def convertLayouttoStorageEnum(newForm):
    """
        Map Constant Layout representations to StorageOrder enums.
    """
    layout_storageorder_map = {
        Layouts.NCHW: StorageOrder.orderZYX,
        Layouts.NHCW: StorageOrder.orderYZX,
        Layouts.NWCH: StorageOrder.orderXZY,
        Layouts.NCWH: StorageOrder.orderZXY,
        Layouts.NHWC: StorageOrder.orderYXZ,
        Layouts.NWHC: StorageOrder.orderXYZ
    }

    if newForm not in layout_storageorder_map:
        raise KeyError('Unrecognised Layout when mapping to StorageOrder.')

    # Error might occur if key doesn't exist
    return layout_storageorder_map[newForm]

def OpClassToEnum(cls):
    """
        Converts an Operation subclass from parser into Stagetype Enums
    """

    # Subtypes specified in old parser
    if type(cls) == Pooling:
        if cls.getType() == Pooling.Type.MAX:
            cls = "maxpooling"
        elif cls.getType() == Pooling.Type.AVE:
            cls = "averagepooling"

    if type(cls) == LRN:
        if cls.getType() == LRN.Type.ACROSS:
            cls = "acrossLRN"
        elif cls.getType() == LRN.Type.WITHIN:
            cls = "withinLRN"
        elif cls.getType() == LRN.Type.WITHIN_PARTIAL:
            cls = "WithinChannelPartial"

    if type(cls) == Eltwise:
        if cls.getType() == Eltwise.Type.WMAX:
            cls = "eltmax"
        elif cls.getType() == Eltwise.Type.WSUM:
            cls = "eltsum"
        elif cls.getType() == Eltwise.Type.WPROD:
            cls = "eltprod"

    if type(cls) == ReLU:
        if cls.reluX != 0:
            cls = 'ReLUX'

    # TODO: Error if key not found
    cls_enum_mapping = {
        Input: StageType.none,
        Bias: StageType.bias,
        Deconvolution: StageType.deconvolution,
        PReLU: StageType.prelu,
        Sigmoid: StageType.sigmoid,
        Convolution2D: StageType.convolution,
        "maxpooling": StageType.max_pooling,
        "averagepooling": StageType.average_pooling,
        ReLU: StageType.relu,
        LeakyReLU: StageType.leaky_relu,
        "ReLUX"         : StageType.relu_x,
        ELU: StageType.elu,
        "acrossLRN": StageType.LRN,
        "WithinChannelPartial": StageType.innerlrn,
        "eltmax": StageType.eltwise_max,
        "eltsum": StageType.eltwise_sum,
        "eltprod": StageType.eltwise_prod,
        Concat: StageType.copy,
        Conversion: StageType.storage_order_convert,
        HwConvolution: StageType.myriadX_convolution,
        HwPooling: StageType.myriadX_pooling,
        HwConvolutionPooling: StageType.myriadX_convolution,
        HwFC: StageType.myriadX_fully_connected_layer,
        InnerProduct: StageType.fully_connected_layer,
        Softmax: StageType.soft_max,
        ConvolutionDepthWise2D: StageType.depthwise_convolution,
        PReLU: StageType.prelu,
        Permute: StageType.permute,
        PermuteFlatten: StageType.permute_flatten,
        Normalize: StageType.normalize,
        PriorBox: StageType.prior_box,
        DetectionOutput: StageType.detection_output,
        Flatten: StageType.toplanemajor,
        Reshape: StageType.reshape,
        Identity: StageType.none,
        TanH: StageType.tanh,
        Scale: StageType.scale,
        Square: StageType.square,
        Crop: StageType.crop,
    }

    if type(cls) is str:
        return cls_enum_mapping[cls]
    elif type(cls) in cls_enum_mapping.keys():
        return cls_enum_mapping[type(cls)]
    else:
        print("No mapping present for layer", cls)


class ArgumentEmulator():
    def __init__(self):
        self.conf_file = "optimization.conf"


class Adaptor():
    """
        High Level Class containing Emulators of old code structures.
    """

    def __init__(self, inputTensorsData, inputTensors, outputTensors):
        """
            Set up a NetworkEmulator for Transformation
        """
        self.net = NetworkEmulator(inputTensorsData, inputTensors, outputTensors)

    def transform(self, layers):
        """
            Convert from current Layer representations to
            the older NetworkStage representation.

            Transformed fields are based on requirements from Blob.py
        """
        for l in layers:
            if l.getImplicit() is False:
                self.net.attach(NetworkStageEmulator(l))

        self.net.close()

    def serialize(self, args, myr):
        """
            Interface with the graph file serialization.
        """
        network_name = "new_net"    # Take from Parser
        pwd = ""                    # This field isn't even used...
        bfile = args.blob_name       # TODO: Needs standardizing across all interfaces

        blob_file = Blob([BLOB_MAJOR_VERSION, BLOB_MINOR_VERSION, BLOB_PATCH_VERSION], network_name, pwd, myr, self.net, bfile)
        blob_file.generate_v2(args)

        return self.net, blob_file


class NetworkEmulator():
    """
        Pretends to be Network.py
    """
    def __init__(self, inputTensorsData, inputTensors, outputTensors):
        self.count = 0          # Amount of Layers in this network.
        self.stageslist = []     # Old construct to hold all 'NetworkStage's

        # Ensure compatibility with MiscIO
        self.outputNeedsTransforming = True

        le = LayoutEmulator()
        le.setNetworkReference(self)
        self.inputTensorsData = inputTensorsData

        # List of tensors
        self.inputTensors = inputTensors

        # Pairs of tensors and whether it is DetectionOutput
        self.outputTensors = outputTensors

        # TODO: Since we support only one input and output:
        self.inputTensor = self.inputTensorsData[0]
        self.outputTensor = self.outputTensors[0][0].getShape()
        self.inputTensorLayout = self.inputTensors[0].getLayout()
        self.outputTensorLayout = self.outputTensors[0][0].getLayout()
        self.outputIsSsdDetOut = self.outputTensors[0][1]
        self.outputTensorShape = self.outputTensor

        # Transform from canonical format to requested format (channel minor).
        self.inputTensor = np.moveaxis(
                                self.inputTensor,
                                1,
                                3).ravel().reshape(
                                self.inputTensor.shape[2],
                                self.inputTensor.shape[3],
                                self.inputTensor.shape[1])

    def close(self):
        """
        Calculates any remaining fields needing population
        """

        # Complete Buffer Tracking
        le = LayoutEmulator()
        le.close()

    def attach(self, layerEmulation):
        """
        Emulating the attach function, takes care of tracker objects
        """
        self.count += 1

        le = LayoutEmulator()
        le.setNetworkReference(self)

        # Attach this layer into the stageslist.
        self.stageslist.append(layerEmulation)

    def bundle(self):
        """
            Unimplemented function for compatibility
            (Used in blob.generate_v2)
        """
        pass

    def gather_metrics(self, timings):
        prev_len = 0
        # The stages have to be processed in the order they have been
        # created, not in a tree-based order, otherwise we risk not
        # respecting dependencies
        for stage in self.stageslist:
            stage.calculate_metrics(timings[prev_len:])
            prev_len = prev_len + 1


class NetworkStageEmulator():
    """
        Pretends to be NetworkStage.py
    """
    def __init__(self, layer):
        self.name = layer.getName().stringifyOriginalName()
        self.original_representation = layer
        self.op = OpClassToEnum(layer)
        self.unprocessed_name = layer.getName().stringifyName()
        self.BWs = -99
        self.flops = -99
        self.ms = -99
        argEmu = ArgumentEmulator()
        self.optMask = readOptimisationMask("test", self, None, argEmu)

        if self.optMask != 0x80000000:
            throw_warning(ErrorTable.no_check_optimization)
            argEmu.compatibility_warnings()

        self.definition = get_op_definition(self.op, force_op=True)
        self.preOp = StageType.none           # Removing these

        try:
            if self.original_representation.biasEnabled() and \
                not type(self.original_representation) in [HwConvolution, HwPooling, HwConvolutionPooling, HwFC] :
                self.postOp = StageType.bias          # Removing these
            else:
                self.postOp = StageType.none
        except:
            self.postOp = StageType.none

        self.specific_fields()

    def specific_fields(self):
        """
        Fill out all fields specific to an Op, as described in StageDefinitions
        """
        _or = self.original_representation

        self.definition.adapt_fields(self, _or)

    def calculate_metrics(self, timings):
        self.ms = timings[0]

class BufferEmulator():
    """
        Pretends to be a Buffer from FileIO
    """

    def __init__(self, rt, track=True):
        """
            Converts from a ResolvedTensor (Buffer)
            into the BufferEmulator
        """

        # Keep the resolved tensor that get converted into buffer
        self.resolvedTensor = rt

        if rt is None:
            # An 'empty buffer' i.e. one that is not used but
            # we need to keep for compatibility reasons
            self.x = self.y = self.z = 0
            self.x_s = self.y_s = self.z_s = 0
            self.offset = 0
            self.location = 0
            self.dtype = 0
            self.order = StorageOrder.orderYXZ
        else:

            """
                The existing buffer object supports only 3 dimensions (X, Y, Z).
                However, `Tensor`s are 4D, so we ignore axis=0 (which is BatchNumber
                for input/output, but is needed for trained parameters).
            """
            self.x = rt.dimensions[3]
            self.y = rt.dimensions[2]
            self.z = rt.dimensions[1]

            self.x_s = rt.strides[3]
            self.y_s = rt.strides[2]
            self.z_s = rt.strides[1]

            # TODO Asserts

            if track:
                le = LayoutEmulator()
                le.increment(rt)
                idx = le.index_of(rt)
                self.offset = idx + rt.local_offset * np.dtype(rt.dtype).itemsize
                self.location = le.getStorageLocationValue(rt)
            else:
                self.offset = 0
                self.location = 0

            self.dtype = 0     # TODO: Confirm no other values for this field are used.
            self.order = convertLayouttoStorageEnum(rt.layout)

    def __getitem__(self, key):
        return self.resolvedTensor.data[0,0,0,key]


    def pprint(self):
        """
        Pretty Print the Buffer
        """
        print("""

                  +-------------------+
                +-------------------+ |
              +-------------------+ | |     X = {} Striding = {}
            +-------------------+ | | |     Y = {} Striding = {}
            |                   | | | |     Z = {} Striding = {}
            |      Buffer       | | | |     Order = {} dtype = {}
            |       Obj         | | | |
            |                   | | +-+     Location ID: {}
            |                   | +-+       Offset:      {}
            |                   +-+
            +-------------------+
            """.format(self.x, self.x_s, self.y, self.y_s, self.z, self.z_s, self.order, self.dtype, self.location, self.offset)
        )


class LayoutEmulator():
    """
        Storage Object for Buffers.
        Uses Singleton internally
    """
    class __LayoutEmulator():

        def __init__(self):
            """
                __LayoutEmulator is a container around an array that tracks all
                resolved tensors that are to be written to file.
                The constructor just initializes that array.
            """
            self.bufferArray = None
            self.bssBufferArray = None
            self.bssActivatedTopTensors = OrderedDict()

        def increment(self, rt):
            """
                Adds a resolved tensor to the tracking array.
                TODO: Check if we need to flatten this data
            """
            if self.__checkExternalPlacement(rt):
                return

            if rt.opaque:
                if self.bufferArray is None:
                    self.bufferArray = [rt]
                else:
                    self.bufferArray.extend([rt])
            else:
                if self.bssBufferArray is None:
                    self.bssBufferArray = [rt]
                else:
                    self.bssBufferArray.extend([rt])

        def __getitem__(self, key):
            """
                Override the default getitem so that we can access buffers like:
                LayoutEmulator["ID of Resolved Tensor"]
                Note that only top level tensors are added to the tracker.
            """

            if self.bufferArray is not None:
                for idx, b in enumerate(self.bufferArray):
                    if key == b.topID:
                        return idx
            if self.bssBufferArray is not None:
                for idx, b in enumerate(self.bssBufferArray):
                    if key == b.topID:
                        return idx

            return None

        def __checkExternalPlacement(self, tensor):
            if isinstance(tensor.original_tensor, UnpopulatedTensor):
                tensorName = tensor.getTopEncloserRecursive().getName().stringifyName()
                externalNames = [t.getName().stringifyName() for t in self.NetRef.inputTensors] + \
                                [t.getName().stringifyName() for t, _ in self.NetRef.outputTensors]
                return tensorName in externalNames
            return False

        def __align64(self, npArray):
            """Very basic alignment, without changing the blob generation
            """
            origSize = npArray.size

            align_size = 64 # BYTE SIZE
            # Account for the byte size of each element in the array
            align_size = int(align_size / np.dtype(npArray.dtype).itemsize)
            rem = origSize % align_size
            newSize = origSize if (rem == 0) else origSize + (align_size - rem)

            return np.pad(npArray, (0, (newSize - origSize)), mode="constant")

        def index_of(self, rt):
            """
                Returns the 'global offset' of a resolved tensor in the buffer array
            """

            if self.__checkExternalPlacement(rt):
                return 0

            if rt.opaque:
                if self.bufferArray:
                    idx = self.bufferArray.index(rt)

                    offset = 0
                    for rt in self.bufferArray[:idx]:
                        offset += self.__align64(rt.data.astype(rt.dtype).flatten()).size * np.dtype(rt.dtype).itemsize

                    return offset
                else:
                    return 0
            else:

                # Take all the input and output tensors
                filteredBssBufferArray = list()

                for i in self.bssBufferArray:
                    if self.__checkExternalPlacement(i):
                        continue
                    filteredBssBufferArray.append(i)

                # Get the top encloser of a resolved tensor
                topEncloser = rt.getTopEncloserRecursive()

                if topEncloser.ID not in self.bssActivatedTopTensors:
                    if not self.__checkExternalPlacement(rt):
                        if len(self.bssActivatedTopTensors) > 0:
                            # Get the offset of the last tensor pushed into the
                            # dictionary:
                            _, (lastEntryTensor, lastEntryOffset) = list(self.bssActivatedTopTensors.items())[-1]
                            lastEntryTensorSize = self.__align64(np.empty(lastEntryTensor.getShape(),
                                                    dtype=lastEntryTensor.getDatatype()).flatten()).size

                            self.bssActivatedTopTensors[topEncloser.ID] = (topEncloser, lastEntryTensorSize + lastEntryOffset)
                        else:
                            self.bssActivatedTopTensors[topEncloser.ID] = (topEncloser, 0)

                try:
                    _, entryOffset = self.bssActivatedTopTensors[topEncloser.ID]
                    # TODO: Don't assume that the BSS tensors are all fp16
                    return entryOffset * np.dtype(np.float16).itemsize
                except KeyError:
                    # For external names the try block throws a KeyError exception
                    return 0

        def close(self):
            """
            Override FileIO's global buffer tracker with this one.
            """

            if self.bufferArray is not None:
                buffer = [self.__align64(x.data.astype(x.dtype).flatten()) for x in self.bufferArray]
                override_global_buffer(buffer)

        def getStorageLocationValue(self, rt):
            if rt.opaque:
                return MemoryIndex.blob.value
            else:
                topEncloserName = rt.getTopEncloserRecursive().name.stringifyName()
                isUnpopulated = isinstance(rt.original_tensor, UnpopulatedTensor)
                if isUnpopulated and topEncloserName in [t.getTopEncloserRecursive().getName().stringifyName() for t in self.NetRef.inputTensors]:
                    return MemoryIndex.input.value
                elif isUnpopulated and topEncloserName in [t.getTopEncloserRecursive().getName().stringifyName() for t, _ in self.NetRef.outputTensors]:
                    return MemoryIndex.output.value
                else:
                    return MemoryIndex.workbuffer.value     # TODO: Increment

        def setNetworkReference(self, net):
            self.NetRef = net

    # Singleton contaner
    instance = None

    def __init__(self):
        """
            Ensures we only return the sole instance.
        """
        if not LayoutEmulator.instance:
            LayoutEmulator.instance = LayoutEmulator.__LayoutEmulator()

    # From here onwards is forwarding functions

    def getStorageLocationValue(self, rt):
        return LayoutEmulator.instance.getStorageLocationValue(rt)

    def __getitem__(self, key):
        return LayoutEmulator.instance[key]

    def setNetworkReference(self, net):
        return LayoutEmulator.instance.setNetworkReference(net)

    def increment(self, rt):
        return LayoutEmulator.instance.increment(rt)

    def index_of(self, rt):
        return LayoutEmulator.instance.index_of(rt)

    def close(self):
        return LayoutEmulator.instance.close()
