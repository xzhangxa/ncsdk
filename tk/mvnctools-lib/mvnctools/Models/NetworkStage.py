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
from ctypes import *
from mvnctools.Controllers.MiscIO import *
from mvnctools.Controllers.DataTransforms import *
from mvnctools.Controllers.EnumController import *
from mvnctools.Models.EnumDeclarations import *
from mvnctools.Views.Graphs import *
from mvnctools.Controllers.CnnHardware import *

from linecache import getline

import mvnctools.Controllers.Globals as GLOBALS

from mvnctools.Models.StageDefinitions.OpManager import get_op_definition

from mvnctools.Controllers.PingPong import getManualHwSchedule, get_null_terminating_name
import mvnctools.Controllers.Globals as GLOBALS

search_list = []


class NetworkStage:

    def __init__(
            self,
            name,
            top,
            s_order,
            pad_y,
            pad_x,
            pad_type,
            dtype,
            precision,
            op_type,
            op_y,
            op_x,
            sy,
            sx,
            x,
            y,
            c,
            fh,
            fw,
            k,
            taps,
            taps_order,
            bias,
            pre_op_type,
            post_op_type,
            post_1,
            post_sx,
            post_sy,
            slicing=None,
            myriad_config=None,
            args=None,          # Currently Not Optional.
            opParams=None,
            new_x=0,
            new_y=0,
            new_c=0,
            network=None,        # If we do not have args, use this.
        ):
        self.changeName(name)

        # Historically cocat layer was working only for axis1 i.e channels.
        # Concat on axis 2 is available only for CaffeParser and concat_axis
        # needs to default to 1 in order not to break the TensorFlowParser.
        self.concat_axis = 1

        # mvTensor cannot deal with such convolution, which is equivalent to fc
        if (op_type == StageType.convolution and op_x == 1 and op_y == 1
                and x == 1 and y == 1):
            op_type = StageType.fully_connected_layer

        self.definition = get_op_definition(op_type)

        # Make sure that the first conv olution layer is padded
        if GLOBALS.USING_MA2480 and \
            top == None and \
            op_type in [StageType.convolution] and \
            c == 3 and \
            taps.shape[1] == 3:
            padded_weights_slice = np.zeros([taps.shape[0], 1, taps.shape[2], taps.shape[3]])
            taps = np.append(taps, padded_weights_slice, axis=1)
            PadMXWeights = True
        else:
            PadMXWeights = False

        if args is not None:
            self.network = args.network
        elif network is not None:
            self.network = network
        else:
            print("No Network Passed to Network Stage")
            quit()
        self.top = top
        self.tail = []
        self.op = op_type
        self.radixX = op_x
        self.radixY = op_y
        self.padX = pad_x
        self.padY = pad_y
        self.alias = [name]

        if self.radixX == -1 and self.radixY == -1:
            # Global Operation
            self.radixX = x
            self.radixY = y

        self.strideX = sx
        self.strideY = sy

        self.optMask = readOptimisationMask(name, self, myriad_config, args)

        self.inputStrideX = 2 * c  # Concat Stride
        self.inputStrideY = 2 * c * x
        self.inputStrideZ = 2
        self.inputOffset = 0
        if slicing:
            if top is None:
                for slice in slicing:
                    if slice[0] is None:
                        c = slice[2] - slice[1]
                        self.inputOffset = slice[1] * 2
                        break
            else:
                for input in top:
                    for slice in slicing:
                        if slice[0] == input:
                            c = slice[2] - slice[1]
                            self.inputOffset = slice[1] * 2
                            break
        if (op_type == StageType.eltwise_sum or op_type ==
                StageType.eltwise_prod or op_type == StageType.eltwise_max):
            # Ignore given k, which could be wrong if it ignored slicing
            k = c

        self.inputDimX = x
        self.inputDimY = y
        self.inputDimZ = c


        if PadMXWeights:
            c = 4

        if fw or fh or c or k:
            # TODO: Confirm that the current Op has taps before doing this...
            self.tapDimX = fw * fh
            self.tapDimY = c
            self.tapDimZ = k
        else:
            self.tapDimX = 0
            self.tapDimY = 0
            self.tapDimZ = 0

        if PadMXWeights:
            c = 3

        self.outputDimZ = k
        if self.op in [StageType.fully_connected_layer]:
            self.inputDimX = 1
            self.inputDimY = 1
            self.inputDimZ = x * y * c

            if PadMXWeights:
                c = 4
            self.tapDimX = 1
            self.tapDimY = x * y * c
            if PadMXWeights:
                c = 3

            self.outputDimX = 1
            self.outputDimY = 1

        elif self.op in [StageType.convolution, StageType.depthwise_convolution, StageType.max_pooling, StageType.average_pooling]:
            if pad_type == PadStyle.tfsame:
                self.outputDimX = math.ceil(x / self.strideX)
                self.outputDimY = math.ceil(y / self.strideY)
            elif pad_type == PadStyle.tfvalid:
                self.outputDimX = math.ceil((x - self.radixX + 1) / self.strideX)
                self.outputDimY = math.ceil((y - self.radixY + 1) / self.strideY)
            # Caffe, convolution uses floor
            elif self.op in [StageType.convolution, StageType.depthwise_convolution]:
                if self.radixX == 1 and self.radixY == 1 and self.padX == 1 and self.padY == 1:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        'Padding 1 not supported for 1x1 convolution in ' + name)
                # This code should be executed only for caffe layers.
                radix_x_extent = self.radixX
                radix_y_extent = self.radixY
                dilation = 1
                if opParams is not None:
                    dilation = opParams[0]
                    radix_x_extent = dilation * (self.radixX - 1) + 1
                    radix_y_extent = dilation * (self.radixY - 1) + 1

                self.outputDimX = (x + 2 * self.padX - radix_x_extent) // self.strideX + 1
                self.outputDimY = (y + 2 * self.padY - radix_y_extent) // self.strideY + 1

            else:  # Caffe, pooling uses ceil
                self.outputDimX = math.ceil((x + 2 * self.padX - self.radixX) / self.strideX) + 1
                self.outputDimY = math.ceil((y + 2 * self.padY - self.radixY) / self.strideY) + 1
                self.outputDimX = min(self.outputDimX, math.ceil((x + self.padX) / self.strideX))
                self.outputDimY = min(self.outputDimY, math.ceil((y + self.padY) / self.strideY))

        elif self.op in [StageType.deconvolution]:
            if pad_type == PadStyle.tfsame:
                pad_X = math.floor(self.radixX / 2)
                pad_Y = math.floor(self.radixY / 2)
            elif pad_type == PadStyle.tfvalid:
                pad_X = self.radixX - 1
                pad_Y = self.radixY - 1
            elif pad_type == PadStyle.caffe:
                pad_X = self.padX
                pad_Y = self.padY
            else:
                pad_X = 0
                pad_Y = 0

            self.outputDimX = self.strideX * (x - 1) + self.radixX - 2 * pad_X
            self.outputDimY = self.strideY * (y - 1) + self.radixY - 2 * pad_Y

        elif self.op == StageType.toplanemajor:
            self.outputDimX = 1
            self.outputDimY = 1
            self.outputDimZ = x * y * c

        elif self.op in [StageType.reshape]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c

            if (new_x == 0):
                self.outputDimX = x
            elif (new_x > 0):
                self.outputDimX = new_x

            if (new_y == 0):
                self.outputDimY = y
            elif (new_y > 0):
                self.outputDimY = new_y

            if (new_c == 0):
                self.outputDimZ = c
            elif (new_c > 0):
                self.outputDimZ = new_c

            if (new_x == -1):
                self.outputDimX = x * y * \
                    c // (self.outputDimY * self.outputDimZ)
            if (new_y == -1):
                self.outputDimY = x * y * \
                    c // (self.outputDimX * self.outputDimZ)
            if (new_c == -1):
                self.outputDimZ = x * y * \
                    c // (self.outputDimX * self.outputDimY)

        elif self.op in [StageType.reorg]:
            stride = opParams[0]
            self.outputDimX = int(self.inputDimX / stride)
            self.outputDimY = int(self.inputDimY / stride)
            self.outputDimZ = int(self.inputDimZ * stride * stride)

        elif self.op in [StageType.crop, StageType.permute]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c

        elif self.op in [StageType.storage_order_convert]:
            # The following lines are included for compatibility with the rest of the initialization function only.
            self.outputDimX = x
            self.outputDimY = y

        elif self.op in [StageType.prior_box]:
            self.tapDimX = int(opParams[1])
            self.tapDimY = int(opParams[0])

            opParams = opParams[2:]
            min_size_size = opParams[0]
            max_size_size = opParams[1]
            flip = int(opParams[4])
            aspect_ratio_size = opParams[2]
            aspect_ratio_size = aspect_ratio_size*2 if (flip) else aspect_ratio_size

            num_priors = (1 + aspect_ratio_size) * min_size_size + max_size_size

            self.outputDimX = 1
            self.outputDimY = int(x * y * num_priors * 4)
            self.outputDimZ = 2
        elif self.op in [StageType.detection_output]:
            self.outputDimX = 7
            # output[0,0,0] = contains the number of detections.
            # The rest of the elements on the first line are grabage.
            # The detections start at the second line and span maximum new_y lines
            # i.e. top_k lines at max.
            self.outputDimY = new_y + 1
            self.outputDimZ = 1
        else:
            self.outputDimX = x
            self.outputDimY = y

        self.output = np.zeros(
            (int(
                self.outputDimZ), int(
                self.outputDimY), int(
                self.outputDimX))).astype(
                    enum_as_dtype(dtype))

        self.tapStrideX = 2 * self.tapDimZ  # Concat Stride
        self.tapStrideY = 2 * self.tapDimZ
        self.tapStrideZ = 2

        self.outputStrideX = 2 * self.outputDimZ  # Concat Stride
        self.outputStrideY = 2 * self.outputDimZ * self.outputDimX
        self.outputStrideZ = 2

        # Provide accessible backups in case concat or similar stages
        # overwrites them.
        self.unprocessed_w = x
        self.unprocessed_h = y
        self.unprocessed_c = c
        self.unprocessed_k = k
        self.unprocessed_output = self.output  # Used for theoretical graph sizes

        self.datatype = dtype
        self.precision = precision

        self.data = np.zeros((int(self.inputDimZ),
                              int(self.inputDimY),
                              int(self.inputDimX)
                              )).astype(enum_as_dtype(dtype))

        self.taps = taps
        self.tapsPointer = 0  # We will fill them in generate
        self.tapsIndex = 0
        self.tapsOrder = taps_order

        self.bias = bias
        self.biasPointer = 0  # We will fill them in generate
        self.biasIndex = 0
        self.scale = None
        self.scalePointer = 0
        self.scaleIndex = 0
        self.opParams = opParams
        self.opParamsPointer = 0  # We will fill them in generate
        self.opParamsIndex = 0

        self.concatResult = False
        self.storageOrder = s_order
        self.padStyle = pad_type

        self.dataPointer = self.inputOffset
        self.dataIndex = 0

        self.outputPointer = 0
        self.outputIndex = 0

        if pre_op_type:
            self.preOp = pre_op_type
        else:
            self.preOp = StageType.none

        if post_op_type and post_op_type != StageType.none:
            self.postOp = post_op_type
            if post_1:
                self.post_param1 = post_1
            else:
                self.post_param1 = int(0)
            self.post_strideX = post_sx
            self.post_strideY = post_sy
        else:
            if (self.op in [StageType.convolution,
                            StageType.depthwise_convolution,
                            StageType.fully_connected_layer,
                            StageType.scale,
                            StageType.deconvolution]) and bias is not None:
                self.postOp = StageType.bias
            else:
                self.postOp = StageType.none
            self.post_param1 = 0
            self.post_strideX = 0
            self.post_strideY = 0

        # in the case of Reshape, the outputDims are used as parameters for reshape;
        # make sure that the parameter values are written to myriad
        if self.op in [StageType.reshape]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c

        if self.op in [StageType.eltwise_max, StageType.eltwise_sum, StageType.eltwise_prod]:
            self.tapLayout = self.definition.requirements["input"]["layout"]
            self.tapDimX = self.inputDimX
            self.tapDimY = self.inputDimY
            self.tapDimZ = self.inputDimZ
            self.recalculate_stride("taps")

        # Accuracy parameters
        default_scale = 256.0
        if hasattr(args, "accuracy_table"):
            if "ALL" in args.accuracy_table:
                self.accuracy_scale = args.accuracy_table["ALL"]
            elif name in  args.accuracy_table:
                self.accuracy_scale = args.accuracy_table[name]
            else:
                self.accuracy_scale = 256.0
        else:
            self.accuracy_scale = 256.0

        # Only to be used after myriad execution
        self.flops = None
        self.ms = None
        self.BWs = None
        self.isoutput = False
        self.isconcat = False

        self.pre_definition = get_op_definition(self.preOp)
        self.post_definition = get_op_definition(self.postOp)

        self.outputLayout = StorageOrder.orderYXZ

    def addBias(self, bias):
        if bias is not None:
            if self.bias is None:
                self.bias = bias
                self.postOp = StageType.bias
            else:
                self.bias = self.bias + bias

    def putBias(self):
        if self.bias is not None:
            self.biasPointer, self.biasBufferIndex = get_buffer(
                self.bias.astype(np.float16), self.datatype)
            self.biasIndex = MemoryIndex.blob.value

    def putScale(self, scale):
        if scale is not None:
            if self.scale is None:
                self.scale = scale
                self.scalePointer, self.scaleBufferIndex = get_buffer(
                    self.scale.astype(np.float16), self.datatype)
                self.scaleIndex = MemoryIndex.blob.value
            else:
                # There was a scale added from another operation, need to multiply values
                self.scale = self.scale * scale


    def putTaps(self):
        if self.taps is not None:
            self.tapsPointer, self.tapsBufferIndex = get_buffer(
                self.taps.astype(np.float16), self.datatype, hwAlignment=True, op=self.op, tapsOrder=self.tapsOrder)
            self.tapsIndex = MemoryIndex.blob.value

    def putOpParams(self):
        """ Puts the operation parameters in the blob buffer """
        if self.opParams is not None:
            self.opParamsPointer, self.opParamsBufferIndex = \
                get_buffer(self.opParams, DataType.fp32)

            self.opParamsIndex = MemoryIndex.blob.value

    def changeName(self, new_name):
        self.unprocessed_name = new_name
        self.name = set_string_range(new_name, 100).encode('ascii')

    def close(self):
        self.outputPointer = 0
        self.outputIndex = MemoryIndex.output

    def attach(self, stage):
        """
        Attaches a node to this one.
        :param stage:
        :return:
        """
        if (stage.op == StageType.convolution and self.op == StageType.depthwise_convolution and
            stage.radixX == 1 and stage.radixY == 1 and self.postOp == StageType.none):
            print('Fusing depthconv and conv in',self.unprocessed_name,'and',stage.unprocessed_name)
            #Create the weights for a convolution that does deptwhise convolution (inCH, outCH, kH, kW)
            taps = np.zeros([self.inputDimZ, self.tapDimZ, self.radixY, self.radixX], np.float32)
            multiplier = int(self.tapDimZ/self.tapDimY)
            for y in range(self.radixY):
                for x in range(self.radixX):
                    for c in range(self.tapDimY):
                        for i in range(multiplier):
                            taps[c,c*multiplier+i,y,x] = self.taps[y,x,c,i]
            #Turn them to [kH, kW, inCH, outCH) in order to be able to use matmul
            taps = taps.transpose(2,3,0,1)
            #Fuse the weights of the following 1x1 convolution into the just created weights
            stage.taps = np.matmul(taps,stage.taps[0,0])
            #Bring some data from the previous stage (self) to this one (stage) as we are saving this one
            #Saving the previous node would be simpler, but unfortunately the parser keeps track
            #of what's the latest created node (stage), so we must keep it
            stage.inputDimX = self.inputDimX
            stage.inputDimY = self.inputDimY
            stage.inputDimZ = self.inputDimZ
            stage.inputStrideX = self.inputStrideX
            stage.inputStrideY = self.inputStrideY
            stage.inputStrideZ = self.inputStrideZ
            if stage.supports_taps():
                stage.tapDimX = self.tapDimX
                stage.tapDimY = self.tapDimY
            else:
                stage.tapDimX = 0
                stage.tapDimY = 0
            stage.radixX = self.radixX
            stage.radixY = self.radixY
            stage.strideX = self.strideX
            stage.strideY = self.strideY
            stage.padStyle = self.padStyle
            stage.top = self.top
            stage.data = self.data
            stage.dataIndex = self.dataIndex
            stage.dataPointer = self.dataPointer
            #Remove self from network and change references
            self.network.count = self.network.count - 1
            self.network.stageslist.remove(self)
            stage.top = self.top
            if self in self.network.head:
                stage.network.storageOrder = stage.storageOrder
                self.network.head.remove(self)
                self.network.head.append(stage)
            else:
                for parents in self.network.search_several(self.top):
                    newtail = []
                    for p in parents.tail:
                        if p == self:
                            newtail.append(stage)
                    parents.tail = newtail
            return

        def is_fused(stage):
            # Disable convolution/pooling fusion when not possible
            input = (stage.inputDimX, stage.inputDimY, stage.inputDimZ)
            output = (stage.outputDimX, stage.outputDimY, stage.outputDimZ)
            kernel = (stage.radixX, stage.radixY)

            if not canFusePooling(input, output, kernel):
                return True
            try:
                return stage.fused_conv_pooling
            except AttributeError:
                return False


        if GLOBALS.USING_MA2480 and \
           (stage.op == StageType.max_pooling or stage.op == StageType.average_pooling) and \
           (self.op == StageType.convolution) and \
            stage.radixX == 2 and stage.radixY == 2 and \
            stage.strideX == 2 and stage.strideY == 2 and \
            not is_fused(self):
            #(self.inputDimX * self.inputDimY / 1024 > 128) and \
            print("Caught non-overlapping pooling after convolution")

            # self.outputDimX = stage.outputDimX
            # self.outputDimY = stage.outputDimY
            # self.outputDimZ = stage.outputDimZ

            # stage.op = StageType.none

            stage.fused_conv_pooling = True
            stage.fused_op = stage.op
            stage.fused_poolingRadixX = stage.radixX
            stage.fused_poolingRadixY = stage.radixY
            stage.op = StageType.convolution
            stage.taps = self.taps
            stage.inputDimX = self.inputDimX
            stage.inputDimY = self.inputDimY
            stage.inputDimZ = self.inputDimZ
            stage.inputStrideX = self.inputStrideX
            stage.inputStrideY = self.inputStrideY
            stage.inputStrideZ = self.inputStrideZ
            stage.tapDimX = self.tapDimX
            stage.tapDimY = self.tapDimY
            stage.radixX = self.radixX
            stage.radixY = self.radixY
            stage.strideX = self.strideX
            stage.strideY = self.strideY
            stage.padX = self.padX
            stage.padY = self.padY
            stage.padStyle = self.padStyle
            stage.top = self.top
            stage.data = self.data
            stage.dataIndex = self.dataIndex
            stage.dataPointer = self.dataPointer
            stage.postOp = self.postOp

            stage.post_param1 = self.post_param1
            stage.bias = self.bias

            #Remove self from network and change references
            self.network.count = self.network.count - 1
            self.network.stageslist.remove(self)
            stage.top = self.top
            if self in self.network.head:
                stage.network.storageOrder = stage.storageOrder
                self.network.head.remove(self)
                self.network.head.append(stage)
            else:
                for parents in self.network.search_several(self.top):
                    newtail = []
                    for p in parents.tail:
                        if p == self:
                            newtail.append(stage)
                    parents.tail = newtail

            return


        # This line is to build a correct graph after renaming due to absorption
        # When scale or batchnorm is absorbed into convolution, its name is appended
        # to the name of the convolution layer, so bottoms (here they are called tops,
        # wrong choice) of attached layers have to be renamed too
        # All the cases (attach, attach_several, attach_eltwise) are present in
        # test t19_UnitTest/NetworkConfig/AbsorptionRenaming.prototxt
        # What needs to be verified is the correct generation of the report
        # diagram
        stage.top = [self.unprocessed_name]
        self.tail.append(stage)
        if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value:
            self.outputPointer, self.outputIndex = get_zero_buffer(
                self.output, self.datatype)

        stage.dataPointer = stage.inputOffset + self.outputPointer      # Input Pointer
        stage.dataIndex = self.outputIndex

        if (stage.op != StageType.fully_connected_layer and not self.isconcat
                and self.op != StageType.reshape):
            stage.inputDimX, stage.inputDimY, stage.inputDimZ = self.outputDimX, self.outputDimY, self.outputDimZ
            if stage.supports_taps():
                stage.tapDimY = self.outputDimZ
        if stage.op in [StageType.max_pooling]:
            stage.output = np.zeros(
                (stage.outputDimZ, stage.outputDimY, stage.outputDimX))

    def setoutput(self, outputStride, outputPointer=None, outputIndex=None):
        if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value:
            self.output = np.zeros(
                (int(
                    outputStride / 2), int(
                    self.outputDimY), int(
                    self.outputDimX))).astype(
                enum_as_dtype(
                    self.datatype))
            if outputPointer is not None and outputIndex is not None:
                self.outputPointer = outputPointer
                self.outputIndex = outputIndex
            else:
                self.outputPointer, self.outputIndex = get_zero_buffer(
                    self.output, self.datatype)
            self.outputStrideX = outputStride
        self.isconcat = True
        return self.outputPointer, self.outputIndex

    def concat(stages, lastlayer=True):
        """
        Set the output pointers and fix the output strides to
        concatenate the outputs into the same buffer
        """

        # This check is almost irrelevant since there are other cases when the code
        # fails and this is a hack not propper code.
        for stage_i, stage in enumerate(stages):
            if stage.concat_axis != stages[0].concat_axis:
                raise Exception("A layer cannot be part of multiple concats")

        if(stages[0].concat_axis == 1):
            z = sum([int(stage.unprocessed_k) for stage in stages])
            x = int(stages[0].outputDimX)
            y = int(stages[0].outputDimY)

            def roundup8(x):
                return ((x + 7) // 8) * 8

            #concat_size = (y, x, z)
            concat_size = (y, roundup8(x), z)

            dtype = stages[0].datatype
            if lastlayer:
                for stage in stages:
                    stage.isoutput = True
                buffer = 0
                buffer_index = MemoryIndex.output.value
            elif stages[0].outputPointer == 0 and stages[0].outputIndex == MemoryIndex.none.value:
                buffer, buffer_index = get_zero_buffer(
                    np.zeros(concat_size).astype(
                        enum_as_dtype(dtype)), dtype)
            else:
                buffer = stages[0].outputPointer
                buffer_index = stages[0].outputIndex

            concat_offset = 0

            for s_num, stage in enumerate(stages):
                offset_pointer = buffer

                if(stage.outputPointer == 0):
                    stage.outputPointer = offset_pointer + concat_offset * 2
                stage.outputIndex = buffer_index

                stage.concatResult = True

                pingPongPair = getManualHwSchedule()

                #The following assumtions are made, here, about the decisions taken
                #later in the compilation process:
                #1. In the current concat layer all the input tensors come from
                #   either HW layers only or SW layers only. Mixing outputs of HW and
                #   SW layers is not supported.
                #2. Storage order of concatenated outputs is the same.
                #3. When a network is compiled for hardware the following layer types
                #   are always going to be run on HW: StageType.convolution,
                #   StageType.max_pooling, StageType.average_pooling.
                #4. When a network is compiled for hardware all other layers types
                #   not included in the list at point 3 are software layers that
                #   output in channel minor format i.e. YXZ storage order.

                if GLOBALS.USING_MA2480 and stage.op in [StageType.convolution,
                    StageType.max_pooling, StageType.average_pooling]:
                    # DimX: Width, DimY: Height, DimZ: Depth
                    if pingPongPair[get_null_terminating_name(stage.name)][1][1] == 'I' or GLOBALS.OPT_SCHEDULER:
                        concat_offset += int(stage.outputDimZ * roundup8(stage.outputDimX))
                    else:
                        concat_offset += int(stage.outputDimZ * stage.outputDimY * roundup8(stage.outputDimX))

                else:
                    # Code for concat in channel minor
                    concat_offset += int(stage.outputDimZ)

                # This creates a new attribute to the stage class, which (if detected by CnnHardware.py)
                # appropriately does the concat in interleaved mode
                stage.totalOutChans = z
                stage.concatBufferOffset = stage.outputPointer - offset_pointer
                stage.concatOutputSize = np.prod(concat_size) * 2 # In bytes
                # print("######stage.concatBufferOffset =", stage.outputPointer - offset_pointer)
                # print("######stage.concatOutputSize =", stage.concatOutputSize)

                stage.outputStrideX = z * 2
                stage.outputStrideY = z * 2 * stage.outputDimX
                stage.tapStrideY = stage.outputDimZ * 2
        elif stages[0].concat_axis == 2:

            z = int(stages[0].outputDimZ)
            y = sum([int(stage.outputDimY) for stage in stages])
            x = int(stages[0].outputDimX)

            concat_size = (y, x, z)

            dtype = stages[0].datatype
            if lastlayer:
                for stage in stages:
                    stage.isoutput = True
                buffer = 0
                buffer_index = MemoryIndex.output.value
            elif stages[0].outputPointer == 0 and stages[0].outputIndex == MemoryIndex.none.value:
                buffer, buffer_index = get_zero_buffer(np.zeros(concat_size).astype(enum_as_dtype(dtype)), dtype)
            else:
                buffer = stages[0].outputPointer
                buffer_index = stages[0].outputIndex

            concat_offset = 0

            for s_num, stage in enumerate(stages):
                offset_pointer = buffer

                if(stage.outputPointer == 0):
                    stage.outputPointer = offset_pointer + concat_offset*2  # TODO: REMOVE HARDCODED 2 For FP16 Size

                stage.outputIndex = buffer_index

                stage.concatResult = True
                concat_offset += int(stage.outputDimY * stage.outputDimX * stage.outputDimZ)

        else:
            # This check is almost irrelevant since there are other cases when the code
            # fails and this is a hack not propper code.
            raise Exception("Concat on axis {0} not implemented".format(stages[0].concat_axis))


    def attach_eltwise(self, parents):
        # Attach two parents to this elementwise operations layer
        # The second layer will be put in the weights pointer

        if hasattr(parents[0], '__iter__'):
            NetworkStage.concat(parents[0], False)
            parents[0] = parents[0][0]
        # This line is to build a correct graph after renaming due to
        # absorption, see attach
        self.top[0] = parents[0].unprocessed_name
        # We have only two cases: intermediate, input or intermediate,
        # intermediate
        if parents[1] == 0:
            parents[0].outputPointer, parents[0].outputIndex = get_zero_buffer(
                parents[0].output, self.datatype)
            self.dataPointer = self.inputOffset + parents[0].outputPointer
            self.dataIndex = parents[0].outputIndex
            self.tapsPointer = 0
            self.tapsIndex = MemoryIndex.input.value
            self.tapLayout = self.definition.requirements["input"]["layout"]
            self.recalculate_stride("taps")

            parents[0].tail.append(self)

        else:
            if hasattr(parents[1], '__iter__'):
                NetworkStage.concat(parents[1], False)
                parents[1] = parents[1][0]
            # This line is to build a correct graph after renaming due to
            # absorption, see attach
            self.top[1] = parents[1].unprocessed_name
            if parents[0].outputIndex == 0:
                parents[0].outputPointer, parents[0].outputIndex = get_zero_buffer(
                    parents[0].output, self.datatype)
            if parents[1].outputIndex == 0:
                parents[1].outputPointer, parents[1].outputIndex = get_zero_buffer(
                    parents[1].output, self.datatype)
            self.dataPointer = self.inputOffset + parents[0].outputPointer
            self.dataIndex = parents[0].outputIndex
            self.tapsPointer = parents[1].outputPointer
            self.tapsIndex = parents[1].outputIndex
            self.tapLayout = self.definition.requirements["input"]["layout"]
            self.recalculate_stride("taps")

            parents[1].tail.append(self)
        return

    def attach_multiple_bottoms(self, parents):
        # Attach a layer with at most 3 bottoms.
        # 1st bottom -> to input data pointer.
        # 2nd bottom -> to weights data pointer.
        # 3rd bottom -> to biases data pointer.

        if(len(parents) > 3):
            raise Exception("Layer with {0} inputs not supported".format(len(parents)))

        for bottom_idx, bottom in enumerate(parents):
            if hasattr(bottom, '__iter__'):
                NetworkStage.concat(parents[bottom_idx], False)
                parents[bottom_idx] = parents[bottom_idx][0]

            if bottom == 0:
                # This bottom is the input (ussualy named "data") to the network.
                if bottom_idx == 0:
                    self.dataPointer = 0
                    self.dataIndex   = MemoryIndex.input.value
                elif bottom_idx == 1:
                    self.tapsPointer = 0
                    self.tapsIndex   = MemoryIndex.input.value
                else:
                    self.biasPointer = 0
                    self.biasIndex   = MemoryIndex.input.value
            else:
                # This bottom is the output of a layer.
                if(parents[bottom_idx].outputIndex == 0):
                    out_ptr, out_idx = get_zero_buffer(parents[bottom_idx].output, self.datatype)
                    parents[bottom_idx].outputPointer = out_ptr
                    parents[bottom_idx].outputIndex   = out_idx

                if bottom_idx == 0:
                    self.dataPointer = self.inputOffset + parents[bottom_idx].outputPointer
                    self.dataIndex   = parents[bottom_idx].outputIndex
                elif bottom_idx == 1:
                    self.tapsPointer = parents[bottom_idx].outputPointer
                    self.tapsIndex   = parents[bottom_idx].outputIndex
                else:
                    self.biasPointer = parents[bottom_idx].outputPointer
                    self.biasIndex   = parents[bottom_idx].outputIndex

        #parents[1].tail.append(self)
        #return
        for bottom_idx, bottom in reversed(list(enumerate(parents))):
            if bottom != 0:
                parents[bottom_idx].tail.append(self)
                return

    def attach_several(self, parents):
        """
        Attach a node to several parents. Under 'concat' rules, the parents will be combined at the channel level.
        Under yxz this means that we need a writeback offset.

        TODO: This is coded for YXZ only.
        The default mode should be ZYX and we should transform it during the optimize() phase of network setup.

        :param parents:
        :return:
        """

        # attach_several works with both one and more parents, which must be
        # concat inputs

        if not hasattr(parents, '__iter__'):
            parents.attach(self)
            return

        # The 'tail' list of a node is built such that when there is a child with
        # multiple parentnodes (e.g: concat, etlwise), only the last parent has the
        # child added to the 'tail' list. However, for the automatic scheduler to
        # correctly build the subgraphs, all parent-child relations need to be added to
        # the 'tail' list
        if GLOBALS.USING_MA2480:
            for l in parents:
                l.tail.append(self)

        # Next three lines are to build a correct graph after renaming due to
        # absorption, see attach
        self.top = []
        for l in parents:
            self.top.append(l.unprocessed_name)

        NetworkStage.concat(parents, False)
        z = sum([int(p.unprocessed_k) for p in parents])
        if not GLOBALS.USING_MA2480:
            parents[len(parents) - 1].tail.append(self)
        self.inputDimZ = z
        self.inputStrideX = z * 2

        self.dataPointer = self.inputOffset + \
            parents[0].outputPointer      # Input Pointer
        self.dataIndex = parents[0].outputIndex

        if self.supports_taps():
            self.tapDimY = z

        if self.op in [StageType.max_pooling]:
            self.outputDimZ = self.inputDimZ
            self.outputStrideX = self.inputStrideX

    def search(self, seek_name):
        """
        Initalizes a search. Uses the global 'search_list' to make sure we don't recurse
        down paths we have already searched.
        Clears before each call.
        """
        global search_list
        search_list = []
        return self.search_recurse(seek_name, [])

    def search_recurse(self, seek_name, _searched):
        """
        return: 0 if not found. The searched node if found.
        :param seek_name: name of node we're looking for
        :param _searched: private param for avoiding already searched nodes.
        """
        global search_list

        if self in search_list:
            # We already searched here to no avail. Return
            return 0

        if search_list == []:
            # Only if there is one node, otherwise, the children check will find the result first.
            if self.name == seek_name or self.unprocessed_name == seek_name or seek_name in self.alias:
                return self

        for t in self.tail:
            if t.name == seek_name or t.unprocessed_name == seek_name or seek_name in t.alias:
                search_list += [t]
                return t
            else:
                # Not found, traverse deeper
                recursive_result = t.search_recurse(seek_name, _searched)

                if (recursive_result != 0 and recursive_result.name == seek_name) or \
                    (recursive_result != 0 and recursive_result.unprocessed_name == seek_name) or \
                    (recursive_result != 0 and seek_name in recursive_result.alias):
                    # Found in one of the tree nodes, bubble up.
                    search_list += [t]
                    return recursive_result
                else:
                    # Children were searched to no avail.
                    search_list += [t]

        search_list += [self]
        return 0  # Not found, backtrack

    def supports_taps(self):
        """
        Checks if the layer has taps, and that they are not empty.
        """
        if hasattr(self, "taps") and self.taps is not None:
            return True
        else:
            return False

    def debug(self, to_file=False, f=None):
        """
        Developers can use this function to print values recursively through every layer.
        :param to_file: A field that could be used to write debug info to a file optionally.
        :param f: The corresponding filename if to_file is True
        :return: Nothing
        """

        for t in self.tail:
            t.debug(to_file, f)

    def finalize(self):
        """
        This 'pass' of the compiler computes final editions of the buffers.
        They will be used in the serialization phase.

        Note: Not a clean function, there are potential undocumented effects.
        """
        # Taps (and maybe bias) can be modified by batch normalization layer
        # that follows, so add them at the end
        self.putTaps()
        self.putBias()
        self.putOpParams()

        # This flag must be set so that we do not traverse a layer multiple times,
        # potentially creating multiple versions of buffers & other bad side-effects.
        self.isFinalized = True

        def get_null_terminating_name(x):
            """Input is a bytes class"""
            return (x.split(b'\0', 1)[0]).decode('utf-8')

        # Altering the taps of the convolution after concat
        if get_null_terminating_name(self.name) == 'just_make_concat_not_be_last':
            print(self.taps.shape)
          # Iterate over the first dimension of the 4D taps, ie. iterate over the
          # taps themselves.
            index = 0

            for tap in self.taps:
                tap.fill(0.0)
                tap[index, :, :] = 1.0
                index += 1

        for t in self.tail:
            if not hasattr(t, "isFinalized"):
                t.finalize()

    def check_algo(self, network):
        # Check if two layers write to the same output and one of them is convolution
        # Force im2col_v2 in this case, where it's normally never used

        self.algo_checked = True

        if self.op == StageType.convolution and self.inputDimZ >= 200:
            if network.writes_output(self, self.outputIndex):
                if self.optMask & 0x80000000 == 0x80000000:
                    self.optMask = (self.optMask & 0x7fffffff) | 4
                    print(
                        'Layer ',
                        self.unprocessed_name,
                        ' forced to im2col_v2, because its output is used in concat')
        for t in self.tail:
            if not hasattr(t, "algo_checked"):
                t.check_algo(network)

    def writes_output(self, exclude_layer, index):
        # Return True if this or a tail layer uses index as input
        self.output_written = True
        for t in self.tail:
            if not hasattr(t,"output_written"):
                if t.writes_output(exclude_layer, index):
                    return True
        if self != exclude_layer and self.outputIndex == index:
            return True
        return False

    def assign_remnant_buffers(self, net):
        sizes = []
        offsets = []
        names = []

        self.assign_remnant_done = True

        if self.top is not None and isinstance(self.top[0], str):
            # This should ensure that the input strides are correct after
            # applying all concat rules.
            parent = net.search(self.top[0])

            self.inputStrideX = parent.outputStrideX
        if self.isoutput:
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
        elif self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value and (self.top is None or len(self.top) <= 1 or get_class_of_op(self.op) != "Pooling"):
            self.outputIndex = MemoryIndex.output.value
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
            self.isoutput = True
        elif self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value and len(self.top) > 1 and get_class_of_op(self.op) == "Pooling":
            self.output = np.zeros(self.output.shape).astype(np.float16)
            # This code was breaking when ending with a Pool:
            # node = net.head[0].search(self.top[0])
            # self.output = np.zeros(
            #     (node.outputDimZ,
            #      node.outputDimY,
            #      node.outputDimX)).astype(np.float16)
            self.outputIndex = MemoryIndex.output.value
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
            self.isoutput = True

        for t in self.tail:
            if not hasattr(t,"assign_remnant_done"):
                t_res = t.assign_remnant_buffers(net)
                sizes.extend(t_res[0])
                offsets.extend(t_res[1])
                names.extend(t_res[2])
        return sizes, offsets, names

    def convert_inputs_outputs_to_yxz(self, recurse, debug=False):
        """
        It is necessary to convert the first input because it contains data, but the other inputs
        and outputs are buffers for myriads use only. But, we will need to have the final outputs shaped correctly, so
        we will transform them too.
        :param recurse: Set to false to apply to a single element, true to traverse.
        :param debug: Set to true to enable debug print messages
        :return: Nothing. Side effect of transformed buffers.
        """

        self.io_was_converted = True

        if self.storageOrder == StorageOrder.orderYXZ:
            if debug:
                print("Already in this form")
        elif self.storageOrder == StorageOrder.orderZYX:
            if len(self.data.shape) == 4:
                self.data = np.reshape(
                    self.data, (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
                self.data = zyx_to_yxz(
                    self.data, self.datatype).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                if not self.concatResult:
                    self.output = zyx_to_yxz(
                        self.output, self.datatype).astype(
                        dtype=np.float16)
                    self.storageOrder = StorageOrder.orderYXZ


        elif self.storageOrder == StorageOrder.orderXYZ:
            if len(self.data.shape) == 4:
                self.data = np.reshape(
                    self.data, (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
                self.data = xyz_to_yxz(
                    self.data, self.datatype).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                throw_error(
                    ErrorTable.ConversionNotSupported,
                    self.storageOrder.name)

        else:
            throw_error(
                ErrorTable.ConversionNotSupported,
                self.storageOrder.name)

        if recurse:
            for node in self.tail:
                if not hasattr(node, "io_was_converted"):
                    node.convert_inputs_outputs_to_yxz(recurse)

    def convert_taps_to_hwck(self, recurse):
        self.taps_were_converted = True

        if self.tapsOrder != TapsOrder.orderHWCK:
            if get_class_of_op(self.op) in [
                    "Convolution",
                    "FCL",
                    "Deconvolution"]:
                if self.op in [StageType.fully_connected_layer]:
                    if self.unprocessed_h > 1 or self.unprocessed_w > 1:

                        self.taps = self.taps.reshape(
                            self.unprocessed_k,
                            self.unprocessed_c,
                            self.unprocessed_h,
                            self.unprocessed_w)
                    else:
                        self.taps = self.taps.reshape(
                            self.taps.shape[0], self.taps.shape[1], 1, 1)

                taps_shape = self.taps.shape
                self.taps = kchw_to_hwck(self.taps)
                self.tapsOrder = TapsOrder.orderHWCK
                replace_buffer(self.taps, self.tapsBufferIndex, self.datatype,
                    hwAlignment=True, originalShape=taps_shape, op=self.op, tapsOrder=self.tapsOrder)
            else:
                if (self.taps is None or
                        get_class_of_op(self.op) == "FCL" or
                        self.op == StageType.scale or
                        self.op == StageType.normalize):
                    pass
                else:
                    throw_error(
                        ErrorTable.ConversionNotSupported,
                        self.op.name)

            self.storageOrder = StorageOrder.orderYXZ.value


        if recurse:
            for node in self.tail:
                if not hasattr(node, "taps_were_converted"):
                    node.convert_taps_to_hwck(recurse)


    def getBWs(self):
        in_dim = self.data.flatten().shape[0]
        if self.taps is not None:
            tap_dim = self.taps.flatten().shape[0]
        else:
            tap_dim = 0
        out_dim = self.output.shape[0]

        KB = 1024
        MB = KB * KB

        MS = self.ms
        S = MS / 1000

        if self.op == StageType.convolution:
            arrays = in_dim * self.radixX * self.radixY  # Read Data NxN Times
            arrays += tap_dim                        # Taps once (already NxN)
            arrays += out_dim * self.radixX * self.radixY  # Accumulate NxN
        else:
            arrays = in_dim + tap_dim + out_dim
        self.BWs = ((arrays * 2) / MB) / S
        return self.BWs

    def getBW(self):
        in_dim = self.data.flatten().shape[0]
        if self.taps is not None:
            tap_dim = self.taps.flatten().shape[0]
        else:
            tap_dim = 0
        out_dim = self.output.shape[0]

        if self.op == StageType.convolution:
            arrays = in_dim * self.radixX * self.radixY  # Read Data NxN Times
            arrays += tap_dim                        # Taps once (already NxN)
            arrays += out_dim * self.radixX * self.radixY  # Accumulate NxN
        else:
            arrays = in_dim + tap_dim + out_dim

        return (arrays * 2)

    def minmax(self, attr, min, max):
        self.gotMinMax = True
        if min > getattr(self, attr):
            min = getattr(self, attr)
        if max < getattr(self, attr):
            max = getattr(self, attr)

        for t in self.tail:
            if not hasattr(t, "gotMinMax"):
                min, max = t.minmax(attr, min, max)

        return min, max

    def calculate_metrics(self, timings):
        self.flops = self.getFlops()
        self.ms = timings[0]
        self.BWs = self.getBWs()

    def getFlops(self):
        """

        :return:

        """
        flops = 0
        if self.op in [StageType.convolution, StageType.myriadX_convolution]:
            # Output channels too.
            flops = self.unprocessed_k * self.outputDimX * self.outputDimY * \
                self.inputDimZ * self.radixX * self.radixY * 2

        elif self.op in [StageType.max_pooling, StageType.myriadX_pooling]:
            flops = self.unprocessed_k * self.outputDimX * \
                self.outputDimY * self.radixX * self.radixY

        elif self.op in [StageType.average_pooling, StageType.myriadX_pooling]:
            flops = self.unprocessed_k * self.outputDimX * \
                self.outputDimY * self.radixX * self.radixY * 2

        elif self.op in [StageType.fully_connected_layer, StageType.myriadX_fully_connected_layer]:
            in_dim = self.data.flatten().shape[0]
            out_channels = self.output.shape[0]
            flops = in_dim * out_channels * 2

        elif self.op == StageType.depthwise_convolution:
            flops = self.radixX * self.radixY * self.unprocessed_k * \
                self.outputDimX * self.outputDimY * 2

        elif self.op == StageType.soft_max:
            in_dim = self.data.flatten().shape[0]
            flops = in_dim * 3

        return flops / 1000000

    def adjust_accuracy(self, scheduler):
        if self.op in [StageType.fully_connected_layer, StageType.convolution, StageType.depthwise_convolution]:
            # To adjust for accuracy multiply the weights by scale, and
            # do post-accumulate multiply by 1/scale.
            if self.accuracy_scale != 1.0:
                # print("adjusting weights for: ", get_null_terminating_name(self.name))
                adjustment_factor = 1.0 / self.accuracy_scale
                scale_vector = np.full(self.outputDimZ, adjustment_factor).astype(np.float16)
                self.putScale(scale_vector)
                if self.taps is not None:
                    self.taps = self.taps * self.accuracy_scale
                if self.bias is not None:
                    self.bias = self.bias * self.accuracy_scale

    def convert_for_hardware(self, scheduler):
        print("Hardwareize One")
        if self.op in [StageType.fully_connected_layer, StageType.convolution, StageType.max_pooling, StageType.average_pooling]:

            self.hardware_solution = hardwareize(self, scheduler)

            input_stride, output_stride, newDimZ, BufSize, hwDescList, taps, bias, scale = self.hardware_solution

            # Write Adjusted Weights
            self.taps = taps
            if self.taps is not None:
                self.taps = replace_buffer(self.taps, self.tapsBufferIndex,  self.datatype,
                    hwAlignment=False, op=self.op, tapsOrder=self.tapsOrder)

            # Hardware may rearrange the taps, if split over the input is activated
            if bias is not None:
                self.bias = replace_buffer(bias, self.biasBufferIndex,  self.datatype)
            if scale is not None:
                self.scale = replace_buffer(scale, self.scaleBufferIndex, self.datatype)
        # Only for a sequential layout

    def recalculate_stride(self, where, myriadX=False, concat=[0,0,0]):
        """
        Recalculates Strides for a buffer.
        @where - A string indicating which NetworkStage buffer should have it's strides re-calculated.
        @myriadX - set to True if the strides should consider padding for myriadX hardware requirements.
        @concat - A 3-element array (X, Y, Z) containing alternate values for dimensions when concatenated.
        """

        if where == "output":
            order = self.outputLayout

            dimX = self.outputDimX
            dimY = self.outputDimY
            dimZ = self.outputDimZ

            if concat[0] != 0:
                print(concat[0], "!=", 0)
                dimX = concat[0]
            if concat[1] != 0:
                dimY = concat[1]
            if concat[2] != 0:
                dimZ = concat[2]

            if myriadX:
                dimX = int (((dimX + 7)//8)*8)


            if order == StorageOrder.orderYZX:
                self.outputStrideX = 2
                self.outputStrideZ = dimX * self.outputStrideX
                self.outputStrideY = dimZ * self.outputStrideZ
            elif order == StorageOrder.orderYXZ:
                self.outputStrideZ = 2
                self.outputStrideX = dimZ * self.outputStrideZ
                self.outputStrideY = dimX * self.outputStrideX
            elif order == StorageOrder.orderZYX:
                self.outputStrideX = 2
                self.outputStrideY = dimX * self.outputStrideX
                self.outputStrideZ = dimY * self.outputStrideY
            else:
                print(order)
                assert 0, "function not implemented for layout"
        elif where == "input":
            order = self.definition.requirements["input"]["layout"]

            dimX = self.inputDimX
            dimY = self.inputDimY
            dimZ = self.inputDimZ

            if concat[0] != 0:
                print(concat[0], "!=", 0)
                dimX = concat[0]
            if concat[1] != 0:
                dimY = concat[1]
            if concat[2] != 0:
                dimZ = concat[2]

            if myriadX:
                dimX = int (((dimX + 7)//8)*8)
                dimZ = dimZ if dimZ > 3 else 4


            if order == StorageOrder.orderYZX:
                self.inputStrideX = 2
                self.inputStrideZ = dimX * self.inputStrideX
                self.inputStrideY = dimZ * self.inputStrideZ
            elif order == StorageOrder.orderYXZ:
                self.inputStrideZ = 2
                self.inputStrideX = dimZ * self.inputStrideZ
                self.inputStrideY = dimX * self.inputStrideX
            elif order == StorageOrder.orderZYX:
                self.inputStrideX = 2
                self.inputStrideY = dimX * self.inputStrideX
                self.inputStrideZ = dimY * self.inputStrideY
            else:
                print(order)
                assert 0, "function not implemented for layout"
        elif where in ["taps","weights"]:
            order = self.tapLayout

            dimX = self.tapDimX
            dimY = self.tapDimY
            dimZ = self.tapDimZ

            if concat[0] != 0:
                print(concat[0], "!=", 0)
                dimX = concat[0]
            if concat[1] != 0:
                dimY = concat[1]
            if concat[2] != 0:
                dimZ = concat[2]

            if myriadX:
                dimX = int (((dimX + 7)//8)*8)


            if order == StorageOrder.orderYZX:
                self.tapStrideX = 2
                self.tapStrideZ = dimX * self.tapStrideX
                self.tapStrideY = dimZ * self.tapStrideZ
            elif order == StorageOrder.orderYXZ:
                self.outputStrideZ = 2
                self.tapStrideX = dimZ * self.tapStrideZ
                self.tapStrideY = dimX * self.tapStrideX
            elif order == StorageOrder.orderZYX:
                self.tapStrideX = 2
                self.tapStrideY = dimX * self.tapStrideX
                self.tapStrideZ = dimY * self.tapStrideY
            else:
                print(order)
                assert 0, "function not implemented for layout"
        else:
            assert 0, "Function not supported for this buffer type"

    def add_conversion_layers(self):
        """
        Handles conversion between layers of different data layouts.

        If a common format is applicable, it will convert the dimensions to fit.
        Otherwise, a conversion layer is placed between them.
        """

        hardware_layers = [StageType.myriadX_convolution, StageType.myriadX_fully_connected_layer, StageType.myriadX_pooling]
        layers_that_support_hw = [StageType.LRN,
            StageType.depthwise_convolution,
            # StageType.soft_max,
            StageType.normalize
            ]
        # layers_that_support_hw = []   # FOR TESTING

        if self.tail and self.op != StageType.none:
            for child_idx in range(len(self.tail)):
                child = self.tail[child_idx]
                """
                Ensure outputLayout is instanciated.
                Note: This is an unsustainable appproach. The output layout will not always be the same as input.
                """
                if not hasattr(self, 'outputLayout'):
                    self.outputLayout = self.definition.requirements["input"]["layout"]

                my_req = self.outputLayout
                child_req = child.definition.requirements

                """
                Some software layers can work with data in hardware layouts,
                If a hardware layer is followed by such a software layer, the software layer is converted appropiately.
                Note: this does not work with FCL as a prior hardware stage currently as it has different restrictions.
                """
                if (my_req != child_req["input"]["layout"]
                        and my_req in [StorageOrder.orderZYX, StorageOrder.orderYXZ, StorageOrder.orderYZX]
                        and child.op in layers_that_support_hw):
                    child.definition.changeLayoutRequirements("input", self.outputLayout)

                    # See comment above, Assumes follow-through of layout and dims/strides.
                    child.outputLayout = self.outputLayout # self.definition.requirements["input"]["layout"]

                    if len(child.top) > 1:
                        # Ensure concat is accounted for
                        concat_Z = self.concatOutputSize//(self.outputDimX*self.outputDimY*2)
                        child.inputDimZ = concat_Z
                        child.outputDimZ = concat_Z

                    child.recalculate_stride("input", myriadX=True)
                    child.recalculate_stride("output", myriadX=True)

                    self.recalculate_stride("output", myriadX=True)

                """
                The same for if we have a software layer followed
                """
                if (my_req != child_req["input"]["layout"]
                        and child_req["input"]["layout"] in [StorageOrder.orderZYX, StorageOrder.orderYXZ, StorageOrder.orderYZX]
                        and self.op in layers_that_support_hw):
                    self.definition.changeLayoutRequirements("input", child.definition.requirements["input"]["layout"])
                    self.recalculate_stride("output", myriadX=True)

                    # Assumes follow-through of layout and dims/strides.
                    child.outputLayout = self.definition.requirements["input"]["layout"]
                    child.recalculate_stride("output", myriadX=True)


                output_layout = self.outputLayout


                """
                If there is a mis-match of layouts and the child layer is not hardware friendly
                """
                if child_req["input"]["layout"] != output_layout and \
                    child.op not in layers_that_support_hw and \
                    child.op is not StageType.storage_order_convert and \
                    self.op is not StageType.storage_order_convert:

                    if child.op in hardware_layers and self.op in hardware_layers:
                        assert 0, "Something has gone wrong. It is likely a mismatch of Hw Configuration and network description"

                    if self.op not in hardware_layers and child.op not in hardware_layers:
                        """
                        The case where both layers are software.
                        """
                        self.outputLayout = child.definition.requirements["input"]["layout"]
                        self.recalculate_stride("output", myriadX=False)

                    else:
                        """
                        The case where the one layer is hardware and the other is software.
                        """
                        # Create a storage order conversion layer and insert it into the graph. # TODO: Untested on Concats and fancier operations.
                        new_node = NetworkStage(
                                "Convert_"+self.unprocessed_name+"_"+child.unprocessed_name,
                                None,   # Top
                                None,   # Storage Order
                                0,   # Pad Y
                                0,   # Pad X
                                None,   # Pad Type
                                None,   # Dtype
                                None,   # Precision
                                StageType.storage_order_convert, # OpType
                                0,      # Op Y
                                0,      # Op X
                                0,      # Stride Y
                                0,      # Stride X
                                0,      # X
                                0,      # Y
                                0,      # C
                                0,      # FH
                                0,      # FW
                                0,      # K
                                None,   # Weights
                                None, # Weights Storage Order
                                None,   # Bias
                                None,    # Pre-Op
                                None,   # Post-Op
                                None,         # Post-Op Param
                                None,        # Post-Op Stride X
                                None,        # Post-Op Stride Y
                                network=self.network
                            )

                        # As we are post-finalizing, the out buffer of A and in buffer of B in the network A=B are the same,
                        # We insert a 'C' node to differenciate. A=C=B

                        if len(child.top) > 1:
                            # Concat only needs one new buffer for N stages.

                            concat_Z = self.concatOutputSize//(self.outputDimX*self.outputDimY*2)
                            if self.concatBufferOffset == 0:
                                # If it's the first stage of the concat, get a new one.
                                new_buffer_location, new_buffer_index = get_zero_buffer(np.zeros((concat_Z,self.outputDimY,self.outputDimX)), DataType.fp16)
                                # Create a reference for the others to pick up.
                                for parents in child.network.search_several(child.top):
                                    parents.convertBuffer_concat = (new_buffer_location, new_buffer_index)
                            else:
                                # Get the old one from the stage that previously created one.
                                new_buffer_location, new_buffer_index = self.convertBuffer_concat
                        else:
                            # Create a new buffer between A and B
                            new_buffer_location, new_buffer_index = get_zero_buffer(np.zeros((self.outputDimZ,self.outputDimY,self.outputDimX)), DataType.fp16)

                        # print("Convert after", self.unprocessed_name)
                        # print("Output of said layer:", self.outputDimX, self.outputDimY, self.outputDimZ)

                        if len(child.top) > 1:
                            concat_Z = self.concatOutputSize//(self.outputDimX*self.outputDimY*2)
                            concat_mods = [0, 0, concat_Z]
                        else:
                            concat_mods = [0, 0, 0]

                        # Input
                        new_node.inputDimX = self.outputDimX
                        new_node.inputDimY = self.outputDimY
                        new_node.inputDimZ = self.outputDimZ
                        new_node.dataPointer = self.outputPointer
                        new_node.dataIndex = self.outputIndex
                        new_node.definition.requirements["input"]["layout"] = output_layout

                        new_node.recalculate_stride("input", myriadX=True, concat=concat_mods)

                        # Input
                        new_node.outputDimX = child.inputDimX
                        new_node.outputDimY = child.inputDimY
                        new_node.outputDimZ = child.inputDimZ
                        new_node.outputPointer = new_buffer_location # child.dataPointer
                        new_node.outputIndex = new_buffer_index # child.dataIndex
                        new_node.outputLayout = child.definition.requirements["input"]["layout"]

                        new_node.recalculate_stride("output", concat=concat_mods)

                        child.dataPointer = new_buffer_location
                        child.dataIndex = new_buffer_index

                        # net.insert_node_between(this, and_this, withObject)
                        self.network.insert_node(
                            new_node,
                            self,
                            child
                        )
                elif child_req["input"]["layout"] != output_layout and \
                    self.op in layers_that_support_hw and \
                    child.op in layers_that_support_hw and \
                    child.op is not StageType.storage_order_convert and \
                    self.op is not StageType.storage_order_convert:
                    """
                    The case where we have a hw-friendly SWlayer followed by another
                    hw-friendly SWLayer that is not converted yet.
                    """
                    child.definition.requirements["input"]["layout"] = self.definition.requirements["input"]["layout"]
                    child.outputLayout = self.outputLayout
                    child.recalculate_stride("input", myriadX=True)
                    child.recalculate_stride("output", myriadX=True)

        else:
            # Nothing to be done/
            pass


    def bundle(self):

        self.bundled = True

        self.dataBUF = Buffer(self.inputDimX,
                              self.inputDimY,
                              self.inputDimZ,
                              self.inputStrideX,
                              self.inputStrideY,
                              self.inputStrideZ,
                              self.dataPointer,
                              self.dataIndex,
                              order=self.definition.requirements["input"]["layout"])

        if not hasattr(self, 'tapLayout'):
            self.tapLayout = StorageOrder.orderXYZ
            self.tapStrideZ = 2
            self.tapStrideY = self.tapStrideZ * self.tapDimZ
            self.tapStrideX = self.tapStrideY * self.tapDimY

        if not hasattr(self, 'outputLayout'):
            # Unsustainable. Output will not always be same as input.
            self.outputLayout = self.definition.requirements["input"]["layout"]

        self.tapsBUF = Buffer(self.tapDimX,
                             self.tapDimY,
                             self.tapDimZ,
                             self.tapStrideX,
                             self.tapStrideY,
                             self.tapStrideZ,
                             self.tapsPointer,
                             self.tapsIndex,
                             order=self.tapLayout
                            )

        self.outputBUF = Buffer(self.outputDimX,
                                self.outputDimY,
                                self.outputDimZ,
                                self.outputStrideX,
                                self.outputStrideY,
                                self.outputStrideZ,
                                self.outputPointer,
                                self.outputIndex,
                                order=self.outputLayout
                                )
        self.biasBUF = Buffer(None, None, None,
                              None, None, None,
                              self.biasPointer, self.biasIndex,
                              order=StorageOrder.orderZYX)
        self.scaleBUF = Buffer(None, None, None,
                              None, None, None,
                              self.scalePointer, self.scaleIndex)

        self.opParamsBUF = Buffer(None, None, None,
                              None, None, None,
                              self.opParamsPointer, self.opParamsIndex)

        for t in self.tail:
            if not hasattr(t,"bundled"):
                t.bundle()

    def summaryStats(self):
        totalTime = self.ms
        totalBW = self.getBW()

        self.summarized = True

        for t in self.tail:
            if not hasattr(t, "summarized"):
                a, b = t.summaryStats()
                totalTime += a
                totalBW += b

        return totalTime, totalBW

    def graphviz(
            self,
            dot,
            ms_min,
            ms_max,
            bws_min,
            bws_max,
            flop_min,
            flop_max):
        """
        Create the Graph Node for this layer.
        :param dot: graphviz graph object
        :param ms_min: Minimum time for a layer in the graph (used for color scaling)
        :param ms_max: Maximum time for a layer in the graph (used for color scaling)
        :param bws_min: Minimum Bandwidth for a layer in the graph (used for color scaling)
        :param bws_max: Maximum Bandwidth for a layer in the graph (used for color scaling)
        :param flop_min: Minimum FLOPs for a layer in the graph (used for color scaling)
        :param flop_max: Maximum FLOPs for a layer in the graph (used for color scaling)
        :return: the graph object with the graphical node for this layer attached, NetworkStages for recursion
        """


        table = '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
<TR>
    <TD  BGCOLOR = "#A3A3A3" COLSPAN="3">{0}</TD>
</TR>
<TR>
    <TD  BGCOLOR = "#DDDDDD" COLSPAN="3">{7}</TD>
</TR>
<TR>
    <TD BGCOLOR = "{1}"> {2} <br/> (MFLOPs) </TD>
    <TD BGCOLOR = "{3}"> {4} <br/> (MB/s) </TD>
    <TD BGCOLOR = "{5}"> {6} <br/> (ms)</TD>
</TR>
</TABLE>>
'''.format(
            self.unprocessed_name, get_normalized_color(
                "#B1F1EF", "#2ED1C6", flop_min, flop_max, self.flops), self.flops, get_normalized_color(
                "#FFE5FC", "#B2189E", bws_min, bws_max, format(
                    self.BWs, ".2f")), format(
                    self.BWs, ".2f"), get_normalized_color(
                        "#FFFFCC", "#FFFF00", ms_min, ms_max, format(
                            self.ms, ".2f")), format(
                                self.ms, ".2f"), str(
                                    self.unprocessed_output.shape))

        dot.node(self.unprocessed_name, table, shape="plaintext")
        if self.top is not None:
            for t in self.top:
                if not isinstance(t, str):
                    for tt in t:
                        dot.edge(tt, self.unprocessed_name)
                else:
                    dot.edge(t, self.unprocessed_name)

        else:
            dot.edge("Input", self.unprocessed_name)

        last_nodes = []
        self.drawn = True
        for t in self.tail:
            if not hasattr(t, "drawn"):
                dot, last = t.graphviz(
                    dot, ms_min, ms_max, bws_min, bws_max, flop_min, flop_max)
                last_nodes.extend(last)
        if len(self.tail) == 0:
            last_nodes = [self.unprocessed_name]

        return dot, last_nodes
