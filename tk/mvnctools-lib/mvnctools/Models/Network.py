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

from mvnctools.Controllers.MiscIO import *
from mvnctools.Controllers.NCE import NCE_Scheduler
import numpy as np
from mvnctools.Models.NetworkStage import *
import ctypes

import mvnctools.Controllers.Globals as GLOBALS
from mvnctools.Controllers.PingPong import getManualHwSchedule, get_null_terminating_name


class Network:
    def __init__(self, name, data, isMA2480=False, rearrange = False):
        self.name = name
        self.head = []
        self.count = 0
        self.stageslist = []

        # self.data = data
        # self.dataPointer = 0
        self.NCE = NCE_Scheduler(enable_permutation = rearrange)

        self.outputInfo = None
        self.outputTensorShape  = None
        self.outputTensorLayout = None

        self.outputIsSsdDetOut = False

        self.inputTensor = data
        self.datatype = DataType.fp16
        self.isMA2480 = isMA2480
        self.scheduler = GLOBALS.OPT_SCHEDULER

    def attach(self, stage, debug=False):
        """
        Attaches to the top of the network tree, or if already filled, finds
        the appropriate node to attach to.
        Restriction: Names MUST be unique and parents must already be attached.
        :param stage: Stage to attach
        :param debug: enable some debug messages
        :return: 1 if successful, 0 if not
        """

        self.stageslist.append(stage)                               # Attach to overall list of stages.
        if len(self.head) == 0:                                     # If this is the first ever node.
            self.head.append(stage)                                 #   Attach it to the head of the graph
            stage.data = self.inputTensor                           #   Label it as an Input
            stage.dataIndex = MemoryIndex.input.value               #   ""
            self.storageOrder = stage.storageOrder                  #   ""
            self.count = 1                                          #   ""
            if debug:
                print("attached.")
            return 1        # Attached to head
        else:                                                       # If this a subsequent layer
            if stage.top is None:                                   # Check if it is an input to the graph
                stage.data = self.inputTensor
                stage.dataIndex = MemoryIndex.input.value
                self.head.append(stage)
                self.count += 1
            elif len(stage.top) > 1:                                # Check if it has more than one input
                appropriate_nodes = self.search_several(stage.top)
                stage.attach_multiple_bottoms(appropriate_nodes)
                if debug:
                    print("attached.")
                self.count += 1
            else:                                                   # Otherwise, it has only one input
                parent = stage.top[0]
                appropriate_nodes = self.search_several(parent)
                if appropriate_nodes == 0:
                    throw_error(ErrorTable.GraphConstructionFailure, parent)
                else:
                    stage.attach_several(appropriate_nodes)
                    self.count += 1
                    if debug:
                        print("attached.")

            return 1    # Attached to appropiate node in tree.

    def insert_node(self, new_node, top_node, bottom_node):

        top = self.search(top_node.unprocessed_name)
        bottom = self.search(bottom_node.unprocessed_name)

        assert top != -1
        assert bottom != -1

        # Add to graph
        top.tail.remove(bottom)
        top.tail.append(new_node)
        new_node.tail.append(bottom)

        # Add to list?
        top_idx = self.stageslist.index(top)
        bottom_idx = self.stageslist.index(bottom)
        self.stageslist.insert(bottom_idx, new_node)

    def search(self, seek_name):
        """
        Forwarder for tree search of the network tree.
        return: 0 if not found. The searched node if found.
        :param seek_name: name of node, without padded characters
        """
        if seek_name == 0:
            throw_error(ErrorTable.GraphConstructionFailure, seek_name)

        for stage in self.head:
            ret = stage.search(seek_name)
            if ret != 0:
                return ret
        return 0

    def search_several(self, seek_names):
        # This will also work with one seek_names as string
        """

        :param seek_names:
        :return:
        """
        if isinstance(seek_names, str):
            return self.search(seek_names)
        nodes = []
        for name in seek_names:
            # name can be a name or a sequence of names, if it's a concat
            if isinstance(name, str):
                nodes.append(self.search(name))
            else:
                nodes.append(self.search_several(name))

        return nodes

    def generate_info(self, f):
        """
        Writes the information section of the blob file.
        :param f:
        :return:
        """
        sz = 0
        # The stages have to be processed in the order they have been
        # created, not in a tree-based order, otherwise we risk not
        # respecting dependencies
        for stage in self.stageslist:
            sz += stage.generate(f)

        for nul in range(align(sz, np.zeros((1)), align_to=8)[0] - sz):
            # Fill in some padding to align the start of the weights
            f.write(c_char(0))

    def generate_data(self, f):
        """
        Writes the data section of the blob file.
        :param f:
        :return:
        """
        write_data(f)

    def debug(self):
        """
        Print layer information from each node in the network
        :return:
        """
        for stage in self.head:
            stage.debug()

    def bundle(self):
        """

        :return:
        """
        for stage in self.head:
            stage.bundle()

    def set_output_tensor(self):
        """
        Determine the new outputTensorShape based on the layout set 
        after scheduling/hardware-ization
        """
        print('Network.set_output_tensor')
        prevShape       = self.outputTensorShape
        prevLayout      = self.outputTensorLayout
        output_stage    = self.get_output_stage()
        newLayout       = output_stage.outputLayout

        newShape = storage_order_reshape(prevShape, prevLayout, newLayout)
        self.outputTensorShape  = newShape
        self.outputTensorLayout = newLayout

    def finalize(self):
        """
        Run any actions that need to happen after network is constructed.
        :return:
        """
        sizes = []
        pointers = []
        names = []
        # Go through all output pointers and make sure they are assigned.
        for stage in self.head:
            t_res = stage.assign_remnant_buffers(self)
            sizes.extend(t_res[0])
            pointers.extend(t_res[1])
            names.extend(t_res[2])
        # self.debug()
        self.outputInfo = (sizes, pointers, names)
        self.check_algo()
        for stage in self.head:
            stage.finalize()
            # stage.set_blob_vars()

    def check_algo(self):
        """
        Force im2col_v2 when convolutions are concatenated, because otherwise
        other versions could be used, which write outside their buffer
        """
        for stage in self.head:
            stage.check_algo(self)

    def writes_output(self, exclude_layer, index):
        """
        Return true if there is at least one layer which writes to an output
        with index; used by check_algo
        """
        for stage in self.head:
            if stage.writes_output(exclude_layer, index):
                return True
        return False

    def optimize(self):
        """
        Convert into our ideal representation for myriad
        :return: Nothing
        """
        # scheduling operations
        if self.isMA2480 and self.scheduler:
            self.NCE.scheduling_operations(self)

        self.convert_network_input_to_yxz()
        for stage in self.head:
            stage.convert_inputs_outputs_to_yxz(True)
            stage.convert_taps_to_hwck(True)    # recursively
        for idx, out_node in enumerate(self.outputInfo[0]):
            self.outputInfo[0][idx] = (out_node[2], out_node[1], out_node[0])

    def convert_for_hardware(self):
        print("Hardwareize Net")

        def round_up(x, multiple):
            """ Round up the integer value `x` to the closest multiple `multiple`"""
            return ((x + multiple - 1) // multiple) * multiple

        if self.head[0].op in [StageType.myriadX_fully_connected_layer]:
            data = self.inputTensor.flatten()
            arr = np.zeros(data.shape[0] * 8)
            arr[::8] = data
        elif self.head[0].op in [StageType.myriadX_convolution, StageType.myriadX_pooling]:
            line_size = self.inputTensor.shape[-1]
            pad_needed = round_up(line_size, 8) - line_size
            # Input needs to be padded to x16bytes for first line. for hardware
            arr = np.pad(self.inputTensor, ((0, 0), (0, 0), (0, 0), (0, pad_needed)), "constant")
        else:
            arr = self.inputTensor

        self.inputTensor = arr

        # Adjust network accuracy
        for stage in self.stageslist:
            stage.adjust_accuracy(stage)

        # After optimization, convert stages to hardware, if supported by the stage itself
        for stage in self.stageslist:
            stage.convert_for_hardware(self.NCE)

        # Add conversion layers between HW & SW
        for stage in self.stageslist:
            stage.add_conversion_layers()

        # set the appropriate outputTensorShape and outputTensorLayout 
        # following hardware-ization
        self.set_output_tensor()


    def get_output_stage(self):
        return self.stageslist[-1]

    def gather_metrics(self, timings):
        prev_len = 0
        # The stages have to be processed in the order they have been
        # created, not in a tree-based order, otherwise we risk not
        # respecting dependencies
        for stage in self.stageslist:
            stage.calculate_metrics(timings[prev_len:])
            prev_len = prev_len + 1

    def convert_network_input_to_yxz(self, debug=False):
        """
        It is necessary to convert the first input because it contains data, but the other inputs
        and outputs are buffers for myriads use only. But, we will need to have the final outputs shaped correctly, so
        we will transform them too.
        :param recurse: Set to false to apply to a single element, true to traverse.
        :param debug: Set to true to enable debug print messages
        :return: Nothing. Side effect of transformed buffers.
        """

        def convert_to_interleaved(inputTensor, name):
            stageName = get_null_terminating_name(name)
            pingPongPair = getManualHwSchedule()
            _, pingPongFmt, _ = pingPongPair[stageName]

            # Interleaved input
            if pingPongFmt[0] == 'I':
                s = inputTensor.shape
                # We reshape to the original dimensions to avoid errors
                # in different parts of the code.
                if len(s) == 4:
                    GLOBALS.INPUT_IN_INTERLEAVED = True
                    return inputTensor.transpose(0, 2, 1, 3).reshape(s)

                if len(s) == 3:
                    GLOBALS.INPUT_IN_INTERLEAVED = True
                    return inputTensor.transpose(1, 0, 2).reshape(s)

            return inputTensor

        if self.stageslist[0].op in [StageType.fully_connected_layer, StageType.convolution, StageType.max_pooling,
                       StageType.average_pooling] and self.isMA2480:

            self.inputTensor = convert_to_interleaved(self.inputTensor, self.stageslist[0].name)

            print("Returning Immediately for Hardware")
            return

        # The first stage is the input data (source) layer.
        if len(self.stageslist) >= 2:   # Ensure that there is a second stage to process.
            if self.stageslist[0].op == StageType.none:
                if self.stageslist[1].op in [StageType.fully_connected_layer, StageType.convolution,
                                             StageType.max_pooling, StageType.average_pooling] and self.isMA2480:
                    old = self.inputTensor
                    self.inputTensor = convert_to_interleaved(self.inputTensor, self.stageslist[1].name)

                    print("Returning Immediately for Hardware")
                    return

        if self.storageOrder.value == StorageOrder.orderYXZ.value:
            if debug:
                print("Already in this form")
        elif self.storageOrder.value == StorageOrder.orderZYX.value:
            if len(self.inputTensor.shape) == 4:
                self.inputTensor = np.reshape(
                    self.inputTensor,
                    (self.inputTensor.shape[1],
                     self.inputTensor.shape[2],
                     self.inputTensor.shape[3]))
                self.inputTensor = zyx_to_yxz(
                    self.inputTensor,
                    self.datatype.value).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                self.inputTensor = zyx_to_yxz(
                    self.inputTensor,
                    self.datatype.value).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
        elif self.storageOrder.value == StorageOrder.orderXYZ.value:
            if len(self.inputTensor.shape) == 4:
                self.inputTensor = np.reshape(
                    self.inputTensor,
                    (self.inputTensor.shape[1],
                     self.inputTensor.shape[2],
                     self.inputTensor.shape[3]))
                self.inputTensor = xyz_to_yxz(
                    self.inputTensor,
                    self.datatype.value).astype(
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

    def verify(self):
        """
        Calls verify() on the top of the network to be recursed down
        :return:
        """
        for stage in self.head:
            stage.verify()

    def newick(self):
        """
        Outer wrapper for newick format.
        :return:
        """
        # To review
        nw = "( "
        for idx, t in enumerate(self.head):
            nw += t.newick(head=True)
            if idx + 1 != len(self.head):
                nw += ","
        nw += " );"
        return nw
