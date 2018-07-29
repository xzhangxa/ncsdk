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


import os
import subprocess
import tempfile

os.environ["GLOG_minloglevel"] = "3"
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors

import caffe

import numpy as np
from collections import OrderedDict

from google.protobuf import text_format
from caffe.proto import caffe_pb2

from mvnctools.Controllers.Preprocess import preprocess
from mvnctools.Models.EnumDeclarations import SeedData

from mvnctools.Controllers.Parsers import CaffeParser as cp
from mvnctools.Controllers.Tensor import UnpopulatedTensor

from mvnctools.Controllers.Parsers.Parser.Layer import OriginalName, MangledName
from mvnctools.Controllers.Parsers.Parser.Output import Output
from mvnctools.Controllers.Parsers.Parser.DetectionOutput import DetectionOutput

from mvnctools.Views.IRVisualize import drawGraph, drawGraphFromLayers


RAND_HI = 1
RAND_LO = -1

def protoConverterPath():
    path, _ = os.path.split(os.path.abspath(caffe.__file__))
    path, _ = os.path.split(path)
    path, _ = os.path.split(path)

    converter_path = ['build', 'tools', 'upgrade_net_proto_text']
    return os.path.join(path, *converter_path)


def regularizeInPlaceOps(parsedLayers):
    # Some operations in Caffe can be inplace. Introduce new names for these layers
    inPlaceOps = OrderedDict()
    tensorProducer = OrderedDict()
    for layer in parsedLayers:
        for i in layer.getOutputTensorNames():
            try:
                tensorProducer[i.stringifyName()].append(layer)
            except:
                tensorProducer[i.stringifyName()] = [layer]

        if set(layer.getOutputTensorNames()).intersection(set(layer.getInputTensorNames())):
            assert(len(layer.getOutputTensorNames()) == 1)

            tensorName = layer.getOutputTensorNames()[0]
            try:
                inPlaceOps[tensorName.stringifyName()].append(layer)
            except:
                inPlaceOps[tensorName.stringifyName()] = [layer]

    def remangleName(mangledNameList, matchingName):
        for idx, n in enumerate(mangledNameList):
            if n.stringifyName() == matchingName:
                newName = n.remangle()
                mangledNameList[idx] = newName
                return newName

    def replaceName(mangledNameList, matchingName, newName):
        for idx, n in enumerate(mangledNameList):
            if n.stringifyName() == matchingName:
                mangledNameList[idx] = newName

    for tensorName, layerGroup in inPlaceOps.items():
        extendedList = list(set(tensorProducer[tensorName]).difference(set(layerGroup)))
        extendedList.extend(layerGroup[:-1])
        for producer, consumer in zip(extendedList, layerGroup):
            newName = remangleName(producer.getOutputTensorNames(), tensorName)
            replaceName(consumer.getInputTensorNames(), tensorName, newName)

    return parsedLayers

def createTensors(parsedLayers):
    # Collect all the tensorNames and sizes
    tensorNames = OrderedDict()
    for layer in parsedLayers:
        for tensorName, tensorSize in zip(layer.getInputTensorNames(), layer.getInputTensorSizes()):
            if tensorName.stringifyName() not in tensorNames:
                tensorNames[tensorName.stringifyName()] = UnpopulatedTensor(tensorSize)
                tensorNames[tensorName.stringifyName()].setName(tensorName)

        for tensorName, tensorSize in zip(layer.getOutputTensorNames(), layer.getOutputTensorSizes()):
            if tensorName.stringifyName() not in tensorNames:
                tensorNames[tensorName.stringifyName()] = UnpopulatedTensor(tensorSize)
                tensorNames[tensorName.stringifyName()].setName(tensorName)

    for layer in parsedLayers:
        layer.setInputTensors([tensorNames[n.stringifyName()] for n in layer.getInputTensorNames()])
        layer.setOutputTensors([tensorNames[n.stringifyName()] for n in layer.getOutputTensorNames()])

def insertOutputOps(parsedLayers):
    # Find all tensors that are not consumed by anybody
    tensorNames = OrderedDict()
    for layer in parsedLayers:
        for tensor in layer.getInputTensors():
            if tensor.getName().stringifyName() not in tensorNames:
                tensorNames[tensor.getName().stringifyName()] = ([0, 0], tensor, [])

            tensorNames[tensor.getName().stringifyName()][0][0] += 1

        for tensor in layer.getOutputTensors():
            if tensor.getName().stringifyName() not in tensorNames:
                tensorNames[tensor.getName().stringifyName()] = ([0, 0], tensor, [])

            tensorNames[tensor.getName().stringifyName()][0][1] += 1
            tensorNames[tensor.getName().stringifyName()][2].append(layer)

    for tensorName, tensorValue in tensorNames.items():
        consumersAndProducers, tensor, producers = tensorValue
        consumers = consumersAndProducers[0]
        if consumers == 0:
            x = Output('Output', [tensor.getName()], [])
            x.setInputTensors([tensor])
            x.setOutputTensors([])
            x.loadInputTensorSizes([tensor.getShape()])
            x.loadOutputTensorSizes([])

            assert(len(producers) == 1)
            if isinstance(producers[0], DetectionOutput):
                x.enableDetectionOutput()

            parsedLayers.append(x)

    return parsedLayers


class CaffeParser:

    def calculateReference(self, _, parsedNetwork, arguments):
        """
        Fill the input to the network and execute it, getting the
        expected result.
        """

        global RAND_HI
        global RAND_LO

        img = arguments.image

        # Input Node Name
        if arguments.input_node_name is not None:
            in_label = arguments.input_node_name
            first_layer_idx = list(parsedNetwork._layer_names).index(in_label) - 1
            first_layer = parsedNetwork._layer_names[first_layer_idx]
            in_bottom = parsedNetwork.bottom_names[in_label]
            data_label = in_bottom[0]
            first_layer = in_label
        else:
            in_label = parsedNetwork.inputs[0]
            first_layer = parsedNetwork._layer_names[0]
            data_label = in_label

        # Output Node Name
        if arguments.output_node_name is not None:
            out_label = arguments.output_node_name
        else:
            out_label = parsedNetwork.outputs[0]

        input_shape = parsedNetwork.blobs[data_label].data.shape

        #  Select Input
        img = preprocess(img, arguments, input_shape)

        if img is SeedData.all_ones:
            input_data = np.ones(input_shape)
        elif img is SeedData.all_zeros:
            input_data = np.zeros(input_shape)
        elif img is SeedData.random or img is None:
            if arguments.seed != -1:
                np.random.seed(arguments.seed)
            input_data = np.random.uniform(RAND_LO, RAND_HI, input_shape).astype(dtype=np.float16)
        elif img is SeedData.random_int:
            input_data = np.random.randint(RAND_LO, RAND_HI, input_shape).astype(dtype=np.float16)
        else:
            input_data = img

        input_data = input_data.astype(dtype=np.float16)

        # print(data_label)
        # print("Forward, ", first_layer)
        # quit()

        parsedNetwork.blobs[data_label].data[...] = input_data

        output = parsedNetwork.forward(start=first_layer, end=out_label)
        expected = parsedNetwork.blobs[out_label].data.astype(dtype=np.float16)

        np.save("Fathom_expected.npy", expected)    # TODO: Change to argument given

        return input_data, expected, None

    def loadNetworkObjects(self, prototxt_path, caffemodel_path=None):
        """ Get the original prototxt and caffemodel (optional),
            convert the prototxt from legacy format to current format
            and load the caffemodel (if provided, use fillers otherwise).
        """

        # TODO: There should be a warning and advice if a user is using a legacy format prototxt
        with tempfile.NamedTemporaryFile() as temp_file:
            process = subprocess.Popen(protoConverterPath() + " " + prototxt_path + " " + temp_file.name, shell=True, stdout=subprocess.PIPE)
            process.wait()

            if caffemodel_path:
                parsedNetwork = caffe.Net(temp_file.name, caffemodel_path, caffe.TEST)
            else:
                parsedNetwork = caffe.Net(temp_file.name, caffe.TEST)

            temp_file.seek(0)
            buffer = temp_file.read().decode("utf-8")

            parsedPrototxt = caffe_pb2.NetParameter()
            text_format.Merge(buffer, parsedPrototxt)

            return (parsedPrototxt, parsedNetwork)

    def parse(self, parsedProtoObj, parsedNetworkObj, arguments):
        # Define which subparser needs to be called for each layer
        subParsers = {
            'Input': cp.Input.load,
            'Bias': cp.Bias.load,
            'Convolution': cp.Convolution.load,
            'Deconvolution': cp.Deconvolution.load,
            'Pooling': cp.Pooling.load,
            'ReLU': cp.ReLU.load,
            'Concat': cp.Concat.load,
            'Slice': cp.Slice.load,
            'Eltwise': cp.Eltwise.load,
            'ELU': cp.ELU.load,
            'PReLU': cp.PReLU.load,
            'LRN': cp.LRN.load,
            'InnerProduct': cp.InnerProduct.load,
            'Softmax': cp.Softmax.load,
            'Sigmoid': cp.Sigmoid.load,
            'BatchNorm': cp.BatchNorm.load,
            'Scale': cp.Scale.load,
            'Reshape': cp.Reshape.load,
            'Dropout': cp.Dropout.load,
            'Permute': cp.permute.load,
            'Normalize': cp.Normalize.load,
            'PriorBox': cp.PriorBox.load,
            'DetectionOutput': cp.DetectionOutput.load,
            'Flatten': cp.Flatten.load,
            'TanH': cp.tan_h.load,
            'Crop': cp.crop.load
        }

        parsedLayers = []

        for obj in parsedProtoObj.layer:
            parsedLayers.extend(subParsers[obj.type](obj, parsedNetworkObj))

        cp.Helpers.loadTensorSizes(parsedLayers, parsedNetworkObj)

        # Mangle tensor names
        tensorNamesDict = OrderedDict()
        for layer in parsedLayers:
            for tensorName in list(layer.getInputTensorNames()) + list(layer.getOutputTensorNames()):
                if tensorName not in tensorNamesDict:
                    tensorNamesDict[tensorName] = MangledName(OriginalName(tensorName))

        # Replace tensor names in layers with mangled ones:
        for layer in parsedLayers:
            layer.setInputTensorNames([tensorNamesDict[name] for name in layer.getInputTensorNames()])
            layer.setOutputTensorNames([tensorNamesDict[name] for name in layer.getOutputTensorNames()])

        # Convert inPlace operations into regular ones
        parsedLayers = regularizeInPlaceOps(parsedLayers)

        # Create tensor objects for each operation
        createTensors(parsedLayers)

        # Introduce Output operations
        parsedLayers = insertOutputOps(parsedLayers)

        # drawGraphFromLayers(parsedLayers, '/tmp/test.png')

        return parsedLayers
