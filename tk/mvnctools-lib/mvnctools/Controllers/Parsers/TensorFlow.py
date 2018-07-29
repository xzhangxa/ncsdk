# Copyright 2017 Intel Corporation.
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
import sys

# Tensorflow 1.7 has some deprecated features that result in warnings when it is
# imported. Suppress these warnings until TF resolves them.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

import google.protobuf as proto
import numpy as np
import networkx as nx

from collections import OrderedDict
import math
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops

from mvnctools.Controllers.EnumController import throw_error, ErrorTable
from mvnctools.Controllers.Parsers.Parser.Layer import OriginalName, MangledName
from mvnctools.Controllers.Parsers.Parser.Output import Output
from mvnctools.Controllers.Parsers.Parser.DetectionOutput import DetectionOutput
from mvnctools.Controllers.Parsers.Parser.Bias import Bias
from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D

from mvnctools.Controllers.Parsers import TensorFlowParser as tfp
from mvnctools.Controllers.Tensor import UnpopulatedTensor

from mvnctools.Controllers.MiscIO import parse_img
from mvnctools.Controllers.GraphUtils import buildGraph, buildLayerLists



inputnode = 'input'
debug = False


# TensorFlow files contain many nodes that are not relevant to inference.  This
# function removes all layers that are not ancecstors of the output layers.
def pruneNodes(parsedLayers, output_node_name):
    outputMangledNames = [output_layer.getName().stringifyName() for output_layer in parsedLayers if output_layer.getName().stringifyOriginalName() == output_node_name]
    g = buildGraph(parsedLayers)
    outputNode = outputMangledNames[0]
    croppedNodes = nx.ancestors(g, outputNode)
    croppedNodes.add(outputNode)
    croppedGraph = nx.subgraph(g, croppedNodes)
    croppedLayers = buildLayerLists(croppedGraph)
    return croppedLayers


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

def insertOutputOps(parsedLayers, output_name):
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
        if consumers == 0 and tensor.getName().stringifyOriginalName() == output_name:
            x = Output('output', [tensor.getName()], [])
            x.setInputTensors([tensor])
            x.setOutputTensors([])
            x.loadInputTensorSizes([tensor.getShape()])
            x.loadOutputTensorSizes([])

            assert(len(producers) == 1)
            if isinstance(producers[0], DetectionOutput):
                x.enableDetectionOutput()

            parsedLayers.append(x)

    return parsedLayers


class TensorFlowParser:

    # calculate the reference output of the graph to compare with myriad results
    def calculateReference(self, tfModel, sess, arguments):

        global inputnode

        image = arguments.image
        input_node_name = arguments.input_node_name
        output_node_name = arguments.output_node_name

        if input_node_name is not None:
            inputnode = input_node_name
        if output_node_name is None:
            output_node_name = 'output'


        with sess.as_default():
            try:
                inputTensor = tfModel.get_tensor_by_name(inputnode + ':0')
            except:
                throw_error(ErrorTable.NoOutputNode, inputnode)

            try:
                outputTensor = tfModel.get_tensor_by_name(output_node_name + ':0')
            except:
                throw_error(ErrorTable.NoOutputNode, output_node_name)
            print("output tensor shape", outputTensor.get_shape())

            shape = inputTensor.get_shape()
            if isinstance(shape, tf.TensorShape):
                shape_list = shape.as_list()
                # Tensorflow can have None in the batch size field of the
                # input shape, if that is the case then set it to 1
                if None == shape_list[0]:
                    shape_list[0] = 1
                    shape = shape_list
                    inputTensor.set_shape(shape)
                elif None in shape:
                    print("No shape information for tensor")

            if image is None or image == "Debug":
                input_data = np.random.uniform(0, 1, shape)
                print("Input image shape", shape)
            else:
                input_data = parse_img(image,
                                    [int(shape[0]),
                                        int(shape[3]),
                                        int(shape[1]),
                                        int(shape[2])],
                                    raw_scale=arguments.raw_scale,
                                    mean=arguments.mean,
                                    channel_swap=arguments.channel_swap)
                input_data = input_data.transpose([0, 2, 3, 1])

            feed_dict = {inputnode + ':0': input_data}
            tf.global_variables_initializer()
            expected_result = outputTensor.eval(feed_dict)

            # convert shape
            input_data = np.transpose(input_data, (0, 3, 1, 2))
            if len(expected_result.shape) == 4:
                expected_result = np.transpose(expected_result, (0, 3, 1, 2))
            elif len(expected_result.shape) == 3:
                pass
            elif len(expected_result.shape) == 2:
                expected_result = expected_result.reshape(1, expected_result.shape[1], expected_result.shape[0])
            else:
                expected_result = expected_result.reshape(1, 1, expected_result.shape[0])

        return input_data, expected_result, outputTensor.name


    def loadNetworkObjects(self,graph_path, model_path=None):
        """ Get the tensorflow protobuff model and parse it via tensorflow
        """
        tfModel = None
        print(graph_path)

        sess = tf.Session()
        with sess.as_default():
            filetype = graph_path.split(".")[-1]
            if filetype == 'pb':
                graph_def = graph_pb2.GraphDef()
                with open(graph_path, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            else:
                saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
                if saver is not None:
                    #print(graph_path[:graph_path.rfind('.')])
                    saver.restore(sess, graph_path[:graph_path.rfind('.')])

            tfModel = tf.get_default_graph()

            # allow default session for obtaining variable values

        return tfModel, sess

    def parse(self, tfModel, sess, arguments):


        if arguments.output_node_name is None:
            output_node_name = 'output'
        else:
            output_node_name = arguments.output_node_name

        if arguments.output_node_name is None:
            input_node_name = 'input'
        else:
            input_node_name = arguments.input_node_name

        output_tensor_name = output_node_name + ':0'


        with sess.as_default():

            input_node_name = arguments.input_node_name
            filename = arguments.outputs_name

            if debug:
                for idx, node in enumerate(tfModel.get_operations()):
                    print("       ", idx, node.type, node.name)
                    for a in node.inputs:
                        print("           IN:", a.name)
                    for a in node.outputs:
                        print("           OUT:", a.name)

            # Define which subparser needs to be called for each layer
            subParsers = {
                'Placeholder': tfp.Placeholder.load,
                'Conv2D': tfp.Convolution.load,
                'MaxPool': tfp.Pooling.load,
                'AvgPool': tfp.Pooling.load,
                'Relu': tfp.ReLU.load,
                'BiasAdd': tfp.BiasAdd.load,
                'Concat': tfp.Concat.load,
                'ConcatV2': tfp.Concat.load,
                'Identity' : tfp.Identity.load,
                'Slice': tfp.Slice.load,
                'Add': tfp.Eltwise.load,
                'Mul': tfp.Eltwise.load,
                'Sub': tfp.Eltwise.load,
                # 'ELU': tfp.ELU.load,
                # 'PReLU': tfp.PReLU.load,
                'LRN': tfp.LRN.load,
                # 'InnerProduct': tfp.InnerProduct.load,
                'Softmax': tfp.Softmax.load,
                'FusedBatchNorm': tfp.FusedBatchNorm.load,
                # 'Scale': tfp.Scale.load,
                # TODO: Needs to be checked first.
                'Reshape' : tfp.Reshape.load,
                # 'Dropout' : tfp.Dropout.load,
                'Pad' : tfp.Pad.load,
                'Maximum' : tfp.Eltwise.load,
                'RealDiv' : tfp.RealDiv.load
            }

            parsedLayers = []
            operations = tfModel.get_operations()
            for obj in operations:
                opKey = subParsers.get(obj.type, None)
                if opKey is not None:
                    parsedLayers.extend(opKey(obj, operations))
                if obj.name == output_node_name:
                    break

            tfp.Helpers.loadTensorSizes(parsedLayers, tfModel)

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

            # Prune subgraphs
            parsedLayers = pruneNodes(parsedLayers, output_node_name)

            if debug: # print set of layers not yet implemented
                print('Layers not yet implemented:')
                print(set([x.type for x in operations if x.type not in subParsers]))

            # Create tensor objects for each operation
            createTensors(parsedLayers)

            # Introduce Output operations
            parsedLayers = insertOutputOps(parsedLayers, output_tensor_name)

            print("Fusing Add and Batch after Convolution")
            g = buildGraph(parsedLayers)
            g = fuseBiasAdd(g)
            parsedLayers = buildLayerLists(g)

        return parsedLayers



def fuseBiasAdd(g):
    from mvnctools.Controllers.Parsers.Parser.InnerProduct import InnerProduct
    from mvnctools.Controllers.Optimizer import fuse_nodes

    """
        Iterates over the graph removing any qualifying fusions for
        bias and add until we are complete.
    """

    def isBiasOrAdd(layer):
        """
            Returns True/False if the given layer is/is not a Bias or Add Layer
        """
        from mvnctools.Controllers.Parsers.Parser.Bias import Bias
        from mvnctools.Controllers.Parsers.Parser.Eltwise import Eltwise
        return (type(layer) == Bias) or \
                ((type(layer) == Eltwise) and (layer.getType() == Eltwise.Type.WSUM))

    def isConvOrFC(layer):
        """
            Returns True/False if the given layer is/is not a Convolution or InnerProduct Layer
        """
        from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
        return type(layer) in [Convolution2D, ConvolutionDepthWise2D, InnerProduct]

    def PutBiasInConv(layer, absorbed_layer):
        """
            To account for the removal of Bias and Add, we insert the bias
        """
        from mvnctools.Controllers.Parsers.Parser.Bias import Bias
        from mvnctools.Controllers.Parsers.Parser.Eltwise import Eltwise

        if (type(absorbed_layer) == Bias):
            b = absorbed_layer.getBias()
        else:
            # TODO: pull the data out of the Add input
            b = 0

        # Change Bias
        if layer.biasEnabled():
            if b is not None:
                layer.setBias(layer.getBias().data + b)
        else:
            # If there is no bias, it is possible that we will need to now have one
            if b is not None:
                layer.setBiasEnabled(True)
                layer.setBias(np.array(b).astype(np.float16))
        return layer

    check_again = True
    while check_again:
        g, check_again = fuse_nodes(g, isBiasOrAdd, isConvOrFC, PutBiasInConv)

    return g