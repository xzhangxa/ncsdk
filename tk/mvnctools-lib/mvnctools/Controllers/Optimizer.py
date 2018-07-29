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
import sys
import numpy as np
from mvnctools.Models.Network import *
from mvnctools.Models.NetworkStage import *
from mvnctools.Models.MyriadParam import *
from mvnctools.Models.EnumDeclarations import *
import networkx as nx
from mvnctools.Controllers.Parsers.Parser.Conversion import Conversion
from mvnctools.Controllers.Parsers.Parser.Concat import Concat
from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D, Deconvolution, ConvolutionDepthWise2D
from mvnctools.Controllers.Parsers.Parser.Pooling import Pooling
from mvnctools.Controllers.Parsers.Parser.InnerProduct import InnerProduct
from mvnctools.Controllers.Parsers.Parser.Permute import Permute, PermuteFlatten
from mvnctools.Controllers.Parsers.Parser.ReLU import ReLU
from mvnctools.Controllers.Parsers.Parser.Input import Input
from mvnctools.Controllers.Parsers.Parser.Eltwise import Eltwise
from mvnctools.Controllers.Parsers.Parser.Hw import HwOp, HwConvolution, HwConvolutionPooling, HwFC, HwPooling
import mvnctools.Controllers.Globals as GLOBALS
from mvnctools.Controllers.NCE import NCE_Scheduler
from mvnctools.Controllers.Tensor import UnpopulatedTensor, Tensor
from mvnctools.Controllers.Adaptor import convertLayouttoStorageEnum
from mvnctools.Controllers.CnnHardware import hardwareize
from mvnctools.Views.IRVisualize import drawIRNetworkX, drawIR, drawGraph
from mvnctools.Controllers.ConversionPlacement import fixTensors as fixTensorImported
from mvnctools.Controllers.Parsers.Parser.NoOp import NoOp, Identity
from mvnctools.Controllers.Parsers.Parser.Layer import MangledName, OriginalName

import mvnctools.Models.Layouts as Layouts

from mvnctools.Controllers.GraphUtils import buildGraph, buildLayerLists

def postParsingOptimizations(parsedLayers):
    # Build optimization graph
    g = buildGraph(parsedLayers)

    """
        Removes Add or Batch layers by absorbing them into leading Convolutions
    """

    """
        Removes BatchNorm and Scale layers by absorbing them into leading Convolutions
    """
    print("Fusing BatchNorm and Scale after Convolution")
    g = fuseBatchNormScale(g)

    """
        Fuse Permute->Flatten sequences, by absorbing Flatten into Permute
    """
    print("Fusing Permute and Flatten")
    g = fusePermuteFlatten(g)

    """
        Eliminate layers that have been parsed as NoOp (e.g. Dropout)
    """
    print("Eliminate layers that have been parsed as NoOp")
    g = eliminateNoOps(g)

    return buildLayerLists(g)

def selectImplementations(parsedLayers, scheduler, myriadX, bypass_opt=False):
    '''
    Perform several optimization pass to the parsedLayers list
    :param: list of parsed layers from CaffeParser
    :return: TBD
    '''
    # Build optimization graph
    g = buildGraph(parsedLayers)

    if myriadX:
        # Decide about the implementation of each layer
        # print("Replace element-wise sum operations with concats and convolutions")
        # g = eltWiseSumAsConv(g, scheduler)
        print("Hw layers implementation")
        g = implementHwOps(g, scheduler, bypass_opt)

    """
    SW layers integration: At this stage, the input and output layers will be configured, such that:
       1) they are compatible with the hw layers,
       2) no consideration to whether the layer is HW-friendly of not,
       3) the tensors between two sw layer will follow Channel-Minor format,
       4) Concat is not be regarded as sw nor hw layer, which means its inputs/outputs will follow layout of layers connected to it.
    """
    # print("SW layer integration")
    # g = swIntegration(g, myriadX)

    """
    SW layers adaptation: At this pass data layout conversion layers will be inserted in the following situations:
       1) before and/or an hw-unfriendly sw layer if its input and/or output is in hw layout,
       2) applied to an input of a concat to adapt it to preferred input/output layout.
    Furthermore, Copy layer added to the output of a concat which an input to another layer which doesn’t support the stride of the concat output.
    """
    # print("SW layer adaptation")
    # g = swAdaptation(g, )

    if myriadX:
        print("Stream-everything scheduling")
        streamEverythingSchedule(g, scheduler, bypass_opt)

    return buildLayerLists(g)

def streamEverythingSchedule(g, scheduler, bypass_opt=False):
    opt_graph = g.copy()
    opt_graph = optimizeSubgraphs(opt_graph, scheduler)
    optimizeBranches(opt_graph, scheduler)

    # Apply data location
    for stageName in g.node:
        node = g.node[stageName]['ref']
        if isHWLayer(node):
            node.setImplementation(scheduler, bypass_opt)

def fixTensors(parsedLayers, scheduler, myriadX):
    return fixTensorImported(parsedLayers, scheduler, myriadX)


def eliminateNoOps(g):
    """
        Iterates over the graph removing NoOps
    """

    def isNoOp(layer):
        """
            Layers such as Dropout is considered a NoOp for inference
        """
        from mvnctools.Controllers.Parsers.Parser.NoOp import NoOp
        return type(layer) == NoOp

    def noCompensation(layer, absorbed_layer):
        return layer

    check_again = True
    while check_again:
        g, check_again = fuse_nodes(g, isNoOp, lambda layer: True, noCompensation)
    return g


def fuseBatchNormScale(g):
    """
        Iterates over the graph removing any qualifying fusions for
        scale and batchnorm until we are complete.
    """
    def isBatchNormOrScale(layer):
        """
            Returns True/False if the given layer is/is not a BatchNorm or Scale Layer
        """
        from mvnctools.Controllers.Parsers.Parser.BatchNorm import BatchNorm
        from mvnctools.Controllers.Parsers.Parser.Scale import Scale
        return type(layer) in [BatchNorm, Scale]


    def isConv(layer):
        """
            Returns True/False if the given layer is/is not a Convolution Layer
        """
        from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D, ConvolutionDepthWise2D
        return type(layer) in [Convolution2D, ConvolutionDepthWise2D]

    def WeightCompensationBNScale(layer, absorbed_layer):
        """
            To account for the removal of Batch Norm and scale, we adjust the weights
            and bias accordingly.
            See https://github.com/zhang-xin/CNN-Conv-BatchNorm-fusion for reference.
        """
        from mvnctools.Controllers.Parsers.Parser.BatchNorm import BatchNorm

        b = absorbed_layer.getBiasBeta()
        w = absorbed_layer.getMultiplier()

        # Change Weights
        layer.setWeights((layer.getWeights().data.T * w).T)   # Transpose in order to be able to use dimension broadcasting

        # Change Bias
        if layer.biasEnabled():
            if b is not None:
                layer.setBias(layer.getBias().data * w + b)
            else:
                layer.setBias(layer.getBias().data * w)
        else:
            # If there is no bias, it is possible that we will need to now have one
            if b is not None:
                layer.setBiasEnabled(True)
                layer.setBias(np.array(b).astype(np.float16))

        return layer

    check_again = True
    while check_again:
        g, check_again = fuse_nodes(g, isBatchNormOrScale, isConv, WeightCompensationBNScale)

    return g

def fusePermuteFlatten(g):
    """
        Iterates over the graph removing any qualifying fusions for
        Flatten until we are complete.
    """
    import operator

    def isPermute(layer):
        """
            Returns True/False if the given layer is/is not a Permute Layer
        """
        from mvnctools.Controllers.Parsers.Parser.Permute import Permute
        return type(layer) in [Permute]


    def isFusibleFlatten(layer):
        """
            Returns True/False if the given layer is/is not a Flatten Layer
        """
        from mvnctools.Controllers.Parsers.Parser.Flatten import Flatten
        fusible_flatten = type(layer) in [Flatten] and layer.axis == 1 and \
                        layer.end_axis in [-1, 3, None]
        return fusible_flatten


    def FusePermuteFlatten(layer, absorbed_layer):
        fused_layer = PermuteFlatten(layer)
        new_name = 'FUSED_' + layer.name.stringifyOriginalName() + '_' + \
                   absorbed_layer.name.stringifyOriginalName()
        mangled_name = MangledName(OriginalName(new_name))
        fused_layer.setName(mangled_name)
        return fused_layer

    check_again = True
    while check_again:
        g, check_again = fuse_nodes(g, isFusibleFlatten, isPermute, FusePermuteFlatten)

    return g

def findClosestChild(g, source, children):
    """
    As this is a directed graph, we only get paths that move 'downwards'.
    If an equally distant child is found, returns the last discovered entry
    """

    source_name = source.name.stringifyName()

    shortest = None
    shortest_child = None
    for child in children:
        child_name = child.name.stringifyName()
        # Search for children
        try:
            path = nx.shortest_path(g, source=source_name, target=child_name)
        except nx.exception.NetworkXNoPath:
            path = None

        # Check if shorter than before
        if shortest is None:
            shortest = path
            shortest_child = child
        elif path is not None and len(path) <= len(shortest):
            shortest = path
            shortest_child = child

    return shortest_child


def fuse_nodes(g, test_for_fusing_node, test_for_concrete_node, compensation_function):
    """
        A generic function to fuse nodes together.

        We 'fuse' two layers together by absorbing the "Fusing Node" into the "Concrete Node",

        e.g. A -> B becomes A,
        where the 'Fusing node' B is absorbed into the 'Concrete Node' A.

        Parameters:
        @g - graph object from networkx.
        @test_for_fusing_node - condition to match for node 'A' in above example
        @test_for_concrete_node - condition to match for node 'B' in above example
        @compensation_function - once the fuse has been identified, perform this user-provided function to account for
            the loss of this layer.
    """

    contraction_list = []


    # print('Before fusion')
    # for stageName in g.node:
    #     print(stageName, g.node[stageName]['ref'].getStringifiedName(), g.node[stageName]['type'])

    # print('postParsingOptimizations')
    # for stageName in buildGraph(buildLayerLists(g)).node:
    #     print('>>>', stageName, g.node[stageName]['ref'].getStringifiedName(), g.node[stageName]['type'])
    #     for inputNodes in g.node[stageName]['ref'].getInputTensorNames():
    #         print(inputNodes.stringifyName())

    # TODO: Check that the output of concrete node goes only to the fusing node,
    # otherwise we have a logical error.

    # Any qualifying fusions are identified here.
    for n in g.node:
        if test_for_concrete_node(g.node[n]['ref']):
            for succ in g.successors(n):
                if test_for_fusing_node(g.node[succ]['ref']):
                    contraction_list.append((n, succ))

    # Fuse and Remove
    for n, succ in contraction_list:
        old_layer = g.node[succ]['ref']

        # TODO: Compensation.
        g.node[n]['ref'] = compensation_function(g.node[n]['ref'], g.node[succ]['ref'])

        g = nx.contracted_edge(g, (n, succ), self_loops=False)
        g.node[n].pop('contraction', None)

        # Set the output of n to be equal to the output of old_layer
        if g.node[n].get('ref') == None or not hasattr(g.node[n]['ref'], 'setOutputTensors'):
            print("Error: trying to contract Unpopulated Tensor: {}\t{}".format(n, succ))
            continue
        g.node[n]['ref'].setOutputTensors(old_layer.getOutputTensors())
        g.node[n]['ref'].setOutputTensorNames(old_layer.getOutputTensorNames())
        g.node[n]['ref'].loadOutputTensorSizes([t.getShape() for t in old_layer.getOutputTensors()])

    return g, len(contraction_list) > 0

def eltWiseSumAsConv(g, scheduler):
    '''
    Replace hardwareizable Element Wise sum in concat + conv
    the two input blobs are concatenated together (implitly) and then summed elementwise
    using an appropriate 1x1 convolutional kernel
    '''
    replace_lst = []
    for name in g.node:
        layer = g.node[name]['ref']
        if isinstance(layer, Eltwise):
            eltwise_conv, eltwise_concat = layer.convert2Conv()
            if layer.getType() == Eltwise.Type.WSUM and HwConvolution.isHardwarizeable(eltwise_conv, scheduler):
                replace_lst.append((name, (eltwise_conv, eltwise_concat)))

    for name, (eltwise_conv, eltwise_concat) in replace_lst:
        pred_list = list(g.predecessors(name))
        succ_list = list(g.successors(name))
        g.remove_node(name)
        # Add nodes
        g.add_node(eltwise_concat.name.stringifyName(),type="OP", ref=eltwise_concat)
        g.add_node(eltwise_conv.name.stringifyName(),type="OP", ref=eltwise_conv)
        # Add edges
        g.add_edge(eltwise_concat.name.stringifyName(), eltwise_conv.name.stringifyName())
        for pred in pred_list:
            g.add_edge(pred, eltwise_concat.name.stringifyName())
        for succ in succ_list:
            g.add_edge(eltwise_conv.name.stringifyName(), succ)

    return g

def implementHwOps(g, scheduler, bypass_opt = False):
    # Pass 1: decide if layer is hw or sw
    for name in g.node:
        layer = g.node[name]['ref']
        # TODO: Shorten this code.
        if isinstance(layer, Deconvolution):
            if  layer.strideHeight ==1 and layer.strideWidth == 1:
                # stride > 1 in deconv are fractional stride in conv -> not supported
                layer = layer.convert2Conv()

        if isinstance(layer, Convolution2D):# or isinstance(layer, ConvolutionDepthWise2D):
            # check if really layer can be implemented in hw
            if HwConvolution.isHardwarizeable(layer, scheduler):
                hw_layer = HwConvolution(layer)
                # Only for solution and splits
                hw_layer.setImplementation(scheduler, True)
                if bypass_opt:
                    hw_layer.compatible_layouts = [Layouts.NHCW]
                g.node[name]['ref'] = hw_layer

        if isinstance(layer, Pooling):
            # check if really layer can be implemented in hw
            if HwPooling.isHardwarizeable(layer):
                hw_layer = HwPooling(layer)
                if bypass_opt:
                    hw_layer.compatible_layouts = [Layouts.NHCW]
                g.node[name]['ref'] = hw_layer

        if isinstance(layer, InnerProduct):
            # check if really layer can be implemented in hw
            if HwFC.isHardwarizeable(layer):
                hw_layer = HwFC(layer)
                if bypass_opt:
                    hw_layer.compatible_layouts = [Layouts.NHCW]
                g.node[name]['ref'] = hw_layer
            elif layer.canBeConvolution():
                conv_layer = layer.convert2Conv()
                if HwConvolution.isHardwarizeable(conv_layer, scheduler):
                    print('FC layer {} converted as convolution!'.format(conv_layer.getStringifiedName()))
                    hw_layer = HwConvolution(conv_layer)
                    # Only for solution and splits
                    hw_layer.setImplementation(scheduler, True)
                    if bypass_opt:
                        hw_layer.compatible_layouts = [Layouts.NHCW]
                    g.node[name]['ref'] = hw_layer

    # Pass 2: fusing inPlace ops (Prelu, Relu, Scale, Bias etc...)
    check_again = True
    while check_again:
        g, check_again = contractHwOp(g, isInPlace)

    # Pass 3: fusing hw related op (like convolutions and non overlapping pooling)
    g, _ = contractHwOp(g, isNonOvlPooling)

    # return a clean graph
    return g

def hwScheduling(g, scheduler, bypass_opt = False, verbose = False):
    '''
    Idea: optimize every branch of Hw layers of the network independently
    1) optimize subgraphs
    2) optimize individual branches
    '''
    opt_graph = g.copy()
    opt_graph = optimizeSubgraphs(opt_graph,scheduler)
    optimizeBranches(opt_graph, scheduler)

    # ApplyAdapt shapes
    for stageName in g.node:
        node = g.node[stageName]['ref']
        if isHWLayer(node):
            node.setImplementation(scheduler, bypass_opt)
            in_ch, out_ch, _ = node.getSolution()
            splits = node.getSplitOverC()
            if isinstance(node, HwConvolution):
                in_ch *= splits
            for tensor in node.inputTensors:
                tensor.proposeShape((tensor.shape[0], in_ch, tensor.shape[2], ((tensor.shape[3] + 7)//8)*8))

            for tensor in node.outputTensors:
                tensor.proposeShape((tensor.shape[0], out_ch, tensor.shape[2], ((tensor.shape[3] + 7)//8)*8))


    for stageName in g.node:
        if isHWLayer(g.node[stageName]['ref']):
            g.node[stageName]['ref'].adaptTensors()

    if verbose:
        print('************************************************')
        for stageName in  g.node:
            try:
                print(stageName, scheduler.ordered_dict[stageName])
            except Exception as e:
                pass
        print('************************************************')

def swIntegration(g, myriadX, verbose = False):

    # TODO: Naive approach, change
    # Preference order: interleaved, planar, channel minor
    if myriadX:
        preference_layouts = [Layouts.NHCW, Layouts.NCHW, Layouts.NHWC]
    else:
        preference_layouts = [Layouts.NHWC]
    for name in nx.lexicographical_topological_sort(g):
        layer = g.node[name]['ref']
        if layer.getInputTensorsCount() < 2:
            for layout in preference_layouts:
                # if isinstance(layer, Input):
                #     if layout in g.node[g.successors(name)[0]]['ref'].compatible_layouts:
                #         break
                # else:
                    if layout in layer.compatible_layouts:
                        break
            for outputTensor in layer.outputTensors:
                print("Set layout of {}: {}".format(outputTensor.name, layout))
                outputTensor.setLayout(layout)
        else:
            in_layouts = [x.getLayout() for x in layer.inputTensors]

            if len(set(in_layouts)) != 1:
                # Minimize the number of eventual conversion layer
                layout = max(in_layouts,key=in_layouts.count)
            else:
                layout = in_layouts[0]

            layer.compatible_layouts = [layout]
            for outputTensor in layer.outputTensors:
                outputTensor.setLayout(layout)

    return g

def swAdaptation(g, verbose = False):
    for child in nx.lexicographical_topological_sort(g):
        layer = g.node[child]['ref']
        for inputTensor, inputTensorName in zip(layer.inputTensors, layer.inputTensorNames):
            if not inputTensor.getLayout() in [x for x in layer.compatible_layouts]:
                # remove previous edge and add connection in and from
                predecessors = list(g.predecessors(child))
                for parent in predecessors:
                    if inputTensorName in g.node[parent]['ref'].outputTensorNames:
                        print("Add a conversion layer for tensor {}".format(inputTensorName))
                        new_layer_name = '{}_converted'.format(child)
                        addLayerInBetween(g, parent, child, Convert, new_layer_name)
                        g.node[new_layer_name]['ref'].setLayouts(inputTensor.getLayout(), layer.inputTensors[0].getLayout())

    return g

# TODO: remove legacy code, reimplement with networkx
def optimizeSubgraphs(g, scheduler):
    '''
    Optimize each subgraph and then contract edges
    '''
    sorted_layers = list(nx.lexicographical_topological_sort(g))
    subgraphs = extractSubgraphs(g)
    for subgraph, subgraph_start, subgraph_end in subgraphs:
        net = genOptNetwork(subgraph)
        # find last element of subgraph (if subgraph in CMX, the one that need CMX unload)
        last_elem = list(filter(lambda x: get_null_terminating_name(x.name) == sorted_layers[sorted_layers.index(subgraph_end) - 1], net.stageslist))
        scheduler.process_subgraph(net.stageslist,net, 0, net.stageslist.index(last_elem[0])+1, 'D')

        g = contractSubgraph(g, subgraph_start, subgraph_end)

    return g

def optimizeBranches(g,scheduler):

    branches = extractBranches(g)

    print("Found {} branches".format(len(branches)))
    for idx, branch in enumerate(branches):
        sg = g.subgraph(branch)
        print('\t[{}]: {}'.format(idx, list(nx.lexicographical_topological_sort(sg))))
        net = genOptNetwork(sg)
        scheduler.process_branch(net.stageslist, initial_config = 'D', final_config = 'D')

# TODO: add support for PreOp
def contractHwOp(g, test):
    contraction_list = []
    # Contract the blob into the operation
    for n in g.node:
        if isinstance(g.node[n]['ref'], HwConvolution):
            for succ in g.successors(n):
                if test(g.node[succ]['ref']):
                    contraction_list.append((n,succ))

    # Contract PostOp
    for n,succ in contraction_list:
        old_layer = g.node[succ]['ref']
        if isNonOvlPooling(old_layer):
            # Can fuse only if split over c is == 1
            if (g.node[n]['ref'].getSplitOverC() == 1) and HwConvolutionPooling.canFuse(conv = g.node[n]['ref'], pool = g.node[succ]['ref']):
                print("Fusing convolution {} with non overlapping pooling {}".format(n,succ))
                g.node[n]['ref'] = HwConvolutionPooling(g.node[n]['ref'])
                g.node[n]['ref'].setPoolingParameter(old_layer)
            else:
                continue
        else:
            print("Fusing {} with {}".format(n, succ))
            g.node[n]['ref'].setPostOp(old_layer)

        # Set the output of n to be equal to the output of old_layer
        g.node[n]['ref'].setOutputTensors(old_layer.getOutputTensors())
        g.node[n]['ref'].setOutputTensorNames(old_layer.getOutputTensorNames())
        g.node[n]['ref'].loadOutputTensorSizes([t.getShape() for t in old_layer.getOutputTensors()])

        # # Adjust tensors
        # for tensor in set.intersection(set(g.node[n]['ref'].outputTensors), set(g.node[succ]['ref'].inputTensors)):
        #     out_tensor_idx  = g.node[n]['ref'].outputTensors.index(tensor)

        #     assert len(g.node[succ]['ref'].outputTensorNames) == 1

        #     g.node[n]['ref'].outputTensorNames[out_tensor_idx] = g.node[succ]['ref'].outputTensorNames[0]
        #     g.node[n]['ref'].outputTensorSizes[out_tensor_idx] = g.node[succ]['ref'].outputTensorSizes[0]

        #     tensor_lst = list(g.node[n]['ref'].outputTensors)
        #     tensor_lst[out_tensor_idx] = g.node[succ]['ref'].outputTensors[0]
        #     g.node[n]['ref'].setOutputTensors(tensor_lst)

        g = nx.contracted_edge(g, (n, succ), self_loops=False)
        g.node[n].pop('contraction', None)

    return g, len(contraction_list) > 0

def contractSubgraph(g, start_subgraph, end_subgraph):
    found_end = False
    while not found_end:
        found_end = True
        for succ in g.successors(start_subgraph):
            if succ != end_subgraph:
                g = nx.contracted_nodes(g, start_subgraph, succ, self_loops=False)
                found_end = False
    return g

def extractSubgraphs(g):
    '''
    This fuinction generates N subgraphs
    '''
    subgraphs = []
    for name in g.node:
        if type(g.node[name]['ref']) == Concat:
            pred_concat = []
            for pred in g.predecessors(name):
                ancestor = nx.ancestors(g,pred)
                pred_concat.append(set.union(ancestor, {pred}))

            # List all predecessors
            pred_lst = list(set.intersection(*pred_concat))
            start = min([(node, nx.shortest_path_length(g,node, name)) for node in pred_lst], key = lambda x: x[1])[0]

            for i, tt in enumerate(nx.all_simple_paths(g,source=start,target=name)):
                if i > 10:
                    break
            if i > 10:
                continue

            # Extract subgraph
            paths_between_generator = nx.all_simple_paths(g,source=start,target=name)
            sg = g.subgraph({node for path in paths_between_generator for node in path if node != name})
            # Get the subgraph depth
            if (nx.dag_longest_path_length(sg) <= 2) and (nx.shortest_path_length(g,source=start, target=name) > 1):
                subgraphs.append((sg, start, name))

    return subgraphs



def extractBranches(g):

    # Remove element with multiple child/parents
    edge_exclusion_list = []
    for name in g.node:
        if len(list(g.predecessors(name))) > 1:
            for pred in g.predecessors(name):
                edge_exclusion_list.append((pred, name))
        if len(list(g.successors(name))) > 1:
            for succ in g.successors(name):
                edge_exclusion_list.append((name,succ))
    # Remove all the edges
    g.remove_edges_from(edge_exclusion_list)

    # Remove non-hw layer
    node_exclusion_list = []
    for name in g.node:
        if not isHWLayer(g.node[name]['ref']):
            node_exclusion_list.append(name)
    g.remove_nodes_from(node_exclusion_list)

    return list(nx.connected_components(g.to_undirected()))


def newConnection(g, parentName, childName, new_layer,
                  childInputAlreadySet=False):
    """
        Inserts a new entry into the graph.
        childOutput already set tries to make sure they have the same output buffer

        @childInputAlreadySet - If True, will attempt to create buffers and NoOp layers.
            If False, it will just do simple attachments.
    """
    new_layer_name = new_layer.name.stringifyName()
    new_layer_original_name = new_layer.name.stringifyOriginalName()
    g.add_node(new_layer_name, type="OP", ref=new_layer)
    g.add_edge(parentName, new_layer_name)

    childLayer = g.node[childName]['ref']
    parentLayer = g.node[parentName]['ref']

    if not childInputAlreadySet:
        filler = Concat("CombineInputs", None, None)
        fillerName = filler.name.stringifyName()

        sameLevelNodes = list(g.predecessors(childName))

        g.add_node(fillerName, type="OP", ref=filler)
        g.add_edge(fillerName, childName)
        g.add_edge(new_layer_name, fillerName)

        new_out = UnpopulatedTensor(g.node[sameLevelNodes[0]]['ref'].getOutputTensors()[0].getShape())
        new_out.setName(MangledName(OriginalName(str(new_layer_original_name))))
        new_out.setLayout(Layouts.NHWC)

        new_layer_OT_ORIGINAL = new_layer.getOutputTensors()[0]
        new_layer.setOutputTensorsAllFields([new_out])

        ot_sln = g.node[sameLevelNodes[0]]['ref'].getOutputTensors()[0]
        containerTensor = UnpopulatedTensor(ot_sln.getTopEncloserRecursive().getShape())
        containerTensor.setLayout(Layouts.NHWC)
        containerTensor.setName(MangledName(OriginalName("inplace_accumulation")))

        filler.setInputTensors([])
        filler.loadInputTensorSizes([])
        filler.setInputTensorNames([])

        for slname in sameLevelNodes:
            slnode = g.node[slname]['ref']
            sln_orignal_name = g.node[slname]['ref'].name.stringifyOriginalName()
            old_output_slnode = slnode.getOutputTensors()[0]

            old_output_slnodeEnc = old_output_slnode.getTopEncloserRecursive()
            filler_in = UnpopulatedTensor(old_output_slnode.shape)
            filler_in.setLayout(Layouts.NHWC)
            filler_in.setName(MangledName(OriginalName(str(sln_orignal_name))))
            filler.appendInputTensorsAllFields([filler_in])
            slnode.setOutputTensorsAllFields([filler_in])
            g.add_edge(slname, fillerName)
            g.remove_edge(slname, childName)

            childLayer.removeInputTensorsAllFields([old_output_slnodeEnc])

        filler.appendInputTensorsAllFields([new_out])

        filler.setOutputTensorsAllFields([containerTensor])
        childLayer.appendInputTensorsAllFields([containerTensor])   # TODO: Remove

    else:

        childLayer = g.node[childName]['ref']
        succLayer = g.node[new_layer_name]['ref']

        childLayer.appendInputTensorsAllFields(succLayer.getOutputTensors())

        g.add_edge(new_layer_name, childName)

    return g


# Add a layer in between two node. Adjust tensors and edges
# Note: Requires a prior between parent and child connection
def addLayerInBetween(g, parentName, childName, new_layer_class, new_layer_name, preserve_layout=True):

    childLayer = g.node[childName]['ref']
    parentLayer = g.node[parentName]['ref']

    tensorNames = list(set.intersection(set(parentLayer.outputTensorNames), set(childLayer.inputTensorNames)))
    tensors = list(set.intersection(set(parentLayer.outputTensors), set(childLayer.inputTensors)))

    # Check for errors
    assert len(tensorNames) == len(tensors)
    assert len(tensorNames) == 1
    assert len(tensors) == 1

    # this is the tensor and the tensor name in common between the two layer
    inputTensor = tensors[0]
    inputTensorName = tensorNames[0].stringifyOriginalName()

    if preserve_layout:
        layout = childLayer.inputTensors[0].getLayout()
    # Create a new tensor for layout conversion
    ConvertedTensor = UnpopulatedTensor(shape = inputTensor.shape)
    for attr_name in inputTensor.__dict__:
        setattr(ConvertedTensor, attr_name, getattr(inputTensor, attr_name))
    if preserve_layout:
        ConvertedTensor.setLayout(layout)

    ConvertedTensor.setName(MangledName(OriginalName('{}_converted'.format(inputTensorName))))

    # Create the new layer
    convert_layer = new_layer_class(new_layer_name, [tensorNames[0]], [ConvertedTensor.getName()])
    convert_layer.setInputTensors([inputTensor])
    convert_layer.setOutputTensors([ConvertedTensor])

    # Same input and output size
    convert_layer.inputTensorSizes = childLayer.inputTensorSizes
    convert_layer.outputTensorSizes = childLayer.inputTensorSizes

    g.add_node(new_layer_name, type="OP", ref=convert_layer)
    g.add_edge(new_layer_name, childName)

    # The child needs to be set to the new (relative) parent
    childLayer.setInputTensors([ConvertedTensor])
    childLayer.setInputTensorNames([ConvertedTensor.name])
    childLayer.loadInputTensorSizes([ConvertedTensor.shape])

    # remove previous edge and add connection in and from
    if tensorNames[0] in g.node[parentName]['ref'].outputTensorNames:
        g.add_edge(parentName, new_layer_name)
        g.remove_edge(parentName, childName)

    return g

def genOptNetwork(g):
    net = Network(g.name, None)
    for name in nx.lexicographical_topological_sort(g):
        net.stageslist.append(OptStage(g, name))
    # add tail
    for stage in net.stageslist:
        for name in stage.tail_names:
            stage.tail.extend([x for x in net.stageslist if get_null_terminating_name(x.name) == name])

    return net

def debug_graph(g):
    for name in nx.lexicographical_topological_sort(g):
        print('--------------')
        print('name', name)
        node = g.node[name]['ref']
        print(type(node))
        print('predecessors', list(g.predecessors(name)))
        print('successors', list(g.successors(name)))
        if g.node[name]['type'] == 'OP':
            print('input tensor', [x.stringifyName() for x in node.inputTensorNames], "TopEncloser:", [x.getTopEncloserRecursive().name.stringifyName() for x in node.inputTensors], node.inputTensorSizes)
            print('output tensor', [x.stringifyName() for x in node.outputTensorNames], "TopEncloser:", [x.getTopEncloserRecursive().name.stringifyName() for x in node.outputTensors] ,  node.outputTensorSizes)
            if 'contraction' in g.node[name].keys():
                print('Contractions', g.node[name]['contraction'])
        else:
            print('shape', node.shape)
            if hasattr(node, 'layout'):
                print('layout', node.layout)

def isHWLayer(layer):
    return layer.isHW

def isInPlace(layer):
    if isinstance(layer, ReLU):
        return True
    return False

def isNonOvlPooling(layer):
    if isinstance(layer, HwPooling):
        if (layer.kernelHeight <= layer.strideHeight) and (layer.kernelWidth <= layer.strideWidth):
            return True
    return False


def convertLayout(layout):
    encodedLayout = {
        Layouts.NHCW : 'I',
        Layouts.NCHW : 'P'
    }
    return encodedLayout[layout]

# Optimization Stage
class OptStage():
    def __init__(self, graph, name):
        self.g = graph
        self.name = str.encode(name)
        self.tail_names = list(self.g.successors(name))
        self.tail = []
        self.top = list(self.g.predecessors(name))
        self.layer = self.g.node[name]['ref']
        self.op = self.getOp()

        self.inputDimZ = 0
        self.inputDimX = 0
        self.inputDimY = 0
        self.outputDimZ = 0
        self.outputDimX = 0
        self.outputDimY = 0


        if self.op in [StageType.convolution, StageType.fully_connected_layer]:
            self.bias = None
            if self.layer.biasEnabled():
                self.bias = self.layer.getBias().data

            # Convert to hwck for hardwerize compatibility
            if self.op == StageType.convolution:
                self.taps = np.transpose(self.layer.getWeights().data, (2, 3, 1, 0))
            else:
                self.taps = np.transpose(self.layer.getWeights().data, (0, 1, 3, 2))
            # TODO: Implement ME!!
            self.scale = None

        for tensor in self.layer.inputTensors:
            size = tensor.getShape()#tensor.getTopEncloserRecursive().getShape()
            self.inputDimZ  += size[1]
            self.inputDimX  += size[3]
            self.inputDimY  += size[2]

        for tensor in self.layer.outputTensors:
            size = tensor.getTopEncloserRecursive().getShape()
            self.outputDimZ += size[1]
            self.outputDimX += size[3]
            self.outputDimY += size[2]

        if  self.op in [StageType.convolution, StageType.average_pooling, StageType.max_pooling]:

            self.radixX = self.layer.kernelWidth
            self.radixY = self.layer.kernelHeight
            self.strideX = self.layer.strideWidth
            self.strideY = self.layer.strideHeight
            self.padX = self.layer.paddingWidth
            self.padY = self.layer.paddingHeight

        if self.op == StageType.fully_connected_layer:
            s = self.layer.inputTensors[0].getTopEncloserRecursive().getShape()
            if len(s) - 1 != s.count(1):
            # if len(np.squeeze(s)) > 1:
                self.inputDimZ *= self.inputDimX * self.inputDimY
                self.inputDimX = 1
                self.inputDimY = 1
                raise ValueError("Cannot support 3D input for HW FC")

    def getOp(self):
        if isinstance(self.layer, HwConvolution) or isinstance(self.layer, HwConvolutionPooling):
            return StageType.convolution
        elif isinstance(self.layer, HwPooling):
            if self.layer.type == Pooling.Type.MAX:
                return StageType.max_pooling
            else:
                return StageType.average_pooling
        elif isinstance(self.layer, HwFC):
            return StageType.fully_connected_layer
        else:
            # raise ValueError("Try to schedule non hw layer: {}".format(type(self.layer)))
            return None
