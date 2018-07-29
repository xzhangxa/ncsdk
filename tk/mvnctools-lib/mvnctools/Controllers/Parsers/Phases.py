
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
import mvnctools.Controllers.Adaptor as adaptor
from mvnctools.Controllers.Parsers.Parser.Hw import HwOp, HwConvolution, HwFC

from collections import OrderedDict

from mvnctools.Controllers.Parsers.Parser.Input import Input
from mvnctools.Controllers.Parsers.Parser.Output import Output
from mvnctools.Controllers.Optimizer import findClosestChild
from mvnctools.Controllers.Parsers.Parser.DetectionOutput import DetectionOutput
from mvnctools.Controllers.Parsers.Parser.Bias import Bias
from mvnctools.Controllers.Parsers.Parser.Eltwise import Eltwise
from mvnctools.Controllers.Parsers.Parser.LRN import LRN
from mvnctools.Controllers.Parsers.Parser.Square import Square
from mvnctools.Controllers.Parsers.Parser.Pooling import Pooling
from mvnctools.Controllers.Parsers.Parser.Layer import MangledName, OriginalName
from mvnctools.Controllers.Tensor import UnpopulatedTensor, Tensor


from mvnctools.Controllers.GraphUtils import buildGraph, buildLayerLists
from mvnctools.Controllers.Optimizer import debug_graph
from mvnctools.Views.IRVisualize import drawGraph
import networkx as nx


"""
    This file is responsible for each framework-independent 'pass'.
    Thus, it should not rely on any prior framework knowledge, and
    not treat frameworks independently.
"""

def adaptHwWeights(parsedNetwork, parsedLayers,  input_range = (-1, 1), n_batch = 1, per_channel = False):
    '''
    Idea: I want to adapt the output to the dinamics of the DPE accumulator
    the DPE accumulator takes fp16 input from the multiplicator and convert in into
    12q12 fixed point format. using 5 sigma confidence,  the 99.9999% of the output
    data is inside the range [-5 sigma, + 5 sigma]. This will adapt nicely the dynamics

    Standard Convolution/FC operation (* may be a convolution or a matrix multiplication)
    Out = in * W + b
    Out' = in * W' + b' = Out * scale = in * (W*scale) + b*scale

    Variance operator is linear to data scaling
    var{Out'}  = scale * var{Out}

    optimal_range = [-2048.xx; 2047.xx] (12 bit of integer part)
    max{optimal_range} = 5*var{Out'} = 5*scale*var{Out}

    scale = max{optimal_range} / 5 / var{Out}

    This operation can be done (in conv) per output channel
    '''

    in_label = parsedNetwork.inputs[0]
    out_label = parsedNetwork.outputs[0]
    input_shape = parsedNetwork.blobs[in_label].data.shape

    # Generate n_batch of random images
    input_shape = (n_batch, input_shape[1], input_shape[2], input_shape[3])
    img = np.random.uniform(input_range[0], input_range[1], input_shape)

    # run the inference
    parsedNetwork.blobs[in_label].reshape(*input_shape)
    parsedNetwork.blobs[in_label].data[...] = img
    output = parsedNetwork.forward()

    print('Evaluating input and weigths for each hw layer')
    for layer in parsedLayers:
        # TODO: do it for FC!
        # TODO: implement per channel scaling
        if (isinstance(layer, HwConvolution) or isinstance(layer, HwFC)) and layer.hasWeights():

            layerName = layer.name.stringifyOriginalName()
            inBlobName = layer.inputTensors[0].name.stringifyOriginalName()
            outBlobName = layer.outputTensors[0].name.stringifyOriginalName()

            try:
                out_data = parsedNetwork.blobs[outBlobName].data
            except:
                print("Layer {} is not present in the network".format(outBlobName))
                print(parsedNetwork.blobs.keys())
                continue

            # Compute per output channel std
            if per_channel:
                data_std_k =  np.std(out_data, axis = (0, 2, 3))
                data_mean_k = np.mean(out_data, axis = (0, 2, 3))
            else:
                data_std_k =  np.std(out_data)
                data_mean_k = np.mean(out_data)

            scale = np.clip(np.divide(1024 * np.ones(data_std_k.shape), 5 * data_std_k), 1, 16384)


            print('Optimal scale for {0}: {1:.2f}'.format(layerName, scale))
            assert (np.isscalar(scale))

            # Adapt scale and bias
            w = layer.getWeights()
            w.data *= scale
            if layer.biasEnabled():
                b = layer.getBias()
                b.data *= scale

            layer.setScale(1/scale)


    print('--------------------------------------')

    return parsedLayers

def flattenFCLayers(parsedLayers):
    """
        Optimization pass to convert 2D InnerProducts to 1D equivilants.
    """

    from mvnctools.Controllers.Parsers.Parser.InnerProduct import InnerProduct

    for layer in parsedLayers:
        if isinstance(layer, InnerProduct):
            layer.getInputTensors()[0].pprint()
            layer.getOutputTensors()[0].pprint()

            s = layer.getInputTensors()[0].getShape()
            if len(s) - 1 == s.count(1):    # If there is only 1 Dimension
                print("1D")
            else:
                print("2D")
                """
                TO BE IMPLEMENTED
                """
                print("WARNING: Unimplemented 2D FC")

    return parsedLayers

def reinsertOutputOps(parsedLayers):
    # TODO: This is a copy of the one in Caffe. Come up with a proper answer to whose responsibility it is

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


def cropNetwork(parsedLayers, arguments):
    """
        Apply output-node-name and input-node-name
    """

    if arguments.output_node_name is None and arguments.input_node_name is None:
        return parsedLayers

    exit_early = False
    start_late = True if arguments.input_node_name is not None else False

    from mvnctools.Controllers.GraphUtils import buildGraph
    from mvnctools.Controllers.Optimizer import debug_graph
    import networkx as nx

    g = buildGraph(parsedLayers)
    # print("========== BEFORE ===========")
    # debug_graph(g)
    croppedParsedLayers = []

    for index, child in enumerate(nx.lexicographical_topological_sort(g)):
        layer = g.node[child]['ref']

        if index == 0:
            first_layer = layer

        if layer.name.stringifyOriginalName() == arguments.input_node_name:
            start_late = False
            print("Start Layer:", layer)

            pred = list(g.predecessors(child))[0]
            p_node = g.node[pred]['ref']
            p_node = Input("Input", first_layer.getInputTensorNames(), p_node.getOutputTensorNames())

            p_node.setInputTensors(first_layer.getInputTensors())  # UnpopulatedTensor(first_layer.getInputTensorNames()[0])
            p_node.loadInputTensorSizes({})

            p_node.setOutputTensors([layer.getInputTensors()[0]])
            p_node.loadOutputTensorSizes({layer.getInputTensors()})

            print("Source: ", p_node)
            croppedParsedLayers.append(p_node)

        if not exit_early and not start_late:
            croppedParsedLayers.append(layer)
        else:
            if exit_early:
                pred = list(g.predecessors(child))[0]
                pred_node = g.node[pred]['ref']
                print(pred_node.name.stringifyOriginalName(), "==", arguments.output_node_name)

                if exit_early is True and pred_node.name.stringifyOriginalName() == arguments.output_node_name and layer.getInPlace():
                    # Only allow in-place operations to be attached from this point on
                    croppedParsedLayers.append(layer)

        if layer.name.stringifyOriginalName() == arguments.output_node_name:
            # Break the network early as instructed by user
            exit_early = True

    if arguments.output_node_name is not None:
        reinsertOutputOps(croppedParsedLayers)


    g = buildGraph(parsedLayers)
    # print("========== AFTER ===========")
    # debug_graph(g)
    # print("========== ===== ===========")

    return croppedParsedLayers

def forceLayout(parsedLayers):
    for layer in parsedLayers:
        if layer.isHW:
            i_l, o_l = layer.getPingPong()[2]
            from mvnctools.Models.Layouts import NCHW, NHCW

            if i_l == 'P':
                layer.getInputTensors()[0].setLayout(NCHW)
            else:
                layer.getInputTensors()[0].setLayout(NHCW)

            if o_l == 'P':
                layer.getOutputTensors()[0].setLayout(NCHW)
            else:
                layer.getOutputTensors()[0].setLayout(NHCW)

def implicitConcatBinding(parsedLayers):
    """
        In any concat layers, contain the inputs inside of the outputs.
        This will allow the myriad to write the preceding layers in the
        correct location, implicitly performing a concat.
    """

    from mvnctools.Controllers.Parsers.Parser.Concat import Concat

    # print("Implicit concat binding")

    for layer in parsedLayers:
        if isinstance(layer, Concat):

            i = layer.getInputTensors()
            o = layer.getOutputTensors()[0].getTopEncloserRecursive()  # Concat only has one output.
            offset = 0    # TODO: Other axes

            o.setDatatype(np.float16)

            concat_shape = list(i[0].getTopEncloserRecursive().shape)
            concat_shape[layer.getAxis()] = i[0].shape[layer.getAxis()] + sum([t.getShape()[layer.getAxis()] for t in i[1:]])
            concat_shape = tuple(concat_shape)
            if concat_shape != o.shape:
                o_extended = UnpopulatedTensor(concat_shape)
                o_extended.setName(MangledName(OriginalName('')))
                o_extended.setLayout(o.getLayout())
                o_extended.setDatatype(o.getDatatype())
                # print('Place {} ({}) in {} ({}). Offset {}'.format(o.name.stringifyName(), o.shape,
                #         o_extended.name.stringifyName(), o_extended.shape, (0,0,0,0)))
                o.place(o_extended, (0,0,0,0))
                o = o_extended

            concatOutputSize = np.prod(o.getShape()) * np.dtype(o.dtype).itemsize

            for t in i:

                t.setDatatype(np.float16)

                if t.getLayout() == (0, 2, 1, 3):
                    # Interleaved
                    concatOffset = o.getShape()[3] * offset * np.dtype(t.dtype).itemsize
                elif t.getLayout() == (0, 1, 2, 3):
                    # Planar
                    concatOffset = o.getShape()[3] * o.getShape()[2] * offset * np.dtype(t.dtype).itemsize

                corner = [0, 0, 0, 0]
                corner[layer.getAxis()] = offset
                # print('Place {} ({}) in {} ({}). Offset {}'.format(t.getTopEncloserRecursive().name.stringifyName(), t.getTopEncloserRecursive().shape,
                #         o.getTopEncloserRecursive().name.stringifyName(), o.getTopEncloserRecursive().shape, tuple(corner)))
                t.getTopEncloserRecursive().place(o.getTopEncloserRecursive(), tuple(corner))
                for l in parsedLayers:
                    if t in l.getOutputTensors() and isinstance(l, HwOp):
                        l.setConcatParameters(concatOffset, concatOutputSize, o.getShape()[1])
                offset += t.getShape()[layer.getAxis()]

            layer.setImplicit()

    return parsedLayers


def serializeNewFmt(parsedLayers, arguments, myriad_conf, input_data):
    """
        Serializes the output/IR of the new parser into a representation
        that is readable by myriad (blob/graphFile)
    """

    # Find the input and output tensors
    inputTensors = set()
    outputTensors = set()
    outputLayers = set()

    for layer in parsedLayers:
        if type(layer) == Input:
            for tensor in layer.getOutputTensors():
                inputTensors.add(tensor)

        if type(layer) == Output:
            for tensor in layer.getInputTensors():
                outputTensors.add((tensor, layer.isDetectionOutput()))
                outputLayers.add(layer)

    # Find which outputTensor is closest to output_node_name

    from mvnctools.Controllers.Optimizer import buildGraph, debug_graph
    g = buildGraph(parsedLayers)

    # debug_graph(g)

    # If output node provided, select one particular output node.
    # e.g. split
    if arguments.output_node_name:
        selected_output = None
        for layer in parsedLayers:
            if layer.name.stringifyOriginalName() == arguments.output_node_name:
                selected_output = layer

        closestLayer = findClosestChild(g, selected_output, outputLayers)

        outputTensors = set()
        for tensor in closestLayer.getInputTensors():
            outputTensors.add((tensor, layer.isDetectionOutput()))


    # Convert to list for simpler manipulation
    inputTensors = list(inputTensors)
    outputTensors = list(outputTensors)

    print('# Network Input tensors', [tensor.getName().stringifyName() for tensor in inputTensors])
    print('# Network Output tensors', [tensor.getName().stringifyName() for tensor, _ in outputTensors])

    adp = adaptor.Adaptor([input_data], inputTensors, outputTensors)
    adp.transform(parsedLayers)     # TODO: DFS Transform?
    net, blob = adp.serialize(arguments, myriad_conf)

    return net, blob


def compatibilityPadding(parsedLayers):

    from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D
    from mvnctools.Controllers.Tensor import UnpopulatedTensor

    for layer in parsedLayers:
        if isinstance(layer, Convolution2D) and not isinstance(layer, HwOp):
            if True:  # TODO: Enable only on specific implementation
                """
                Gets an offset for a buffer in the work buffer. Relative to the start of the blob file.
                                +----------+
                                |          |
                                |   pad    |
                outputPointer+---> +----------+
                                |          |
                                |  buffer  |
                                |          |
                                +----------+
                                |          |
                                |   pad    |
                                +----------+
                Note: The buffer is overallocated for each stage, because convolutions write
                outside the buffers.
                The maximum pad is computed for RADIX_MAX, stride 1, and SAME padding. (i.e
                the kernel is centered from the first point in the image)
                Let's take radix = 3. For the first output we add up the first two elements
                of the first two rows from the input, of course, multiplied by the kernel weights.
                So for the last kernel (bottom-right), the input has an offset of 1 row and 1 column,
                which is radix/2 * row_width + radix/2. (In our algorithm this means it will write
                starting from -(radix/2 * row_width + radix/2))
                If we put the first kernel (upper-left) to overalp the first point in the image,
                we will get the result for the radix/2 * row_width + radix/2 output element.
                (In our algorithm it will write starting from (radix/2 * row_width + radix/2),
                so it will write outside with the same amount of elements)
                Therefore, the maximum offset has to be radix/2 * row_width + radix/2.
                """

                ot = layer.getOutputTensors()[0]
                ot_shape = ot.getShape()

                pad = layer.getKernelSize()[0]

                pad = (pad // 2) + 1

                padded_shape = tuple((
                    ot_shape[0] + 0,
                    ot_shape[1] + 0,
                    ot_shape[2] + 2 * pad,
                    ot_shape[3] + 0,
                ))

                ot.setDatatype(np.float16)  # TODO: This should be dynamically set
                padding = UnpopulatedTensor(padded_shape)
                padding.setName(ot.getName() + "_pad")
                padding.setLayout(ot.getLayout())
                padding.setDatatype(ot.getDatatype())

                ot.place(padding, (0, 0, pad, 0))

                padding.pprint()
                ot.pprint()

    return parsedLayers


def SplitGroupConvolutions(parsedLayers):
    """
        Group Convolution
        Takes a layer and splits it in half.

        There are side-effects of doing this split that this function solves:

        A network with a group Convolution B (group 2)
                    <  A  >
                       |
                      [T1]
                       |
                    <  B  >
                       |
                      [T2]
                       |
                    <  C  >

        becomes the following when we split:

                     <  A  >
                        |
                     <  #0  >   <- We create a new Operation "#0" so that any
                        |       <- changes we have are 'within' the original scope.
                       [T0]         <- An operation cannot have >1 Input or Output
                       /  \         <- However, Tensors can. So here T0 actually
                     [T1] [T2]      <- contains the tensors T1 and T2
                     /      \
                    |         |
                <  B'  >   <  B''  >    <- The myriad does not have a group convolution
                    \        /          <- implementation, so it is done in parts.
                    [T5]   [T6]
                      \    /     <- See the first note, but imagine it applying to an output.
                       [T7]      <- T7 Contains T5 and T6
                        |
                     <  #3  >
                        |
                       [T8]
                        |
                     <  C  >

    """

    from mvnctools.Controllers.Optimizer import buildGraph, debug_graph, buildLayerLists, addLayerInBetween, newConnection
    import networkx as nx
    from mvnctools.Controllers.Parsers.Parser.Convolution2D import Convolution2D
    from mvnctools.Controllers.Tensor import UnpopulatedTensor
    from mvnctools.Controllers.Parsers.Parser.Layer import MangledName, OriginalName
    import mvnctools.Models.Layouts as Layouts
    from mvnctools.Views.IRVisualize import drawGraph, drawIR
    from mvnctools.Controllers.Parsers.Parser.NoOp import Identity

    g = buildGraph(parsedLayers)

    for index, child in enumerate(nx.topological_sort(g)):
        layer = g.node[child]['ref']

        if isinstance(layer, Convolution2D):
            """
                Convert Layer to Group 0,
                add new layers as siblings for Groups 1-N.
            """
            ngroups = layer.getGroupSize()

            if ngroups <= 1:
                # This is not a *real* group convolution.
                continue

            group = 0

            original_name = layer.name.stringifyName()
            prev_name = list(g.predecessors(original_name))[0]

            # Create a Layer to split the input from the predecessor node.
            input_split = Identity("SplitTensor", None, None)
            is_name = input_split.name.stringifyName()
            g.add_node(is_name, type="OP", ref=input_split)
            g.add_edge(prev_name, is_name)
            # Break the old connection
            g.remove_edge(prev_name, original_name)
            g.add_edge(is_name, original_name)

            input_split.setInputTensorsAllFields(g.node[prev_name]['ref'].getOutputTensors())

            # The Weights of the group convolution are split between the groups.
            taps = layer.getWeights()
            tap_data = taps.data

            # The bulk of the code here deals with changing the existing Convolution
            # to one that represents the first Group of the set.
            input_tensor = layer.getInputTensors()[0]
            input_shape = input_tensor.getShape()

            # Work out how many input channels a group is responsible for.
            # And apply that split
            # Note: Currently the frameworks have an assert here to check compatibility,
            # We don't have to worry about it now, maybe in the future though.
            group_filter = (1, ngroups, 1, 1)
            group_shape = list(map(lambda a, b: a // b, input_shape, group_filter))
            group_channels = group_shape[1]

            group_0_input = UnpopulatedTensor(group_shape)
            group_0_input.setName(MangledName(OriginalName(
                input_tensor.name.stringifyOriginalName() + "_G0"
            )))
            group_0_input.setLayout(Layouts.NHWC)
            input_tensor.setLayout(Layouts.NHWC)
            group_0_input.place(input_tensor, (0, 0, 0, 0))

            layer.setInputTensors([group_0_input])
            layer.loadInputTensorSizes([group_0_input.getShape()])
            layer.setInputTensorNames([group_0_input.name])

            input_split.setOutputTensorsAllFields([group_0_input])

            # We also need to split up the trained weights.
            group_0_weights = tap_data[0:tap_data.shape[0] // ngroups * (group + 1), ]
            layer.setWeights(group_0_weights)

            # And finally, the output.
            layer_ot = layer.getOutputTensors()[0]
            layer_ot.setLayout(Layouts.NHWC)
            orig = list(layer_ot.shape)
            new_ = list((0, orig[1] // ngroups, 0, 0))
            o_slice = [x - y for x, y in zip(orig, new_)]
            outputSlice = UnpopulatedTensor((o_slice))
            outputSlice.setName(MangledName(OriginalName(layer_ot.name.stringifyName() + "_sliced_G" + str(group))))
            outputSlice.setLayout(Layouts.NHWC)
            outputSlice.place(layer_ot, (0, 0, 0, 0))
            layer.setOutputTensorsAllFields([outputSlice])


            if layer.biasEnabled():
                original_bias = layer.getBias().data.flatten()
                bias_data = original_bias
                layer.setBias(bias_data[0:orig[1] // ngroups])
            else:
                layer.setBiasEnabled(False)

            # This layer will only represent a single group now.
            layer.loadGroupSize(1)

            for N in range(ngroups - 1):
                # Create the rest.

                group = N + 1

                # Take our slice of Weights
                group_N_weights = tap_data[taps.data.shape[0] // ngroups * group:
                                   tap_data.shape[0] // ngroups * (group + 2), ]

                # Our slice of Input...
                group_N_input = UnpopulatedTensor(group_shape)
                group_N_input.setLayout(Layouts.NHWC)
                group_N_input.setName(MangledName(OriginalName(
                    input_tensor.name.stringifyOriginalName()
                )))
                group_N_input.setLayout(input_tensor.getLayout())
                group_N_input.place(input_tensor, (0, group_channels * (group), 0, 0))
                input_split.appendOutputTensorsAllFields([group_N_input])

                # Create the new Convolution representhing this group.
                new_name = layer.name.stringifyOriginalName()
                l = Convolution2D(
                    new_name,
                    [group_N_input.name],
                    g.node[prev_name]['ref'].getOutputTensorNames(),
                )
                l.loadGroupSize(1)  # Which also is only one group's worth of execution

                after = list(g.successors(child))[0]

                # Apply our split of Weights
                l.setWeights(group_N_weights)

                # And Input
                l.setInputTensors([group_N_input])
                l.loadInputTensorSizes([group_N_input.getShape()])
                l.setInputTensorNames([group_N_input.name])


                # And Output...
                outputSlice = UnpopulatedTensor((o_slice))
                outputSlice.setName(MangledName(OriginalName(layer_ot.name.stringifyName() + "_sliced_G" + str(group))))
                outputSlice.setLayout(Layouts.NHWC)
                outputSlice.place(layer_ot, (0, (group) * (orig[1] // ngroups), 0, 0))
                l.setOutputTensorsAllFields([outputSlice])

                # Copy some of the other properties of the original convolution.
                (kh, kw) = layer.getKernelSize()
                (sh, sw) = layer.getStrideSize()
                (ph, pw) = layer.getPadding()
                l.loadKernelSize(kh, kw)
                l.loadStrideSize(sh, sw)
                l.loadPadding(ph, pw)
                l.loadDilation(layer.getDilation())

                if layer.biasEnabled():
                    bias_data = original_bias
                    l.setBias(bias_data[group * (orig[1] // ngroups) : (group + 1) * (orig[1] // ngroups)])  # Correct?
                    l.setBiasEnabled(True)
                else:
                    l.setBiasEnabled(False)

                alreadyCreatedCompatibilityLayers = (N != 0)    # Make sure we only create stuff once.

                # Account for the splitting buffers and accumulating buffers as described
                # in the header of this funciton.
                g = newConnection(
                    g,
                    is_name,
                    g.node[after]['ref'].name.stringifyName(),
                    l,
                    childInputAlreadySet=alreadyCreatedCompatibilityLayers
                )

    parsedLayers = buildLayerLists(g)
    return parsedLayers
def squashInPlaceLayers(parsedLayers):

    g = buildGraph(parsedLayers)
    # print("Inplace layers squashing")

    contraction_list = []

    # N - inPlaceNode
    # succ - children of that node
    # pred - parent node
    for n in g.node:
        if g.node[n]['ref'].getInPlace():
            pred = list(g.predecessors(n))
            if len(pred) == 1:
                contraction_list.append((n, pred[0]))
    # Fuse and Remove
    for n, pred in contraction_list:
        # print("Place {} {} in {} {}. Offset: {}". format(
        #                 g.node[pred]['ref'].getOutputTensors()[0].getTopEncloserRecursive().name.stringifyName(), g.node[pred]['ref'].getOutputTensors()[0].getTopEncloserRecursive().shape,
        #                 g.node[n]['ref'].getOutputTensors()[0].name.stringifyName(), g.node[n]['ref'].getOutputTensors()[0].shape, (0, 0, 0, 0)))
        g.node[pred]['ref'].getOutputTensors()[0].getTopEncloserRecursive().place(g.node[n]['ref'].getOutputTensors()[0], (0, 0, 0, 0))
        g.node[n]['ref'].getOutputTensors()[0].setDatatype(np.float16)

    return buildLayerLists(g)


def convertBiasLayersToEltwise(parsedLayers):

    g = buildGraph(parsedLayers)

    for n in g.node:
        layer = g.node[n]['ref']
        if isinstance(layer, Bias):

            g.node[n]['ref'] = Eltwise(layer.name.stringifyName() + "eltBiasConvert", layer.getInputTensorNames(), layer.getOutputTensorNames())
            converted = g.node[n]['ref']
            converted.loadType(Eltwise.Type.WSUM)
            converted.setOutputTensors(layer.getOutputTensors())
            converted.setInputTensors(layer.getInputTensors())
            converted.loadInputTensorSizes([t.getShape() for t in layer.getInputTensors()])
            converted.loadOutputTensorSizes([t.getShape() for t in layer.getOutputTensors()])
            converted.loadCoefficients(layer.getBias())

    parsedLayers = buildLayerLists(g)

    return parsedLayers

def breakUpInnerLRN(parsedLayers):
    """
    Convert from:
    <  X  >
       |
    < LRN >
       |
    <  Y  >

    To
        <   X   >
         /     \
        /       \
    < Square >   \
        |        |
    < AvPool >   |
        |        |
    < InnerLRN > |
          \      |
        < Elt Prod >

    Only Valid for Inner LRN
    """

    g = buildGraph(parsedLayers)
    from mvnctools.Controllers.Optimizer import debug_graph

    # for n in g.node:
    for n in nx.lexicographical_topological_sort(g):
        layer = g.node[n]['ref']
        if isinstance(layer, LRN) and layer.getType() == LRN.Type.WITHIN:

            n = layer.name.stringifyOriginalName()
            itOriginal = layer.getInputTensors()
            otSample = layer.getOutputTensors()[0]

            # Square

            sq_out = UnpopulatedTensor(otSample.shape)
            sq_out.setName(MangledName(OriginalName(n + "_sq_out")))

            sq = Square(n + "_sq", layer.getInputTensorNames(), [sq_out.getName()])
            sq.setInputTensors(layer.getInputTensors())
            sq.setInputTensorNames(layer.getInputTensorNames())
            sq.loadInputTensorSizes(layer.getInputTensorSizes())

            sq.setOutputTensors([sq_out])
            sq.setOutputTensorNames([sq_out.name])
            sq.loadOutputTensorSizes([sq_out.shape])

            g.add_node(sq.name.stringifyName(), type="OP", ref=sq)
            for p in list(g.predecessors(layer.name.stringifyName())):
                g.add_edge(p, sq.name.stringifyName())
                g.remove_edge(p, layer.name.stringifyName())

            # Average Pool

            av_out = UnpopulatedTensor(otSample.shape)
            av_out.setName(MangledName(OriginalName(n + "_av_out")))

            av = Pooling(n + "_av", [sq_out.getName()], [av_out.getName()])
            av.loadType(Pooling.Type.AVE)

            av.loadKernelSize(layer.getSquareKernelSize(), layer.getSquareKernelSize())
            av.loadStrideSize(1, 1)
            pad_size = (layer.getSquareKernelSize() - 1) // 2
            av.loadPadding(pad_size, pad_size)

            av.setInputTensors([sq_out])
            av.setInputTensorNames([sq_out.name])
            av.loadInputTensorSizes([sq_out.shape])

            av.setOutputTensors([av_out])
            av.setOutputTensorNames([av_out.name])
            av.loadOutputTensorSizes([av_out.shape])

            g.add_node(av.name.stringifyName(), type="OP", ref=av)
            g.add_edge(sq.name.stringifyName(), av.name.stringifyName())

            # LRN Partial
            # (1 + alpha * prev) ^ -beta
            layer.setInputTensors([av_out])
            layer.setInputTensorNames([av_out.name])
            layer.loadInputTensorSizes([av_out.shape])
            layer.loadType(LRN.Type.WITHIN_PARTIAL)  # Do not re-enter this check
            layer.loadSquareKernelSize(1)
            g.add_edge(av.name.stringifyName(), layer.name.stringifyName())

            # Elt Prod
            ep_out = UnpopulatedTensor(otSample.shape)
            ep_out.setName(MangledName(OriginalName(n + "_ep_out")))

            ep = Eltwise(n + "_ep", layer.getOutputTensorNames(), [ep_out.getName()])

            ep.loadType(Eltwise.Type.WPROD)

            ep_in = list(layer.getOutputTensors())
            assert len(itOriginal) == 1, "Myriad does not support Within LRN where there is multiple input to the layer"
            ep_in.extend(itOriginal)

            ep.setInputTensors(ep_in)
            ep.setInputTensorNames([l.getName() for l in ep_in])
            ep.loadInputTensorSizes([l.getShape() for l in ep_in])

            ep.setOutputTensors([ep_out])
            ep.setOutputTensorNames([ep_out.name])
            ep.loadOutputTensorSizes([ep_out.shape])

            g.add_node(ep.name.stringifyName(), type="OP", ref=ep)

            for s in list(g.successors(layer.name.stringifyName())):
                g.node[s]['ref'].setInputTensors([ep_out])
                g.node[s]['ref'].setInputTensorNames([ep_out.name])
                g.node[s]['ref'].loadInputTensorSizes([ep_out.shape])
                g.remove_edge(layer.name.stringifyName(), s)
                g.add_edge(ep.name.stringifyName(), s)

            g.add_edge(layer.name.stringifyName(), ep.name.stringifyName())

    parsedLayers = buildLayerLists(g)

    return parsedLayers
