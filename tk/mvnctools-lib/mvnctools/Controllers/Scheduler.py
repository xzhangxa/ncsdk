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
from mvnctools.Controllers.Optimizer import postParsingOptimizations, selectImplementations, streamEverythingSchedule, fixTensors, eliminateNoOps
from mvnctools.Controllers.Parsers.Phases import serializeNewFmt, implicitConcatBinding, flattenFCLayers, compatibilityPadding, SplitGroupConvolutions

from mvnctools.Controllers.Parsers.Phases import adaptHwWeights, cropNetwork, forceLayout, squashInPlaceLayers, convertBiasLayersToEltwise, breakUpInnerLRN
from mvnctools.Views.IRVisualize import drawIR, drawGraphFromLayers

import mvnctools.Controllers.Globals as GLOBALS

def load_myriad_config(no_shaves):
    return MyriadParam(0, no_shaves - 1)

def load_network(arguments, parser, myriad_conf, debug=False):
    """
    Loads the information contained in a network description file into our internal representation.
    It does not calculate buffer pointers for myriad. This is done in a seperate step.
    :param weights:
    :param path: absolute path to network description file
    :param parser: enum indicating which parser should be used.
    :return: Network with attributes populated, bar inputs and outputs.
    """
    if arguments.new_parser:

        if parser == Parser.Caffe:
            from mvnctools.Controllers.Parsers.Caffe import CaffeParser
            p = CaffeParser()
        elif parser == Parser.TensorFlow:
            from mvnctools.Controllers.Parsers.TensorFlow import TensorFlowParser
            p = TensorFlowParser()
        else:
            throw_error(ErrorTable.ParserNotSupported, parser.name)

        # Load From Framework
        modelObj, netObj = p.loadNetworkObjects(arguments.net_description, arguments.net_weights)
        input_data, expected_result, output_tensor_name = p.calculateReference(modelObj, netObj, arguments)
        parsedLayers = p.parse(modelObj, netObj, arguments)

        # Apply Transforms
        if parser != Parser.TensorFlow:
            parsedLayers = cropNetwork(parsedLayers, arguments)
        parsedLayers = breakUpInnerLRN(parsedLayers)

        if parser != Parser.TensorFlow: # TODO: This condition should not exist
            parsedLayers = convertBiasLayersToEltwise(parsedLayers)
        parsedLayers = postParsingOptimizations(parsedLayers)

        bypass_opt = arguments.scheduler is not None
        bypass_opt = bypass_opt if parser == Parser.Caffe else False

        scheduler = NCE_Scheduler()
        parsedLayers = selectImplementations(parsedLayers, scheduler, arguments.ma2480, bypass_opt=bypass_opt)
        if arguments.accuracy_table != {} and not bypass_opt:
            if parser != Parser.TensorFlow: # TODO: The following function should not use netObj
                parsedLayers = adaptHwWeights(netObj, parsedLayers, input_range=(np.min(input_data), np.max(input_data)))
        parsedLayers = SplitGroupConvolutions(parsedLayers)
        parsedLayers = fixTensors(parsedLayers, scheduler, arguments.ma2480)

        if bypass_opt:
            forceLayout(parsedLayers)

        parsedLayers = implicitConcatBinding(parsedLayers)
        parsedLayers = squashInPlaceLayers(parsedLayers)
        net, graphFile = serializeNewFmt(parsedLayers, arguments, myriad_conf, input_data)

        network_dict = {
            'network': net,
            'expected': expected_result,
            'expected_shape': expected_result.shape,
            'expected_layout': None,
            'graph': graphFile
        }
        return network_dict

    if parser == Parser.Debug:
        throw_error(ErrorTable.ParserNotSupported)
    elif parser == Parser.TensorFlow:
        from mvnctools.Controllers.TensorFlowParser import parse_tensor
        parse_ret = parse_tensor(arguments, myriad_conf)
    elif parser == Parser.Caffe:
        from mvnctools.Controllers.CaffeParser import parse_caffe
        parse_ret = parse_caffe(arguments, myriad_conf)
    else:
        throw_error(ErrorTable.ParserNotSupported, parser.name)

    network = parse_ret['network']

    network.finalize()
    network.optimize()
    if arguments.ma2480:
        network.convert_for_hardware()

    return parse_ret
