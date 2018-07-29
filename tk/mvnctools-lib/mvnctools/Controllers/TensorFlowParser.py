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
import tensorflow as tf
import google.protobuf as proto
import numpy as np
import math
import re
from mvnctools.Models.Network import *
from mvnctools.Models.NetworkStage import *
from mvnctools.Models.EnumDeclarations import *
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops
from mvnctools.Controllers.TensorFlowPreproc import TFPreprocessor
from mvnctools.Controllers.TensorFlowPreproc import PatternType

placeholder_dict = {}
const_dict = {}
node_dict = {}
variable_dict = {}
concat_tracker = []
reshape_tracker = []
identity_tracker = []
padding_tracker = []
inputnode = 'input'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def apply_padding(pad_type, in_dim, kernel_dim, stride_dim):
    if pad_type == b'SAME':
        return same_padding(in_dim, kernel_dim, stride_dim)
    elif pad_type == b'VALID':
        return valid_padding(in_dim, kernel_dim, stride_dim)
    else:
        throw_error(
            ErrorTable.StageDetailsNotSupported,
            "No Such Pad Type Supported." +
            str(pad_type))


def same_padding(in_dim, kernel_dim, stride_dim):
    """
    Calculates the output dimension and also the padding required for that dimension.
    :param in_dim: Width/Height of Input
    :param kernel_dim: Width/Height of Kernel
    :param stride_dim: Vertical/Horizontal Stride
    """
    output_dim = math.ceil(float(in_dim) / float(stride_dim))
    pad = ((output_dim - 1) * stride_dim + kernel_dim - in_dim) / 2
    return output_dim, pad


def valid_padding(in_dim, kernel_dim, stride_dim):
    output_dim = math.ceil(float(in_dim - kernel_dim + 1) / float(stride_dim))
    pad = 0

    return output_dim, pad

def get_deconv_padding(input_shape, output_shape, kernel_shape, stride):
    """
    input_shape: N,H,W,C
    output_shape: N,H,W,C
    kernel_shape: kh,kw
    stride: 1,sh,sw,sc
    """
    pady = 0
    padx = 0
    # 2Xpad = stride X (input - 1) + kernel - out
    pady = stride[1] * (input_shape[1] - 1) + kernel_shape[0] - output_shape[1]
    padx = stride[2] * (input_shape[2] - 1) + kernel_shape[1] - output_shape[2]
    if (pady % 2 == 1):
        pady = -1
    else:
        pady = pady / 2
    if (padx % 2 == 1):
        padx = -1
    else:
        padx = padx / 2
    return int(pady), int(padx)


def get_input(name, fail=True):
    global placeholder_dict
    global const_dict
    global concat_tracker
    global reshape_tracker
    global identity_tracker

    if len(concat_tracker) != 0:
        # We track all previous concats, as we may have a many-to-many
        # connection
        for concat in concat_tracker:
            if concat[0] == name:
                # If the next layer will try to attach to the non-existant
                # concat intermediary node.
                ret = []
                for l in concat[1]:
                    a = get_input(l)
                    if isinstance(a[0], list):
                        for a1 in a[0]:
                            ret.append(a1)
                    else:
                        ret.append(a[0])
                return [ret]

    if len(reshape_tracker) != 0:
        for reshape in reshape_tracker:
            if reshape[0] == name:
                return get_input(reshape[1])

    if len(identity_tracker) != 0:
        for idn in identity_tracker:
            if idn[0] == name:
                if idn[1] == inputnode:
                    return None
                if get_input(idn[1], False) == 0:
                    return [idn[1]]
                return get_input(idn[1])

    if name == inputnode:
        return None
    if name in node_dict.keys():
        return [node_dict[name].unprocessed_name]
    if not fail:
        return 0
    if name in const_dict.keys():
        throw_error(
            ErrorTable.StageDetailsNotSupported,
            "Top Not Supported - Constants " +
            str(name))
    else:
        throw_error(
            ErrorTable.StageDetailsNotSupported,
            "Top Not Found " + str(name))


def get_padding_input(name):
    global padding_tracker

    for padding in padding_tracker:
        if padding[0] == name:
            return padding[1], padding[2]
    return None, None


def have_first_input(name):
    if name == inputnode:
        return True
    if len(identity_tracker) != 0:
        for idn in identity_tracker:
            if idn[0] == name:
                if idn[1] == inputnode:
                    return True
    return False


def strip_tensor_id(word):
    return re.sub(':\d+', '', word)


# Count how many times we need this as an input
def count_inputs(t):
    graph = tf.get_default_graph()
    count = 0
    for node in graph.get_operations():
        for a in node.inputs:
            if a.name == t:
                count = count + 1
    return count


def parse_tensor(arguments, myriad_conf, preprocess=True, debug=False):
    global const_dict
    global placeholder_dict
    global node_dict
    global concat_tracker
    global identity_tracker
    global reshape_tracker
    global padding_tracker
    global inputnode

    path = arguments.net_description
    image = arguments.image
    output_node_name = arguments.output_node_name
    input_node_name = arguments.input_node_name
    filename = arguments.outputs_name
    if input_node_name is not None:
        inputnode = input_node_name

    debug = False
    with tf.Session() as sess:
        filetype = path.split(".")[-1]
        if filetype == 'pb':
            graph_def = graph_pb2.GraphDef()
            with open(path, 'rb') as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        else:
            saver = tf.train.import_meta_graph(path, clear_devices=True)
            if saver is not None:
                saver.restore(sess, path[:path.rfind('.')])
        graph = tf.get_default_graph()

        preprocessor = None

        if preprocess:
            preprocessor = TFPreprocessor()
            preprocessor.preprocess(graph)

        inputTensor = graph.get_tensor_by_name(inputnode + ':0')
        if output_node_name is None:
            output_node_name = 'output'

        try:
            outputTensor = graph.get_tensor_by_name(output_node_name + ':0')
        except:
            throw_error(ErrorTable.NoOutputNode, output_node_name + ':0')

        shape = inputTensor.get_shape()
        if shape.dims is None:
            # If the network does not give an input shape, assume this
            # we will need to add this as a parameter
            shape = [1, 224, 224, 3]
            if arguments.input_size:
                shape = [1,
                         arguments.input_size[1],
                         arguments.input_size[0],
                         3]

        if isinstance(shape, tf.TensorShape):
            shape_list = shape.as_list()
            # Tensorflow can have None in the batch size field of the
            # input shape, if that is the case then set it to 1
            if None == shape_list[0]:
                shape_list[0] = 1
                shape = shape_list
                inputTensor.set_shape(shape)
            elif None in shape:
                throw_error(ErrorTable.TFNotEvaluated)

        hw = arguments.ma2480
        print('shape:', shape)
        if image is None or image == "Debug":
            input_data = np.random.uniform(0, 1, shape)
            if (hw):
                # assume data in ZYX
                if (len(shape)==4):
                    input_data = input_data.reshape(int(shape[0]), int(shape[3]), int(shape[1]), int(shape[2]))
                input_data_layout = StorageOrder.orderZYX
            else:
                # assume data in YXZ
                if (len(shape)==4):
                    input_data = input_data.reshape(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
                input_data_layout = StorageOrder.orderYXZ
            input_data = preprocess_img(input_data,
                raw_scale=arguments.raw_scale,
                mean=arguments.mean,
                channel_swap=None)
            if debug:
                print("Input image shape", shape)
        else:
            # input image has layout ZYX
            input_data = parse_img(image,
                                   [int(shape[0]),
                                    int(shape[3]),
                                    int(shape[1]),
                                    int(shape[2])],
                                   raw_scale=arguments.raw_scale,
                                   mean=arguments.mean,
                                   channel_swap=arguments.channel_swap)
            input_data_layout = StorageOrder.orderZYX

        if (input_data_layout == StorageOrder.orderZYX):
            # transpose to YXZ
            if (len(shape)==4):
                input_data = input_data.transpose([0, 2, 3, 1])
            input_data_layout = StorageOrder.orderYXZ
        feed_dict={inputnode + ':0': input_data}
        res = outputTensor.eval(feed_dict)

        network = Network("TensorFlow Network", input_data, arguments.ma2480)
        arguments.network = network

        prev_node = None
        prev_node_label = None
        cnt = 0
        inputfound = False
        for idx, node in enumerate(graph.get_operations()):
            if debug:
                print("       ", idx, node.type, node.name)
                for a in node.inputs:
                    print("           IN:", a.name)
                for a in node.outputs:
                    print("           OUT:", a.name)
            if not inputfound:
                if have_first_input(node.name):
                    inputfound = True
                    if debug:
                        print('Starting to process')
                continue

            # Each layer can have a placeholder for the batch size.
            for output_item in node.outputs:
                item_shape = output_item.shape.as_list()
                if len(item_shape) > 0 and \
                        item_shape[0] == None:
                    item_shape[0] = 1
                    output_item.set_shape(item_shape)

            if preprocessor:

                pattern_found, current_pattern = preprocessor.pattern_found(node)

                if pattern_found:
                    if current_pattern.get_type() == PatternType.Completed:
                        continue
                    else:
                        if current_pattern.get_type() == PatternType.LeakyReLU:
                            if debug:
                                print("LeakyRelu")

                            if len(current_pattern.get_input_shape()) == 4:
                                node.outputs[0].set_shape([current_pattern.get_input_shape()[0],
                                                           current_pattern.get_input_shape()[1],
                                                           current_pattern.get_input_shape()[2],
                                                           current_pattern.get_output_shape()[3]])
                            elif len(current_pattern.get_input_shape()) == 2:
                                node.outputs[0].set_shape([current_pattern.get_input_shape()[0],
                                                           current_pattern.get_input_shape()[1]])
                            else:
                                throw_error(ErrorTable.StageDetailsNotSupported, "Unsupported LeakyRelu Dimensions")

                            prev_node.postOp = StageType.leaky_relu
                            prev_node.post_param1 = current_pattern.get_param(0)
                            prev_node.post_definition = get_op_definition(prev_node.postOp)
                            prev_node.changeName(current_pattern.get_name())
                            prev_node_label = strip_tensor_id(current_pattern.get_prev_name())
                            node_dict[prev_node_label] = prev_node
                        else:
                            throw_error(ErrorTable.StageDetailsNotSupported,
                                        "Pattern not supported " + str(current_pattern.get_type().name))

            if node.type == "Const":
                const_dict[node.name] = node.outputs[0].get_shape()
            elif node.type == "Placeholder":
                placeholder_dict[node.name] = node.outputs[0].get_shape()
            elif node.type == "PlaceholderWithDefault":
                const_dict[node.name] = node.outputs[0].get_shape()
            elif node.type == 'Variable' or node.type == 'VariableV2':
                variable_dict[node.name] = node.outputs[0].get_shape()
            elif node.type == "Conv2D":
                if debug:
                    print("Conv2D")

                inputs = node.inputs[0]
                input_shape = inputs.get_shape()
                # If the network does not have predetermined input shape, take
                # if from input
                if input_shape.dims is None and inputs.name == input_node_name + ':0':
                    input_shape = input_data.shape
                taps = node.inputs[1]
                taps_shape = node.inputs[1].get_shape()
                outputs = node.outputs[0].get_shape()

                ksize = taps_shape[0]
                stride = node.get_attr("strides")

                output_size = [input_shape[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[1]),
                                             int(ksize),
                                             stride[1])[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[2]),
                                             int(ksize),
                                             stride[2])[0],
                               outputs[3]]

                if debug:
                    print(output_size)

                node.outputs[0].set_shape(output_size)

                top, padding = get_padding_input(strip_tensor_id(inputs.name))
                padx = 0
                pady = 0
                padstyle = PadStyle.tfsame if node.get_attr(
                    "padding") == b'SAME' else PadStyle.tfvalid
                if top is not None:
                    if top == inputnode:
                        top = None
                    else:
                        top = [top]
                    pady = padding[1][0]
                    padx = padding[2][0]
                    padstyle = PadStyle.caffe
                    input_shape = [
                        input_shape[0],
                        input_shape[1] - 2 * pady,
                        input_shape[2] - 2 * padx,
                        input_shape[3]]
                else:
                    top = get_input(strip_tensor_id(inputs.name))

                xyz = (int(input_shape[2]),
                       int(input_shape[1]),
                       int(input_shape[3]))

                conv_dilation = 1
                layer_params = np.array([conv_dilation], dtype=np.dtype("<i4"))
                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         pady,
                                         padx,
                                         padstyle,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.convolution,
                                         int(taps_shape[0]),
                                         int(taps_shape[1]),
                                         stride[1],
                                         stride[2],
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         int(taps_shape[0]),
                                         int(taps_shape[1]),
                                         int(taps_shape[3]),
                                         np.array(taps.eval()),
                                         TapsOrder.orderHWCK,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments, 
                                         opParams=layer_params)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                # node_dict
                cnt += 1
            elif node.type == 'DepthwiseConv2dNative':
                if debug:
                    print("DepthwiseConv2dNative")

                inputs = node.inputs[0]
                input_shape = inputs.get_shape()
                # If the network does not have predetermined input shape, take
                # if from input
                if input_shape.dims is None and inputs.name == input_node_name + ':0':
                    input_shape = input_data.shape
                taps = node.inputs[1]
                taps_shape = node.inputs[1].get_shape()

                outputs = node.outputs[0].get_shape()

                ksize = taps_shape[0]
                stride = node.get_attr("strides")

                output_size = [input_shape[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[1]),
                                             int(ksize),
                                             stride[1])[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[2]),
                                             int(ksize),
                                             stride[2])[0],
                               outputs[3]]
                if debug:
                    print(output_size)

                node.outputs[0].set_shape(output_size)

                top, padding = get_padding_input(strip_tensor_id(inputs.name))
                padx = 0
                pady = 0
                padstyle = PadStyle.tfsame if node.get_attr(
                    "padding") == b'SAME' else PadStyle.tfvalid
                if top is not None:
                    if top == inputnode:
                        top = None
                    else:
                        top = [top]
                    pady = padding[1][0]
                    padx = padding[2][0]
                    padstyle = PadStyle.caffe
                    input_shape = [
                        input_shape[0],
                        input_shape[1] - 2 * pady,
                        input_shape[2] - 2 * padx,
                        input_shape[3]]
                else:
                    top = get_input(strip_tensor_id(inputs.name))

                xyz = (int(input_shape[2]),
                       int(input_shape[1]),
                       int(input_shape[3]))

                taps2 = np.array(taps.eval())
                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         pady,
                                         padx,
                                         padstyle,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.depthwise_convolution,
                                         int(taps_shape[0]),
                                         int(taps_shape[1]),
                                         stride[1],
                                         stride[2],
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         int(taps_shape[0]),
                                         int(taps_shape[1]),
                                         int(taps_shape[2]) * int(taps_shape[3]),
                                         taps2,
                                         TapsOrder.orderHWCK,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                # node_dict
                cnt += 1
            elif node.type == "Conv2DBackpropInput":
                inputs = node.inputs[2]
                input_shape = inputs.get_shape()
                # If the network does not have predetermined input shape, take
                # if from input
                if input_shape.dims is None and inputs.name == input_node_name + ':0':
                    input_shape = input_data.shape
                taps = node.inputs[1]
                taps_shape = node.inputs[1].get_shape().as_list()
                outputs = node.outputs[0].get_shape()

                ksize = [taps_shape[0], taps_shape[1]]
                stride = node.get_attr("strides")
                output_size = node.inputs[0].eval(feed_dict)

                node.outputs[0].set_shape(output_size)

                top, padding = get_padding_input(strip_tensor_id(inputs.name))
                pady, padx = get_deconv_padding(input_shape.as_list(), output_size, ksize, stride)
                if pady < 0 or padx < 0:
                    throw_error(ErrorTable.StageDetailsNotSupported, "Wrong deconvolution output shape.")

                padstyle = PadStyle.caffe
                if top is not None:
                    if top == inputnode:
                        top = None
                    else:
                        top = [top]
                    pady = padding[1][0]
                    padx = padding[2][0]
                    input_shape = [
                        input_shape[0],
                        input_shape[1] - 2 * pady,
                        input_shape[2] - 2 * padx,
                        input_shape[3]]
                else:
                    top = get_input(strip_tensor_id(inputs.name))

                xyz = (int(input_shape[2]),
                       int(input_shape[1]),
                       int(input_shape[3]))
                tapval = taps.eval()
                tapval = np.swapaxes(tapval, 2, 3)
                tapval = tapval[::-1,::-1,:,:]
                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         pady,
                                         padx,
                                         padstyle,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.deconvolution,
                                         int(taps_shape[0]),
                                         int(taps_shape[1]),
                                         stride[1],
                                         stride[2],
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         int(taps_shape[0]),
                                         int(taps_shape[1]),
                                         int(taps_shape[2]),
                                         np.array(tapval),
                                         TapsOrder.orderHWCK,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                # node_dict
                cnt += 1
            elif node.type == "BiasAdd":
                if debug:
                    print("BiasAdd")
                inputs = node.inputs[0].get_shape()
                bias_data = node.inputs[1]
                outputs = node.outputs[0].get_shape()

                if(len(inputs) == 4):
                    node.outputs[0].set_shape(
                        [inputs[0], inputs[1], inputs[2], outputs[3]])
                elif(len(inputs) == 2):
                    node.outputs[0].set_shape([inputs[0], inputs[1]])
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported Bias Dimensions")
                prev_node.addBias(
                    np.array(
                        bias_data.eval()).astype(
                        np.float16))
                prev_node.changeName(node.name)

                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

            elif node.type == "MaxPool":
                if debug:
                    print("MaxPool")

                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()
                outputs = node.outputs[0].get_shape()
                ksize = node.get_attr("ksize")
                stride = node.get_attr("strides")
                pad = 0

                output_size = [input_shape[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[1]),
                                             int(ksize[1]),
                                             stride[1])[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[2]),
                                             int(ksize[2]),
                                             stride[2])[0],
                               outputs[3]]

                node.outputs[0].set_shape(output_size)

                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))

                prev_node = NetworkStage(
                    node.name,
                    top,
                    StorageOrder.orderYXZ,
                    0,
                    0,
                    PadStyle.tfsame if node.get_attr("padding") == b'SAME' else PadStyle.tfvalid,
                    DataType.fp16,
                    DataType.fp16,
                    StageType.max_pooling,
                    ksize[1],
                    ksize[2],
                    stride[1],
                    stride[2],
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    ksize[1],
                    ksize[2],
                    int(output_size[3]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    myriad_config=myriad_conf,
                    args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1
            elif node.type == "Relu":
                if debug:
                    print("ReLU")

                inputs = node.inputs[0].get_shape()
                outputs = node.outputs[0].get_shape()
                if(len(inputs) == 4):
                    node.outputs[0].set_shape(
                        [inputs[0], inputs[1], inputs[2], outputs[3]])
                elif(len(inputs) == 2):
                    node.outputs[0].set_shape([inputs[0], inputs[1]])
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported ReLU Dimensions")

                prev_node.postOp = StageType.relu
                prev_node.post_definition = get_op_definition(prev_node.postOp)
                prev_node.changeName(node.name)

                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

            elif node.type == "Relu6":
                if debug:
                    print("ReLU6")

                inputs = node.inputs[0].get_shape()
                outputs = node.outputs[0].get_shape()
                if(len(inputs) == 4):
                    node.outputs[0].set_shape(
                        [inputs[0], inputs[1], inputs[2], outputs[3]])
                elif(len(inputs) == 2):
                    node.outputs[0].set_shape([inputs[0], inputs[1]])
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported ReLU Dimensions")

                prev_node.postOp = StageType.relu_x
                prev_node.post_param1 = 6.0
                prev_node.post_definition = get_op_definition(prev_node.postOp)
                prev_node.changeName(node.name)

                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

            elif node.type == "LRN":
                if debug:
                    print("LRN")
                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()

                outputs = node.outputs[0].get_shape()
                node.outputs[0].set_shape(
                    [input_shape[0], input_shape[1], input_shape[2], outputs[3]])

                top = get_input(strip_tensor_id(inputs.name))
                xyz = (int(input_shape[2]),
                       int(input_shape[1]),
                       int(input_shape[3]))
                bias = np.array([node.get_attr("bias"),
                                 node.get_attr("alpha") * (2 * node.get_attr("depth_radius") + 1),
                                 node.get_attr("beta"),
                                 0], dtype=np.float16)
                prev_node = NetworkStage(
                    node.name,
                    top,
                    StorageOrder.orderYXZ,
                    0,
                    0,
                    PadStyle.none,
                    DataType.fp16,
                    DataType.fp16,
                    StageType.LRN,
                    0,
                    2 * node.get_attr("depth_radius") + 1,
                    1,
                    1,
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    0,
                    0,
                    xyz[2],
                    None,
                    None,
                    bias,
                    None,
                    None,
                    None,
                    0,
                    0,
                    myriad_config=myriad_conf,
                    args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node
                cnt += 1

            elif node.type == "MatMul":  # Note: Assuming MatMul and FCL are the same.
                if debug:
                    print("FCL / MatMul")
                inputs = node.inputs
                input_shape = node.inputs[0].get_shape()
                taps = node.inputs[1]
                taps_shape = node.inputs[1].get_shape()

                outputs = node.outputs[0].get_shape()
                node.outputs[0].set_shape([node.inputs[0].get_shape()[0],
                                           node.inputs[1].get_shape()[1]])

                top = get_input(strip_tensor_id(inputs[0].name))
                xyz = (1, 1, int(input_shape[1]))

                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.none,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.fully_connected_layer,
                                         1,
                                         1,
                                         1,
                                         1,
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         1,
                                         1,
                                         int(taps_shape[1]),
                                         np.array(taps.eval()).astype(np.float16),
                                         TapsOrder.orderHWCK,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1

            elif node.type == "Softmax" or node.type == "Sigmoid" or node.type == "Tanh":
                if debug:
                    print(node.type)

                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()

                outputs = node.outputs[0].get_shape()
                if(len(input_shape) == 4):
                    node.outputs[0].set_shape(
                        [input_shape[0], input_shape[1], input_shape[2], outputs[3]])
                elif(len(input_shape) == 2):
                    node.outputs[0].set_shape([input_shape[0], input_shape[1]])
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported " + node.type + " dimensions")

                taps_shape = [1, 1, 1, 1]
                stride = [1, 1, 1, 1]
                pad = 0
                opParams=None
                if node.type == "Softmax":
                    stagetype = StageType.soft_max
                    opParams = np.array([1], dtype=np.dtype("<i4"))  # softmax would be performed on C - axis
                elif node.type == "Sigmoid":
                    stagetype = StageType.sigmoid
                else:
                    stagetype = StageType.tanh
                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))

                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.none,
                                         DataType.fp16,
                                         DataType.fp16,
                                         stagetype,
                                         int(taps_shape[0]),
                                         int(taps_shape[0]),
                                         stride[1],
                                         stride[2],
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         int(taps_shape[0]),
                                         int(taps_shape[0]),
                                         int(input_shape[1]),
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments,
                                         opParams=opParams)

                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1

            elif node.type == "AvgPool":
                if debug:
                    print("Avg Pool")
                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()
                outputs = node.outputs[0].get_shape()
                ksize = node.get_attr("ksize")
                stride = node.get_attr("strides")
                pad = 0

                output_size = [input_shape[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[1]),
                                             int(ksize[1]),
                                             stride[1])[0],
                               apply_padding(node.get_attr("padding"),
                                             int(input_shape[2]),
                                             int(ksize[2]),
                                             stride[2])[0],
                               outputs[3]]

                node.outputs[0].set_shape(output_size)

                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))

                prev_node = NetworkStage(
                    node.name,
                    top,
                    StorageOrder.orderYXZ,
                    0,
                    0,
                    PadStyle.tfsame if node.get_attr("padding") == b'SAME' else PadStyle.tfvalid,
                    DataType.fp16,
                    DataType.fp16,
                    StageType.average_pooling,
                    ksize[1],
                    ksize[2],
                    stride[1],
                    stride[2],
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    ksize[1],
                    ksize[2],
                    int(output_size[3]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    myriad_config=myriad_conf,
                    args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1

            elif node.type == "Mean":
                if debug:
                    print("Mean")
                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()
                dimensions = node.inputs[1].eval(feed_dict)
                if dimensions[0] != 1 or dimensions[1] != 2:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported Mean operation")
                outputs = node.outputs[0].get_shape()
                ksize = [0, int(input_shape[1]), int(input_shape[2]), 0]
                stride = [1, 1, 1, 1]

                output_size = [input_shape[0], 1, 1, outputs[3]]

                node.outputs[0].set_shape(output_size)

                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))

                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.tfvalid,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.average_pooling,
                                         ksize[1],
                                         ksize[2],
                                         stride[1],
                                         stride[2],
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         ksize[1],
                                         ksize[2],
                                         int(output_size[3]),
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1

            elif node.type == "Reshape":
                if debug:
                    print("Reshape")
                inputs = node.inputs
                input_shape = node.inputs[0].get_shape()
                desired_shape = node.inputs[1].eval(feed_dict)
 
                # Check for -1 in desired_shape
                if -1 in desired_shape:
                     if desired_shape[desired_shape == -1].size > 1:
                         throw(
                             ErrorTable.StageDetailsNotSupported,
                             "Illegal Reshape dimension")
#                    desired_shape = np.reshape(input_shape.as_list(), desired_shape)[0]
                     input_size = input_shape.num_elements()
                     desired_size = np.product(desired_shape[desired_shape >= 0])
                     negative_index = np.argmin(desired_shape)
                     desired_shape[negative_index] = input_size / desired_size
                    
                node.outputs[0].set_shape(desired_shape)
                reshape_tracker += [(strip_tensor_id(node.outputs[0].name),
                                     strip_tensor_id(inputs[0].name))]

            elif node.type == "Shape":
                if debug:
                    print(node.type, len(node.inputs[0].get_shape()))
            elif node.type == "Squeeze":
                if debug:
                    print("Squeeze")
                identity_tracker += [(strip_tensor_id(node.outputs[0].name),
                                      strip_tensor_id(node.inputs[0].name))]
            elif node.type == "Identity":

                if debug:
                    print("Identity")
                inputs = node.inputs
                input_shape = node.inputs[0].get_shape()

                node.outputs[0].set_shape(node.inputs[0].get_shape())

                identity_tracker += [(strip_tensor_id(node.outputs[0].name),
                                      strip_tensor_id(inputs[0].name))]

            elif node.type == "NoOp":
                if debug:
                    print("No OP")
                pass
            elif (node.type == "Concat" or node.type == "ConcatV2") and not node.inputs[0].dtype.is_integer:
                if debug:
                    print("Concat")
                concat_channel_size = 0
                inputs = node.inputs
                for src in inputs:
                    if len(src.get_shape()) >= 4:
                        concat_channel_size += int(src.get_shape()[3])
                a_input = node.inputs[1].get_shape()
                node.outputs[0].set_shape(
                    [a_input[0], a_input[1], a_input[2], concat_channel_size])

                rep_arr = []
                if node.type == 'Concat':
                    for inp in inputs[1:]:
                        rep_arr.append(strip_tensor_id(inp.name))
                else:
                    for inp in inputs[:-1]:
                        rep_arr.append(strip_tensor_id(inp.name))
                concat_tracker += [(strip_tensor_id(node.outputs[0].name), rep_arr)]
            elif ((node.type == 'Add' or node.type == 'Mul' or node.type == 'Maximum') and
                    get_input(strip_tensor_id(node.inputs[0].name), False) != 0 and
                    get_input(strip_tensor_id(node.inputs[1].name), False) != 0 and
                    (prev_node is None or
                     (prev_node.op != StageType.convolution and
                     prev_node.op != StageType.fully_connected_layer and
                     prev_node.op != StageType.depthwise_convolution) or
                     prev_node.bias is not None)):
                # Elementwise operations of the outputs of two existing nodes
                # This version of add should not be used
                # for addition of bias to conv, and fc nodes
                if debug:
                    print(node.type)
                top = [get_input(strip_tensor_id(node.inputs[0].name))[0],
                       get_input(strip_tensor_id(node.inputs[1].name))[0]]
                input_shape = node.inputs[0].get_shape()
                outputs = node.outputs[0].get_shape()
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))
                if node.type == 'Add':
                    op = StageType.eltwise_sum
                elif node.type == 'Mul' and node.inputs[1].shape[-1] == 1:
                    op = StageType.scale_with_scalar
                elif node.type == 'Mul':
                    op = StageType.eltwise_prod
                else:
                    op = StageType.eltwise_max
                prev_node = NetworkStage(
                    node.name,
                    top,
                    StorageOrder.orderYXZ,
                    0,
                    0,
                    PadStyle.none,
                    DataType.fp16,
                    DataType.fp16,
                    op,
                    1,
                    1,
                    1,
                    1,
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    myriad_config=myriad_conf,
                    args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node
                cnt += 1
            elif (node.type == 'Mul' or node.type == 'Div' or node.type == 'RealDiv') and prev_node_label is not None \
                    and (node.inputs[0].name == prev_node_label + ":0"
                         or node.inputs[1].name == prev_node_label + ":0"):
                # We are probably multiplying with a constant, try
                iidx = 1 if node.inputs[1].name == prev_node_label + ":0" else 0
                # Check if absorption is possible into a convolution, to be possible, we should use
                # the convolution output only once, here
                if (prev_node.op == StageType.convolution or prev_node.op == StageType.depthwise_convolution) \
                        and count_inputs(prev_node_label + ":0") == 1:
                    if debug:
                        print('Mul with constant absorbed into convolution')

                    taps_shape = prev_node.taps.shape

                    # Rearrange taps for fusing into depthwise convolution
                    if prev_node.op == StageType.depthwise_convolution:
                        prev_node.taps = np.reshape(prev_node.taps, [taps_shape[0], taps_shape[1],
                                                                     taps_shape[2] * taps_shape[3]])

                    if node.type == 'Mul':
                        prev_node.taps = np.multiply(
                            prev_node.taps, node.inputs[1 - iidx].eval(feed_dict))  # Eval may fail
                        if prev_node.bias is not None:
                            prev_node.bias = np.multiply(
                                prev_node.bias, node.inputs[1 - iidx].eval(feed_dict))  # Eval may fail
                    else:
                        prev_node.taps = np.divide(
                            prev_node.taps, node.inputs[1 - iidx].eval(feed_dict))  # Eval may fail
                        if prev_node.bias is not None:
                            prev_node.bias = np.divide(
                                prev_node.bias, node.inputs[1 - iidx].eval(feed_dict))  # Eval may fail

                    # Restore the original shape of taps after fusing into depthwise convolution
                    if prev_node.op == StageType.depthwise_convolution:
                        prev_node.taps = np.reshape(prev_node.taps, [taps_shape[0], taps_shape[1],
                                                                     taps_shape[2], taps_shape[3]])

                    node_dict[node.name] = node_dict[prev_node_label]
                    prev_node_label = node.name
                    prev_node.name = prev_node.unprocessed_name + '/' + node.name
                    prev_node.changeName(prev_node.name)
                    prev_node.alias.append(prev_node.unprocessed_name)
                    prev_node.alias.append(node.name)
                else:
                    if debug:
                        print('Mul/Div/RealDiv with constant')
                    inputs = node.inputs[iidx]
                    input_shape = node.inputs[iidx].get_shape()
                    top = get_input(strip_tensor_id(inputs.name))
                    if len(input_shape) == 4:
                        xyz = (int(input_shape[2]),
                               int(input_shape[1]),
                               int(input_shape[3]))
                    else:
                        xyz = (1, 1, int(input_shape[1]))

                    constant_value = node.inputs[1 - iidx].eval(feed_dict)
                    if node.type != 'Mul':
                        # For divide, use the reciprocal
                        if  constant_value != 0.0:
                            constant_value = 1 / constant_value
                        else:
                            throw_error(ErrorTable.ArgumentErrorRequired,
                                        "Divide by zero")

                    prev_node = NetworkStage(node.name,
                                             top,
                                             StorageOrder.orderYXZ,
                                             0,
                                             0,
                                             PadStyle.none,
                                             DataType.fp16,
                                             DataType.fp16,
                                             StageType.scale,
                                             0,
                                             0,
                                             1,
                                             1,
                                             xyz[0],
                                             xyz[1],
                                             xyz[2],
                                             0,
                                             0,
                                             xyz[2],
                                             constant_value,
                                             TapsOrder.orderHWCK,
                                             None,
                                             None,
                                             None,
                                             None,
                                             0,
                                             0,
                                             myriad_config=myriad_conf,
                                             args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1

            elif (node.type == 'Add' or node.type == 'Sub') and prev_node_label is not None \
                    and node.inputs[0].name == prev_node_label + ":0":
                # We are probaby adding a constant bias, try
                if debug:
                    print('Add (bias)')
                inputs = node.inputs[0].get_shape()
                bias_data = None
                bias_data = node.inputs[1].eval(feed_dict)  # This eval may fail
                outputs = node.outputs[0].get_shape()
                if(len(inputs) == 4):
                    node.outputs[0].set_shape(
                        [inputs[0], inputs[1], inputs[2], outputs[3]])
                elif(len(inputs) == 2):
                    node.outputs[0].set_shape([inputs[0], inputs[1]])
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported Bias Dimensions")

                # Add scalar
                if bias_data.ndim == 1 and bias_data.shape[0] == 1:
                    # Populate bias array with data
                    value = bias_data[0]
                    bias_data = np.empty([int(outputs[3])])
                    bias_data.fill(value)

                if node.type == 'Add':
                    prev_node.addBias(np.array(bias_data).astype(np.float16))
                else:
                    prev_node.addBias(np.array(-bias_data).astype(np.float16))

                prev_node.changeName(node.name)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node
            elif node.type == "Maximum":
                if debug:
                    print(node.type)
                if prev_node_label == None:
                    iidx = 0
                else:
                    iidx = 1 if node.inputs[1].name == prev_node_label + ":0" else 0
                inputs = node.inputs[iidx]
                input_shape = inputs.get_shape()
                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))
                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.none,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.max_with_const,
                                         0,
                                         0,
                                         1,
                                         1,
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         0,
                                         0,
                                         xyz[2],
                                         node.inputs[1 - iidx].eval(feed_dict),
                                         TapsOrder.orderHWCK,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node
                cnt += 1
            elif (node.type == "Square" or node.type == "Rsqrt") and \
                    (get_input(strip_tensor_id(node.inputs[0].name), False) != 0):
                if debug:
                    print(node.type)

                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()

                outputs = node.outputs[0].get_shape()
                if(len(input_shape) == 4):
                    node.outputs[0].set_shape(
                        [input_shape[0], input_shape[1], input_shape[2], outputs[3]])
                elif(len(input_shape) == 2):
                    node.outputs[0].set_shape([input_shape[0], input_shape[1]])
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported " + node.type + " dimensions")

                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))
                
                if node.type == "Square":
                    op_type = StageType.square
                else:
                    op_type = StageType.rsqrt
                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.none,
                                         DataType.fp16,
                                         DataType.fp16,
                                         op_type,
                                         0,
                                         0,
                                         1,
                                         1,
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         0,
                                         0,
                                         int(input_shape[1]),
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments)

                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1
            elif node.type == "Sum":
                if debug:
                    print(node.type)

                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()

                output_shape = node.outputs[0].get_shape()
                axis = node.inputs[1].eval(feed_dict)
                if axis != len(output_shape) - 1:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Unsupported " + node.type + " axis")
                axis_param = np.array([axis, 0], dtype=np.float16)

                top = get_input(strip_tensor_id(inputs.name))
                if len(input_shape) == 4:
                    xyz = (int(input_shape[2]),
                           int(input_shape[1]),
                           int(input_shape[3]))
                else:
                    xyz = (1, 1, int(input_shape[1]))

                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.none,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.sum_reduce,
                                         0,
                                         0,
                                         1,
                                         1,
                                         xyz[0],
                                         xyz[1],
                                         xyz[2],
                                         0,
                                         0,
                                         1,
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         None,
                                         0,
                                         0,
                                         myriad_config=myriad_conf,
                                         args=arguments,
                                         opParams=axis_param)

                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node

                cnt += 1
            elif (node.type == 'FusedBatchNorm'  and prev_node_label is not None and
                 len(node.inputs) == 5):
                if debug:
                    print('FusedBatchNorm')

                # Fold the batchnorm into the weights
                if prev_node.op == StageType.convolution or \
                   prev_node.op == StageType.fully_connected_layer or \
                   prev_node.op == StageType.depthwise_convolution:
                    if debug:
                        print('FusedBatchNorm absorbed into convolution')

                    eps = node.get_attr('epsilon')
                    scale_param = node.inputs[1].eval()
                    offset = node.inputs[2].eval()
                    mean = node.inputs[3].eval()
                    var = node.inputs[4].eval()
                    if len(mean) == 0 or len(var) == 0:
                        throw_error(
                            ErrorTable.StageDetailsNotSupported,
                            "FusedBatchNorm inputs mean and variance are not defined.  The graph is not created for inference.")

                    variance = var + eps
                    scale = np.reciprocal(np.sqrt(variance)) * scale_param
                    bias = offset - (mean * scale)
                    scale = np.reshape(scale, [1, 1, 1, -1])
                    bias = np.reshape(bias, [1, 1, 1, -1])
                    if prev_node.op == StageType.depthwise_convolution:
                       scale = np.swapaxes(scale,2,3)
                       bias = np.swapaxes(bias,2,3)
                    prev_node.taps = prev_node.taps * scale
                    
                    if prev_node.bias is not None:
                        if bias is not None:
                            prev_node.bias = prev_node.bias * scale + bias
                        else:
                            prev_node.bias = prev_node.bias * scale
                    else:
                        if bias is not None:
                            prev_node.addBias(np.array(bias).astype(np.float16))

                    node_dict[node.name] = node_dict[prev_node_label]
                    prev_node_label = node.name
                    prev_node.name = prev_node.unprocessed_name + '/' + node.name
                    prev_node.changeName(prev_node.name)
                    prev_node.alias.append(prev_node.unprocessed_name)
                    prev_node.alias.append(node.name)
                else:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "FusedBatchNorm must be preceded by convolution or fully connected layer")
                    
            elif node.type == 'Slice' and not node.inputs[0].dtype.is_integer:
                if debug:
                    print('Slice')
                input_shape = node.inputs[0].get_shape()
                slicingbegin = node.inputs[1].eval()
                slicingsize = node.inputs[2].eval()
                if (len(input_shape) != 4 or len(slicingbegin) != 4 or len(slicingsize) != 4 or
                        slicingbegin[0] != 0 or slicingbegin[1] != 0 or slicingbegin[2] != 0 or
                        input_shape[0] != slicingsize[0] or input_shape[1] != slicingsize[1] or input_shape[2] != slicingsize[2]):
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        "Slice type not supported")
                top = get_input(strip_tensor_id(node.inputs[0].name))
                curslicing = []
                curslicing.append(
                    (top, int(
                        slicingbegin[3]), int(
                        slicingbegin[3] + slicingsize[3])))
                prev_node = NetworkStage(node.name,
                                         top,
                                         StorageOrder.orderYXZ,
                                         0,
                                         0,
                                         PadStyle.none,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.copy,
                                         1,
                                         1,
                                         1,
                                         1,
                                         int(input_shape[2]),
                                         int(input_shape[1]),
                                         int(input_shape[3]),
                                         1,
                                         1,
                                         slicingsize[3],
                                         None,
                                         TapsOrder.orderKCHW,
                                         None,
                                         None,
                                         StageType.none,
                                         None,
                                         0,
                                         0,
                                         curslicing,
                                         myriad_conf,
                                         args=arguments)
                network.attach(prev_node)
                prev_node_label = strip_tensor_id(node.outputs[0].name)
                node_dict[prev_node_label] = prev_node
                cnt += 1

            elif node.type == 'ExtractImagePatches':
                if debug:
                    print("ExtractImagePatches")

                # Currently not supported, will be interpreted as reorg for yolo-v2
                throw_error(ErrorTable.StageDetailsNotSupported, node.type)

                '''
                inputs = node.inputs[0]
                input_shape = node.inputs[0].get_shape()
                stride = node.get_attr("stride")

                output_shape = [input_shape[0], tf.Dimension(int(input_shape[1]) / stride),
                                tf.Dimension(int(input_shape[2]) / stride),
                                tf.Dimension(int(input_shape[3]) * stride * stride)]

                node.outputs[0].set_shape(output_shape)

                top = get_input(strip_tensor_id(inputs.name))

                op_node = NetworkStage(
                    node.name + '_op',
                    top,
                    StorageOrder.orderYXZ,
                    0,
                    0,
                    PadStyle.none,
                    DataType.fp16,
                    DataType.fp16,
                    StageType.reorg,
                    0,
                    0,
                    0,
                    0,
                    int(input_shape[2]),
                    int(input_shape[1]),
                    int(input_shape[3]),
                    int(output_shape[1]),
                    int(output_shape[2]),
                    int(output_shape[3]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    myriad_config=myriad_conf,
                    args=arguments,
                    opParams=np.array([stride], dtype=np.int32))
                network.attach(op_node)
                prev_node_label = node.name + '_op'
                node_dict[prev_node_label] = prev_node

                cnt += 1

                prev_node = NetworkStage(
                    node.name,
                    [node.name + '_op'],
                    StorageOrder.orderYXZ,
                    0,
                    0,
                    PadStyle.none,
                    DataType.fp16,
                    DataType.fp16,
                    StageType.copy,
                    1,
                    1,
                    1,
                    1,
                    int(output_shape[2]),
                    int(output_shape[1]),
                    int(output_shape[3]),
                    int(output_shape[1]),
                    int(output_shape[2]),
                    int(output_shape[3]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    myriad_config=myriad_conf,
                    args=arguments)
                network.attach(prev_node)
                prev_node_label = node.name
                node_dict[prev_node_label] = prev_node

                cnt += 1
                '''

            elif node.type == 'Pad':
                if debug:
                    print('Pad')
                padding_tracker += [(strip_tensor_id(node.outputs[0].name),
                                     strip_tensor_id(node.inputs[0].name),
                                     node.inputs[1].eval(feed_dict))]
            elif (node.type == 'TruncatedNormal' or node.type == 'Assign' or
                  node.type == 'RandomUniform' or node.type == 'Div' or node.type == 'RealDiv' or node.type == 'Mul' or
                  node.type == 'Floor' or node.type == 'Add' or node.type == 'Sub' or
                  node.type == 'Rsqrt' or node.type == 'RandomStandardNormal' or
                  node.type == 'L2Loss' or node.type == 'Pack' or node.type == 'Slice' or
                  node.type == 'Prod' or node.type == 'ExpandDims' or node.type == 'ConcatV2' or
                  node.type == 'StridedSlice' or node.type == 'Fill'):
                pass
            else:
                throw_error(ErrorTable.StageDetailsNotSupported, node.type)
            if node.name == output_node_name:
                if node.type == 'Concat' or node.type == 'ConcatV2':
                    nodes = network.search_several(get_input(node.name)[0])
                    NetworkStage.concat(nodes)
                break

        print('res.shape: ', res.shape)
        # reduce/extend to 3 axes
        if len(res.shape) == 4:
            # discard the batch axis
            assert res.shape[0] == 1, "output tensor's batch size must be 1 !"
            res = res[0]
            # res = res.reshape(res.shape[0], res.shape[1], res.shape[2])
        elif len(res.shape) == 3:
            res = res.reshape(res.shape[0], res.shape[1], res.shape[2])
        elif len(res.shape) == 2:
            res = res.reshape(1, res.shape[0], res.shape[1])
        else:
            res = res.reshape(1, 1, res.shape[0])

    network.outputTensorShape   = res.shape
    network.outputTensorLayout  = StorageOrder.orderYXZ

    print('TensorFlow output shape: ', res.shape)
    # for inference, batch is always 1
    # discard the batch axis
    expected        = res
    expected_shape  = res.shape
    expected_layout = StorageOrder.orderYXZ

    parse_ret = {
            'network': network,
            'expected': expected,
            'expected_shape': expected_shape,
            'expected_layout': expected_layout
    }

    return parse_ret
