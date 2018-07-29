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

from mvnctools.Models.EnumDeclarations import *
from mvnctools.Controllers.EnumController import *

from mvnctools.Models.StageDefinitions.AveragePooling import *
from mvnctools.Models.StageDefinitions.Bias import *
from mvnctools.Models.StageDefinitions.StorageOrderConvert import *
from mvnctools.Models.StageDefinitions.Convolution import *
from mvnctools.Models.StageDefinitions.Copy import *
from mvnctools.Models.StageDefinitions.Crop import *
from mvnctools.Models.StageDefinitions.Deconv import *
from mvnctools.Models.StageDefinitions.DepthConv import *
from mvnctools.Models.StageDefinitions.Eltwise import *
from mvnctools.Models.StageDefinitions.Elu import *
from mvnctools.Models.StageDefinitions.FCL import *
from mvnctools.Models.StageDefinitions.MyriadXHardwareLayer import *
from mvnctools.Models.StageDefinitions.InnerLRN import *
from mvnctools.Models.StageDefinitions.LRN import *
from mvnctools.Models.StageDefinitions.MaxConst import *
from mvnctools.Models.StageDefinitions.MaxPooling import *
from mvnctools.Models.StageDefinitions.NoOp import *
from mvnctools.Models.StageDefinitions.Prelu import *
from mvnctools.Models.StageDefinitions.Power import *
from mvnctools.Models.StageDefinitions.Relu import *
from mvnctools.Models.StageDefinitions.Reshape import *
from mvnctools.Models.StageDefinitions.Rsqrt import *
from mvnctools.Models.StageDefinitions.Scale import *
from mvnctools.Models.StageDefinitions.ScaleScalar import *
from mvnctools.Models.StageDefinitions.Softmax import *
from mvnctools.Models.StageDefinitions.Square import *
from mvnctools.Models.StageDefinitions.Sigmoid import *
from mvnctools.Models.StageDefinitions.SumReduce import *
from mvnctools.Models.StageDefinitions.TanH import *
from mvnctools.Models.StageDefinitions.ToPlaneMajor import *
from mvnctools.Models.StageDefinitions.Permute import *
from mvnctools.Models.StageDefinitions.Normalize import *
from mvnctools.Models.StageDefinitions.PriorBox import *
from mvnctools.Models.StageDefinitions.DetectionOutput import *
from mvnctools.Models.StageDefinitions.Relu_Op import ReluOp
from mvnctools.Models.StageDefinitions.Elu_Op import EluOp

def get_op_definition(op_type, force_op=False):
    """
    Get the global definition of an operation
    :param op_type: the operation to lookup
    :return: definition object
    """

    op_mapping = {
        StageType.average_pooling:                  AveragePooling(),
        StageType.bias:                             Bias(),
        StageType.storage_order_convert:            StorageOrderConvert(),
        StageType.convolution:                      Convolution(),
        StageType.copy:                             Copy(),
        StageType.crop:                             Crop(),
        StageType.deconvolution:                    Deconv(),
        StageType.depthwise_convolution:            DepthConv(),
        StageType.eltwise_prod:                     Eltwise(),
        StageType.eltwise_sum:                      Eltwise(),
        StageType.eltwise_max:                      Eltwise(),
        StageType.elu:                              Elu(),
        StageType.fully_connected_layer:            FCL(),
        StageType.innerlrn:                         InnerLRN(),
        StageType.LRN:                              LRN(),
        StageType.max_pooling:                      MaxPooling(),
        StageType.max_with_const:                   MaxConst(),
        StageType.myriadX_convolution:              MyriadXHardwareLayer(),
        StageType.myriadX_fully_connected_layer:    MyriadXHardwareLayer(),
        StageType.myriadX_pooling:                  MyriadXHardwareLayer(),
        StageType.none:                             NoOp(),
        StageType.power:                            Power(),
        StageType.prelu:                            Prelu(),
        StageType.relu:                             Relu(),
        StageType.relu_x:                           Relu(),
        StageType.leaky_relu:                       Relu(),
        StageType.reshape:                          Reshape(),
        StageType.rsqrt:                            Rsqrt(),
        StageType.scale:                            Scale(),
        StageType.scale_with_scalar:                ScaleScalar(),
        StageType.sigmoid:                          Sigmoid(),
        StageType.soft_max:                         Softmax(),
        StageType.square:                           Square(),
        StageType.sum_reduce:                       SumReduce(),
        StageType.tanh:                             TanH(),
        StageType.toplanemajor:                     ToPlaneMajor(),
        StageType.permute:                          Permute(),
        StageType.permute_flatten:                  PermuteFlatten(),
        StageType.normalize:                        Normalize(),
        StageType.prior_box:                        PriorBox(),
        StageType.detection_output:                 DetectionOutput(),
    }

    if force_op:
        # In the new parser, we use in-place Operations, rather than postOperations.
        op_mapping[StageType.relu] = ReluOp()
        op_mapping[StageType.relu_x] = ReluOp()
        op_mapping[StageType.leaky_relu] = ReluOp()
        op_mapping[StageType.elu] = EluOp()

    val = op_mapping.get(op_type, -1)



    # print(op_type)

    if val == -1:
        throw_error(ErrorTable.StageTypeNotSupported, op_type)

    if val == -1 and op_type != StageType.none:
        print("WARNING TYPE NOT PRESENT IN DEFINITION LIBRARY: ", op_type)
    else:
        return val
