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
import math
from mvnctools.Models.NetworkStage import *
from mvnctools.Models.EnumDeclarations import *

from mvnctools.Controllers.HwDescriptorSerializer import SerializedDescriptor, HwDescOp
from mvnctools.Models.StageDefinitions.OpManager import get_op_definition
import mvnctools.Controllers.Globals as GLOBALS
from mvnctools.Controllers.PingPong import getManualHwSchedule, get_null_terminating_name
from mvnctools.Controllers.Soh import heightSolution, heightSolutionWithPooling

def round_up(x, mult):
    return ((x + mult -1) // mult) * mult

def getPadding(inputDimX, inputDimY, outputDimX, outputDimY, kernelDimX, kernelDimY, kernelStride):
    valid_out_x = math.ceil((inputDimX - kernelDimX + 1) / kernelStride)
    valid_out_y = math.ceil((inputDimY - kernelDimY + 1) / kernelStride)
    # same_out_x = math.ceil(inputDimX / kernelStride)
    # same_out_y = math.ceil(inputDimY / kernelStride)
    pad_along_x = (outputDimX - 1) * kernelStride + kernelDimX - inputDimX
    pad_along_y = (outputDimY - 1) * kernelStride + kernelDimY - inputDimY
    pad_left = pad_along_x // 2
    pad_right = pad_along_x - pad_left
    pad_top = pad_along_y // 2
    pad_bottom = pad_along_y - pad_top
    return (outputDimX != valid_out_x or outputDimY != valid_out_y, pad_left, pad_right, pad_top, pad_bottom)

def canFusePooling(input, output, kernel):
    # input == (inputDimX, inputDimY, inputDimZ)
    # output == (outputDimX, outputDimY, outputDimZ)
    # kernel == (kernelDimX, kernelDimY)

    # If we have split over input depth, then fused pooling
    # can be used (it is mathematically wrong)
    if manualSplit(input, output, kernel) > 1:
        return False

    return True

def manualSplit(input, output, kernel):
    # input == (inputDimX, inputDimY, inputDimZ)
    # output == (outputDimX, outputDimY, outputDimZ)
    # kernel == (kernelDimX, kernelDimY)

    # VGG conv1_2
    if input == (224, 224, 64) and \
       output == (224, 224, 64) and \
       kernel == (3, 3):
       return 2

    # VGG conv2_2
    if input == (112, 112, 128) and \
       output == (112, 112, 128) and \
       kernel == (3, 3):
       return 2

    # VGG conv3_2, conv3_3
    if input == (56, 56, 256) and \
       output == (56, 56, 256) and \
       kernel == (3, 3):
       return 2

    # VGG conv4_2, conv4_3
    if input == (28, 28, 512) and \
       output == (28, 28, 512) and \
       kernel == (3, 3):
       return 2

    # VGG conv5_2, conv5_3
    if input == (14, 14, 512) and \
       output == (14, 14, 512) and \
       kernel == (3, 3):
       return 2

    # VGG fc6
    if input == (7, 7, 512) and \
       output == (1, 1, 4096) and \
       kernel == (7, 7):
       return 16

    # Yolo-tiny conv7
    if input == (7, 7, 512) and \
       output == (7, 7, 1024) and \
       kernel == (3, 3):
       return 2

    # Yolo-tiny conv8
    if input == (7, 7, 1024) and \
       output == (7, 7, 256) and \
       kernel == (3, 3):
       return 4

    # Yolo-tiny fc9 (casted as conv)
    if input == (7, 7, 256) and \
       output == (1, 1, 640) and \
       kernel == (7, 7):
       return 4

    # Googlenet SSD conv6_2_mbox_conf
    if input == (6, 6, 512) and \
       output ==  (6, 6, 126) and \
       kernel == (3, 3):
       return 2

    # Googlenet SSD conv6_2_mbox_loc
    if input == (6, 6, 512) and \
       output ==  (6, 6, 24) and \
       kernel == (3, 3):
       return 2

    # Googlenet SSD inception_4e/output_mbox_conf
    if input == (12, 12, 832) and \
       output ==  (12, 12, 126) and \
       kernel == (3, 3):
       return 2

    # Googlenet SSD inception_4e/output_mbox_loc
    if input == (12, 12, 832) and \
       output ==  (12, 12, 24) and \
       kernel == (3, 3):
       return 2

    # Googlenet SSD inception_3b/output_norm_mbox_conf
    if input == (24, 24, 480) and \
       output ==  (24, 24, 126) and \
       kernel == (3, 3):
       return 2

    # Googlenet SSD inception_3b/output_norm_mbox_loc
    if input == (24, 24, 480) and \
       output ==  (24, 24, 24) and \
       kernel == (3, 3):
       return 2

    # Default: no split
    return 1

def trueRowAdaptationB0(maxOutputLines, pad_left, pad_right, pad_top, pad_bottom, inputDimX, kernelHeight):

    if pad_bottom:
        max_size = 2048
    elif (pad_left or pad_right): 
        max_size = 4096
    else:
        max_size = round_up(inputDimX, 16) * (maxOutputLines-1 + kernelHeight-1)
    maxOutputLines = max_size//round_up(inputDimX, 16) -kernelHeight + 2

    return maxOutputLines

def hardwareize(stage, scheduler = None):

    stageName = get_null_terminating_name(stage.name)
    pingPongPair = getManualHwSchedule()

    if pingPongPair[stageName] == False:
        return None, None, None, None, None, None, None, None     # Fallback to SW

    CMX_SIZE = pingPongPair.cmxForStreaming(stageName)

    bytesPerPixel = 2
    # stage.is_hardware = True
    solution = None

    pingPongDir, pingPongFmt, rud = pingPongPair[stageName]

    if stage.op == StageType.convolution:
        # For now, we can only compute the splits by trial and error:
        if GLOBALS.OPT_SCHEDULER:
            splits = scheduler.getSplits(stage)
        else:
            splits = manualSplit((stage.inputDimX, stage.inputDimY, stage.inputDimZ),
                (stage.outputDimX, stage.outputDimY, stage.outputDimZ),
                (stage.radixX, stage.radixY))

        # print(">>>>>>>> >>>>>>>>> >>>>>>>>", splits)
        # print((stage.inputDimX, stage.inputDimY, stage.inputDimZ),
        #     (stage.outputDimX, stage.outputDimY, stage.outputDimZ),
        #     (stage.radixX, stage.radixY))

        origInputDimZ = stage.inputDimZ
        completeHwDescList = []

        reluOnShaveAccumulation = False
        reluNegSlope = 0
        reluPosSlope = 1
        solution = None

        assert(stage.inputDimZ % splits == 0)

        if GLOBALS.OPT_SCHEDULER:
            solution = scheduler.getSolution(stage)
            solution[2].sort(key = lambda tup: tup[1])
            stage.inputDimZ = stage.inputDimZ // splits

            padEnable, pad_left, pad_right, pad_top, pad_bottom = getPadding(stage.inputDimX, stage.inputDimY, \
                                                                        stage.outputDimX, stage.outputDimY, \
                                                                        stage.radixX, stage.radixY, stage.strideX)

        else:
            stage.inputDimZ = stage.inputDimZ // splits

            padEnable, pad_left, pad_right, pad_top, pad_bottom = getPadding(stage.inputDimX, stage.inputDimY, \
                                                                        stage.outputDimX, stage.outputDimY, \
                                                                        stage.radixX, stage.radixY, stage.strideX)

            solution = splitConvolution(stage.inputDimX, stage.inputDimY, stage.inputDimZ, \
                                    stage.outputDimX, stage.outputDimY, stage.outputDimZ, \
                                    stage.radixX, stage.radixY, stage.strideX, \
                                    padEnable, "FP16", "FP16")

        for sodIndex in range(splits):
            newInputDimZ, newOutputDimZ, tiles = solution
            #Calculate in/out strides
            inputStrideX = bytesPerPixel
            inputStrideY = ((stage.inputDimX * inputStrideX + 15) // 16) * 16
            inputStrideZ = inputStrideY * stage.inputDimY
            outputStrideX = bytesPerPixel
            outputStrideY = ((stage.outputDimX * outputStrideX + 15) // 16) * 16
            outputStrideZ = outputStrideY * stage.outputDimY
            inputBufferSize = newInputDimZ * inputStrideZ
            outputBufferSize = newOutputDimZ * outputStrideZ
            tapsBufferSize = newInputDimZ * newOutputDimZ * stage.radixX * stage.radixY * bytesPerPixel
            biasBufferSize = newOutputDimZ * bytesPerPixel if stage.bias is not None else 0

            heightSplits = []
            if pingPongPair.isStreamed(stageName):
                if pingPongPair.isStreamedAndSplit(stageName):
                    maxOutputChannelsInDescr = max(tiles, key=lambda x: x[0])[0]
                    bytesPerFullDepthSlice = bytesPerPixel * maxOutputChannelsInDescr * round_up(stage.outputDimX, 8)
                    maxOutputLines = CMX_SIZE // bytesPerFullDepthSlice
                else:
                    maxOutputLines = stage.outputDimY
            else:
                maxOutputLines = stage.outputDimY

            # print(maxOutputLines)

            try:
                if stage.fused_conv_pooling:
                    # For conv3x3s1p1 and fused 2x2s2 pooling, we need 4 extra lines
                    # Also, for this specific case, the maxOutputLines is doubled, because the output
                    # is reduced by a factor of 2
                    heightSplits = heightSolutionWithPooling(stage.inputDimY, stage.radixY, stage.strideY, stage.padY, maxOutputLines)
            except AttributeError:
                # For convolution without fused pooling
                # The following is not correct for convolution. We cannot have selective zero padding
                # pad = (stage.radixY // 2 if pad_top > 0 else 0, stage.radixY // 2 if pad_bottom > 0 else 0)
                assert(stage.padX == stage.padY)
                assert(stage.padY == 0 or stage.padY == (stage.radixY // 2))
                pad = (stage.padY, stage.padY)
                heightSplits = heightSolution(stage.inputDimY, stage.radixY, stage.strideY, pad, maxOutputLines)

            # trueInputNeeded, outputWithJunk, junkBefore, junkAfter, inputStartIndex, inputEndIndex, outputStartIndex, outputEndIndex
            # print(heightSplits)

            sohGroups = []
            for sohIndex, heightSplitSol in enumerate(heightSplits):

                trueInputNeeded, _, junkBefore, junkAfter, inputStartIndex, _, outputStartIndex, _ = heightSplitSol
                outChanOffset = 0

                sohGroup = HwDescriptorList()

                # print(tiles)
                # quit()

                for tileIndex, (outChans, mode) in enumerate(tiles):
                    if tileIndex == (len(tiles) - 1) and \
                        sohIndex == len(heightSplits) - 1 and \
                        sodIndex == splits - 1:
                        lastTile = True
                    else:
                        lastTile = False

                    conv = HwDescriptor(0, 0, 1, 0, mode, tileIndex, lastTile, stageName)
                    conv.topOutputJunk = junkBefore
                    conv.bottomOutputJunk = junkAfter

                    if round_up(stage.inputDimX, 8) * stage.inputDimY * stage.inputDimZ * 2 < 128*1024:
                        if tileIndex == 0:
                            conv.reuseData = rud
                        else:
                            if tiles[tileIndex-1][1] != mode:
                                conv.reuseData = 0
                            else:
                                conv.reuseData = 1

                    if pingPongFmt[0] == 'I':
                        conv.setupInterleaved(True)

                        split_by_input_sz = sodIndex * newInputDimZ * inputStrideY
                        split_by_height_sz = origInputDimZ * inputStrideY * inputStartIndex
                        offset = split_by_input_sz + split_by_height_sz
                        conv.setupInput(offset, stage.inputDimX, trueInputNeeded, newInputDimZ, stage.inputDimY, origInputDimZ)
                    else:
                        split_by_input_sz = sodIndex * newInputDimZ * inputStrideY * stage.inputDimY
                        split_by_height_sz = inputStrideY * inputStartIndex
                        offset = split_by_input_sz + split_by_height_sz
                        conv.setupInput(offset, stage.inputDimX, trueInputNeeded, newInputDimZ, stage.inputDimY, origInputDimZ)

                    totalOutChans = sum([outChans for outChans, _ in tiles])

                    if pingPongFmt[1] == 'I':
                        # Interleaved output
                        try:
                            conv.setupOutput(outputStartIndex * stage.totalOutChans * outputStrideY + outChanOffset * outputStrideY, stage.outputDimX, stage.outputDimY, outChans, stage.outputDimY, stage.totalOutChans)
                        except AttributeError:
                            conv.setupOutput(outputStartIndex * totalOutChans * outputStrideY + outChanOffset * outputStrideY, stage.outputDimX, stage.outputDimY, outChans, stage.outputDimY, totalOutChans)
                    else:
                        conv.setupOutput(bytesPerPixel*outputStartIndex*round_up(stage.outputDimX, 8) + outChanOffset * outputStrideZ, stage.outputDimX, stage.outputDimY, outChans, stage.outputDimY, totalOutChans)

                    taps_processed = sum([a for a, _ in tiles[:tileIndex]])
                    # taps_total = sum([a for a, _ in tiles])
                    taps_total = newOutputDimZ

                    split_by_input_taps_sz = bytesPerPixel * taps_total * newInputDimZ * stage.radixX * stage.radixY
                    conv.setupConvolutionCoefficients(0,
                            sodIndex * split_by_input_taps_sz +
                            bytesPerPixel * taps_processed * newInputDimZ * stage.radixX * stage.radixY,
                            stage.radixX, stage.radixY, stage.strideX)
                    if padEnable:
                        conv.setupPadding(0)

                    # if sohIndex == 0 and (stage.bias is not None):
                    if sodIndex == 0 and stage.bias is not None:
                        conv.setupBias(outChanOffset * bytesPerPixel)

                    try:
                        if stage.fused_conv_pooling:
                            # If we have fused pooling and more that one depth groups, then
                            # we need not to fuse the pooling
                            assert (sodIndex == 0)
                            # print("Fusing HW convolution with pooling")
                            # print(stage.fused_op)
                            if stage.fused_op == StageType.max_pooling:
                                poolType = 0
                            else:
                                # Not supported for now
                                assert(False)
                                poolType = 1

                            # conv.setupPooling(poolType, stage.radixX, stage.radixY, stage.strideX)
                            conv.setupConvolutionPooling(poolType, stage.fused_poolingRadixX, stage.fused_poolingRadixY)
                    except AttributeError:
                        pass

                    # Since we run this in hardware, we need to convert postops (and whatever else) that are
                    # supported by the hardware
                    if stage.postOp in [StageType.relu, StageType.leaky_relu, StageType.prelu]:
                        # General form of hardware Parametrized ReLU (PRelu):
                        #                     / a0*x, if x < t0
                        # f(x; t0, a0, a1) = |
                        #                     \ a1*x, if x >= t0

                        if splits > 1:
                            reluOnShaveAccumulation = True
                            reluPosSlope = 1
                            reluNegSlope = stage.post_param1
                        else:
                            if stage.post_param1 == 0.0:
                                conv.setupRelu(0, 0, 1)
                            else:
                                # Check that the scaling with be 10 bits signed integer
                                assert(abs(int(round(1 / stage.post_param1))) < 2 ** 9)
                                scale = np.full((newOutputDimZ), stage.post_param1).astype(np.float16)
                                conv.setupRelu(0, 1, int(round(1 / stage.post_param1)))
                                conv.setupScale(outChanOffset * bytesPerPixel)
                                # we only should do this once
                                if sohIndex == 0:
                                    stage.putScale(scale)


                    outChanOffset += outChans
                    conv.sohGroup = sohIndex
                    conv.sodGroup = sodIndex
                    sohGroup.pushDescriptor(conv)

                sohGroups.append(sohGroup)
            completeHwDescList.append(sohGroups)

        # Remove the post-op
        if stage.postOp == StageType.bias:
            stage.postOp = StageType.none
        elif stage.postOp in [StageType.relu, StageType.leaky_relu, StageType.prelu] and \
                        splits > 1:
            stage.postOp = StageType.none

        # Transpose the descriptors so that we have a list of SoH,
        # where each element will be a SoD sublist.
        transposedDescrList = list(map(list, zip(*completeHwDescList)))

        # Flatten the descriptors
        finalHwDescList = HwDescriptorList()

        for soh in transposedDescrList:
            for sod in soh:
                for item in sod.descList:
                    finalHwDescList.pushDescriptor(item)

        taps = taps_hwck_to_cnnhw(stage.taps, newInputDimZ * splits, newOutputDimZ)

        if splits > 1:
            taps = regroup_taps(taps, splits)

        concatOffset = 0
        try:
            concatOffset = stage.concatBufferOffset
        except AttributeError:
            pass

        finalOutputBufferSize = outputBufferSize
        try:
            finalOutputBufferSize = stage.concatOutputSize
        except AttributeError:
            pass

        stage.op = StageType.myriadX_convolution
        solution = ((inputStrideX, inputStrideY, inputStrideZ), (outputStrideX, outputStrideY, outputStrideZ), \
                    (newInputDimZ, newOutputDimZ, concatOffset, reluOnShaveAccumulation, reluNegSlope, reluPosSlope), \
                    (inputBufferSize, finalOutputBufferSize, tapsBufferSize, biasBufferSize), finalHwDescList, taps, stage.bias, stage.scale)

        stage.inputDimZ = origInputDimZ

    elif stage.op == StageType.max_pooling or stage.op == StageType.average_pooling:

        splits = 1 # which means no splits
        completeHwDescList = []
        solution = None

        padEnable, pad_left, pad_right, pad_top, pad_bottom = getPadding(stage.inputDimX, stage.inputDimY, \
                                                                        stage.outputDimX, stage.outputDimY, \
                                                                        stage.radixX, stage.radixY, stage.strideX)
        solution = splitPooling(stage.inputDimZ)

        for sodIndex in range(splits):
            newInputDimZ, newOutputDimZ, tiles = solution
            #Calculate in/out strides
            inputStrideX = bytesPerPixel
            inputStrideY = ((stage.inputDimX * inputStrideX + 15) // 16) * 16
            inputStrideZ = inputStrideY * stage.inputDimY
            outputStrideX = bytesPerPixel
            outputStrideY = ((stage.outputDimX * outputStrideX + 15) // 16) * 16
            outputStrideZ = outputStrideY * stage.outputDimY
            inputBufferSize = newInputDimZ * inputStrideZ
            outputBufferSize = newOutputDimZ * outputStrideZ
            tapsBufferSize = 0
            biasBufferSize = 0

            heightSplits = []
            if pingPongPair.isStreamed(stageName):
                if pingPongPair.isStreamedAndSplit(stageName):
                    maxOutputChannelsInDescr = max(tiles, key=lambda x: x[0])[0]
                    bytesPerFullDepthSlice = bytesPerPixel * maxOutputChannelsInDescr * round_up(stage.outputDimX, 8)
                    maxOutputLines = CMX_SIZE // bytesPerFullDepthSlice
                else:
                    maxOutputLines = stage.outputDimY
            else:
                maxOutputLines = stage.outputDimY

            # print(maxOutputLines)
            if padEnable == 1:
                maxOutputLines = trueRowAdaptationB0(maxOutputLines, pad_left, pad_right, pad_top, pad_bottom, stage.inputDimX, stage.radixY)

            pad = (stage.radixY // 2 if pad_top > 0 else 0, stage.radixY // 2 if pad_bottom > 0 else 0)
            heightSplits = heightSolution(stage.inputDimY, stage.radixY, stage.strideY, pad, maxOutputLines)

            # trueInputNeeded, outputWithJunk, junkBefore, junkAfter, inputStartIndex, inputEndIndex, outputStartIndex, outputEndIndex
            # print(heightSplits)

            sohGroups = []
            for sohIndex, heightSplitSol in enumerate(heightSplits):

                trueInputNeeded, _, junkBefore, junkAfter, inputStartIndex, _, outputStartIndex, _ = heightSplitSol
                outChanOffset = 0

                sohGroup = HwDescriptorList()

                for tileIndex, (outChans, mode) in enumerate(tiles):
                    if tileIndex == (len(tiles) - 1) and \
                        sohIndex == len(heightSplits) - 1 and \
                        sodIndex == splits - 1:
                        lastTile = True
                    else:
                        lastTile = False

                    pool = HwDescriptor(0, 0, 1, 0, mode, tileIndex, lastTile, stageName)
                    pool.topOutputJunk = junkBefore
                    pool.bottomOutputJunk = junkAfter

                    chansSoFar = sum([outChans for outChans, _ in tiles[0:tileIndex]])
                    totalOutChans = sum([outChans for outChans, _ in tiles])


                    if pingPongFmt[0] == 'I':
                        pool.setupInterleaved(True)

                        split_by_input_sz = chansSoFar * inputStrideY
                        split_by_height_sz = stage.inputDimZ * inputStrideY * inputStartIndex
                        offset = split_by_input_sz + split_by_height_sz
                        pool.setupInput(offset, stage.inputDimX, trueInputNeeded, outChans, stage.inputDimY, stage.inputDimZ)
                    else:
                        split_by_input_sz = chansSoFar * inputStrideY * stage.inputDimY
                        split_by_height_sz = inputStrideY * inputStartIndex
                        offset = split_by_input_sz + split_by_height_sz
                        pool.setupInput(offset, stage.inputDimX, trueInputNeeded, outChans, stage.inputDimY, stage.inputDimZ)

                    if pingPongFmt[1] == 'I':
                        # Interleaved output
                        try:
                            pool.setupOutput(outputStartIndex * stage.totalOutChans * outputStrideY + outChanOffset * outputStrideY, stage.outputDimX, stage.outputDimY, outChans, stage.outputDimY, stage.totalOutChans)
                        except AttributeError:
                            pool.setupOutput(outputStartIndex * totalOutChans * outputStrideY + outChanOffset * outputStrideY, stage.outputDimX, stage.outputDimY, outChans, stage.outputDimY, totalOutChans)
                    else:
                        pool.setupOutput(bytesPerPixel*outputStartIndex*round_up(stage.outputDimX, 8) + outChanOffset * outputStrideZ, stage.outputDimX, stage.outputDimY, outChans, stage.outputDimY, totalOutChans)

                    if stage.op == StageType.max_pooling:
                        poolType = 0
                    else:
                        poolType = 1
                    pool.setupPooling(poolType, stage.radixX, stage.radixY, stage.strideX)
                    padType = 0

                    if padEnable:
                        # Repeat padding is not used in CNNs at all.
                        if pad_left > 0:
                            padType |= 0x08
                        if pad_right > 0:
                            padType |= 0x01
                        if pad_top > 0:
                            padType |= 0x04
                        if pad_bottom > 0:
                            padType |= 0x02
                        pool.setupPadding(padType)

                    outChanOffset += outChans
                    pool.sohGroup = sohIndex
                    pool.sodGroup = sodIndex
                    sohGroup.pushDescriptor(pool)

                sohGroups.append(sohGroup)
            completeHwDescList.append(sohGroups)

        # Transpose the descriptors so that we have a list of SoH,
        # where each element will be a SoD sublist.
        transposedDescrList = list(map(list, zip(*completeHwDescList)))

        # Flatten the descriptors
        finalHwDescList = HwDescriptorList()

        for soh in transposedDescrList:
            for sod in soh:
                for item in sod.descList:
                    finalHwDescList.pushDescriptor(item)

        concatOffset = 0
        try:
            concatOffset = stage.concatBufferOffset
        except AttributeError:
            pass

        finalOutputBufferSize = outputBufferSize
        try:
            finalOutputBufferSize = stage.concatOutputSize
        except AttributeError:
            pass

        stage.op = StageType.myriadX_pooling
        solution = ((inputStrideX, inputStrideY, inputStrideZ), (outputStrideX, outputStrideY, outputStrideZ), \
                    (newInputDimZ, newOutputDimZ, concatOffset, False, 0.0, 1.0), \
                    (inputBufferSize, finalOutputBufferSize, tapsBufferSize, biasBufferSize), finalHwDescList, None, None, None)

    elif stage.op == StageType.fully_connected_layer:
        # Split in the best mode possible. However, if there are more than one tiles,
        #  we force mode 0 and split again, because of a hardware bug (which does not reset
        #  the accumulator)
        newInputDimZ, newOutputDimZ, tiles = splitFullyConnected(stage.inputDimZ, stage.outputDimZ)
        if len(tiles) > 1:
            newInputDimZ, newOutputDimZ, tiles = splitFullyConnected(stage.inputDimZ, stage.outputDimZ, modes=[0])

        #Calculate in/out strides
        inputStrideX = 0
        inputStrideY = 0
        inputStrideZ = 16
        outputStrideX = 0
        outputStrideY = 0
        outputStrideZ = 16
        inputBufferSize = newInputDimZ * inputStrideZ
        outputBufferSize = newOutputDimZ * outputStrideZ
        tapsBufferSize = newInputDimZ * newOutputDimZ * bytesPerPixel
        biasBufferSize = 0
        if stage.bias is not None:
            biasBufferSize = newOutputDimZ * bytesPerPixel
        hwDescList = HwDescriptorList()
        genTileIndex = 0
        outputOffset = 0
        for tileIndex, subTiles in enumerate(tiles):
            lastTile = tileIndex == len(tiles) - 1
            inputOffset = 0

            # Detect how many of the output channels the current subTiles
            # are generating is actual/real. This is needed for the early interrupts
            # workaround.
            outChansSoFar = sum([outChans[0][1] for outChans in tiles[0:tileIndex+1]])
            actualOutputChannels = stage.outputDimZ
            if outChansSoFar > actualOutputChannels:
                if tileIndex == 0:
                    descriptorOutputChannels = actualOutputChannels
                else:
                    runningOutputChans = sum([outChans[0][1] for outChans in tiles[0:tileIndex]])
                    descriptorOutputChannels = actualOutputChannels - runningOutputChans
            else:
                descriptorOutputChannels = subTiles[0][1]

            for subTileIndex, (maxW, maxN, mode) in enumerate(subTiles):

                lastSubTile = subTileIndex == len(subTiles) - 1
                genLastTile = lastTile and lastSubTile
                if lastSubTile:
                    accum = 0
                else:
                    accum = 1
                tapsOffset = (outputOffset * newInputDimZ) + (inputOffset * 8)
                fc = HwDescriptor(0, 0, 1, 0, mode, genTileIndex, genLastTile, stageName)
                fc.setupInputFullyConnected(inputOffset * inputStrideZ, accum, newInputDimZ, maxW)
                fc.setupOutputFullyConnected(outputOffset * outputStrideZ)
                fc.setupVectorsFullyConnected(0, tapsOffset * 2, maxN)

                # Store the actual output in the descriptor
                fc.actualOutChannels = descriptorOutputChannels

                if lastSubTile:
                    if stage.postOp in [StageType.relu, StageType.leaky_relu, StageType.prelu]:
                        fc.setupRelu(0, 0, 1)

                if stage.bias is not None:
                    fc.setupBias(outputOffset * bytesPerPixel)
                hwDescList.pushDescriptor(fc)
                genTileIndex += 1
                inputOffset += maxW
            outputOffset += maxN

        taps = taps_hwck_to_cnnhw(stage.taps, newInputDimZ, newOutputDimZ)

        concatOffset = 0
        try:
            concatOffset = stage.concatBufferOffset
        except AttributeError:
            pass

        finalOutputBufferSize = outputBufferSize
        try:
            finalOutputBufferSize = stage.concatOutputSize
        except AttributeError:
            pass

        stage.op = StageType.myriadX_fully_connected_layer
        solution = ((inputStrideX, inputStrideY, inputStrideZ), (outputStrideX, outputStrideY, outputStrideZ), \
                    (newInputDimZ, newOutputDimZ, concatOffset, False, 0.0, 1.0), \
                    (inputBufferSize, finalOutputBufferSize, tapsBufferSize, biasBufferSize), hwDescList, taps, stage.bias, stage.scale)


    stage.inputStrideX = inputStrideX
    stage.inputStrideY = inputStrideY
    stage.inputStrideZ = inputStrideZ
    stage.outputStrideX = outputStrideX
    stage.outputStrideY = outputStrideY
    stage.outputStrideZ = outputStrideZ

    if 'taps' in locals():  # TODO: Only do this if the hardware has weights. (and dont do it like this)
        # From XYZ to ZYX
        stage.tapDimX, stage.tapDimY, stage.tapDimZ = stage.tapDimZ, stage.tapDimY, stage.tapDimX
        stage.tapStrideX = 2
        stage.tapStrideY = stage.tapDimX * stage.tapStrideX
        stage.tapStrideZ = stage.tapStrideY * stage.tapDimY
        stage.tapLayout = StorageOrder.orderZYX

    stage.definition = get_op_definition(stage.op)

    # Calculate correct strides for myriadX NetworkStages
    if pingPongFmt[0] == 'I':
        stage.definition.changeLayoutRequirements("input", StorageOrder.orderYZX)
        stage.recalculate_stride("input", myriadX=True)
    else:
        stage.definition.changeLayoutRequirements("input", StorageOrder.orderZYX)
        stage.recalculate_stride("input", myriadX=True)


    if pingPongFmt[1] == 'I':
        stage.outputLayout = StorageOrder.orderYZX
        stage.recalculate_stride("output", myriadX=True)
    else:
        stage.outputLayout = StorageOrder.orderZYX
        stage.recalculate_stride("output", myriadX=True)


    return solution

class HwDescriptorList:
    def __init__(self):
        self.descList = []
    def setInputAddress(self, baseInputAddress):
        for desc in self.descList:
            desc.adjustInputAddress(baseInputAddress)
    def setOutputAddress(self, baseOutputAddress):
        for desc in self.descList:
            desc.adjustOutputAddress(baseOutputAddress)
    def setTapsAddress(self, baseTapsAddress):
        for desc in self.descList:
            desc.adjustTapsAddress(baseTapsAddress)
    def setBiasAddress(self, baseBiasAddress):
        for desc in self.descList:
            desc.adjustBiasAddress(baseBiasAddress)
    def setScaleAddress(self, baseScaleAddress):
        for desc in self.descList:
            desc.adjustScaleAddress(baseScaleAddress)
    def pushDescriptor(self, descriptor):
        self.descList.append(descriptor)
    def getContentSize(self):
        return 16 * 8 * len(self.descList)
    def getContent(self, baseAddress):
        content = []
        relocInstance = []
        relocWorkBuffer = []
        relocInBlob = []

        for desc in self.descList:
            (dc, dri, drwb, drib), lastDescAddress = desc.getContent(baseAddress)
            content.extend(dc)
            relocInstance.extend(dri)
            relocWorkBuffer.extend(drwb)
            relocInBlob.extend(drib)
        return content, relocInstance, relocWorkBuffer, relocInBlob, lastDescAddress


class HwDescriptor:
    def __init__(self, dataMode, id, disableInt, interruptTrigger, mode, tileIndex, lastTile, stageName):
        self.dataMode = dataMode
        self.opType = HwDescOp.convolution.value
        self.id = id
        self.disableInt = disableInt
        self.disableInt = 0
        self.interruptTrigger = interruptTrigger
        self.tileIndex = tileIndex
        self.interleavedInput = 0
        self.interleavedOutput = 0
        self.sodGroup = 0
        self.sohGroup = 0
        self.actualOutChannels = 0
        self.topOutputJunk = 0
        self.bottomOutputJunk = 0
        self.interleaved = False
        self.lastTile = lastTile
        self.inputDataAddr = 0
        self.inputDimX = 0
        self.inputDimY = 0
        self.inputDimZ = 0
        self.outputDataAddr = 0
        self.outputDimX = 0
        self.outputDimY = 0
        self.outputDimZ = 0
        self.coeffMode = 0
        self.coeffData = 0
        self.kerDimX = 0
        self.kerDimY = 0
        self.stride = 0
        self.poolEn = 0
        self.poolType = 0
        self.poolKerDimX = 0
        self.poolKerDimY = 0
        self.accumulate = 0
        self.totalDimX = 0
        self.vectorData = 0
        self.noOfVectors = 0
        self.padEn = 0
        self.padMode = 0
        self.reluEn = 0
        self.reluXEn = 0
        self.t0 = 0
        self.a0 = 0
        self.a1 = 0
        self.x = 0
        self.reuseData = 0
        self.biasAddress = 0
        self.scaleAddress = 0
        self.useBias = False
        self.useScale = False
        self.mode = mode
        self.type = 0
        self.stageName = stageName
    def adjustInputAddress(self, baseInputAddr):
        self.inputDataAddr += baseInputAddr
    def adjustOutputAddress(self, baseOutputAddr):
        self.outputDataAddr += baseOutputAddr
    def adjustTapsAddress(self, baseTapsAddr):
        self.coeffData += baseTapsAddr
    def adjustBiasAddress(self, baseBiasAddr):
        if self.useBias:
            self.biasAddress += baseBiasAddr
    def adjustScaleAddress(self, baseScaleAddr):
        if self.useScale:
            self.scaleAddress += baseScaleAddr
    def setupInput(self, inputDataAddr, inputDimX, inputDimY, inputDimZ, totalInputDimY, totalInputDimZ):
        self.type = 0
        self.inputDataAddr = inputDataAddr
        self.inputDimX = inputDimX
        self.inputDimY = inputDimY
        self.inputDimZ = inputDimZ
        self.totalInputDimY = totalInputDimY
        self.totalInputDimZ = totalInputDimZ
    def setupInterleaved(self, interleaved):
        self.interleaved = interleaved
    def setupOutput(self, outputDataAddr, outputDimX, outputDimY, outputDimZ, totalOutputDimY, totalOutputDimZ):
        self.type = 0
        self.outputDataAddr = outputDataAddr
        self.outputDimX = outputDimX
        self.outputDimY = outputDimY
        self.outputDimZ = outputDimZ
        self.totalOutputDimY = totalOutputDimY
        self.totalOutputDimZ = totalOutputDimZ
    def setupConvolutionCoefficients(self, mode, coeffData, kerDimX, kerDimY, stride):
        self.type = 0
        self.coeffMode = mode
        self.coeffData = coeffData
        self.kerDimX = kerDimX
        self.kerDimY = kerDimY
        self.stride = stride
        self.type = 0
    def setupConvolutionPooling(self, poolType, poolKerDimX, poolKerDimY):
        self.type = 0
        self.opType = HwDescOp.convolution_with_pooling.value
        self.poolType = poolType
        self.poolEn = 1
        self.poolKerDimX = poolKerDimX
        self.poolKerDimY = poolKerDimX
    def setupPooling(self, poolType, poolKerDimX, poolKerDimY, stride):
        self.type = 4
        self.poolType = poolType
        self.poolEn = 1
        self.poolKerDimX = poolKerDimX
        self.poolKerDimY = poolKerDimY
        self.stride = stride
    def setupInputFullyConnected(self, inputDataAddr, accumulate, totalDimX, inputDimX):
        self.type = 2
        self.inputDataAddr = inputDataAddr
        self.accumulate = accumulate
        self.totalDimX = totalDimX
        self.inputDimX = inputDimX
    def setupOutputFullyConnected(self, outputDataAddr):
        self.type = 2
        self.outputDataAddr = outputDataAddr
    def setupVectorsFullyConnected(self, coeffMode, vectorData, noOfVectors):
        self.type = 2
        self.coeffMode = coeffMode
        self.vectorData = vectorData
        self.noOfVectors = noOfVectors
    def setupPadding(self, padMode):
        self.padEn = 1
        self.padMode = padMode #0
    def setupRelu(self, t0, a0, a1):
        self.reluEn = 1
        self.t0 = t0
        self.a0 = a0
        self.a1 = a1
    def setupReluX(self, x):
        self.reluXEn = 1
        self.x = x
    def setupBias(self, bias):
        self.useBias = True
        self.biasAddress = bias
    def setupScale(self, scale):
        self.useScale = True
        self.scaleAddress = scale
    def getContent(self, baseAddress):
        baseAddress += self.tileIndex * 16 * 8
        if self.type == 0:
            return self.getContentForConvolution(baseAddress), baseAddress
        elif self.type == 4:
            return self.getContentForPooling(baseAddress), baseAddress
        elif self.type == 2:
            return self.getContentForFullyConnected(baseAddress), baseAddress
        return None
    def getContentForConvolution(self, baseAddress, debug=False):
        noOfBlocks = 1 << self.mode
        sizeOfBlock = (128 * 1024) >> self.mode
        bytesPerPixel = 1 << (1 - self.dataMode)
        pixelsPerCMXLine = 128 // (bytesPerPixel * 8)
        inDataLineStride = bytesPerPixel * self.inputDimX
        inDataLineStride = ((inDataLineStride + 15) // 16) * 16
        inDataChanStride = inDataLineStride * self.totalInputDimY

        pingPongPair = getManualHwSchedule()

        pingPongDir, pingPongFmt, _ = pingPongPair[self.stageName]

        if pingPongFmt[0] == 'I':
            # Assume interleaved input:
            inDataChanStride = inDataLineStride
            inDataLineStride = inDataLineStride * self.totalInputDimZ

        localLineStride = (self.inputDimX + (pixelsPerCMXLine - 1)) // pixelsPerCMXLine

        chanPerBlock = self.inputDimZ // noOfBlocks
        availableBytesPerChan = sizeOfBlock // chanPerBlock
        bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel
        linesPerChan = availableBytesPerChan // bytesPerLine
        if(linesPerChan > self.inputDimY):
            linesPerChan = self.inputDimY
        localChanStride = linesPerChan * localLineStride
        if(self.poolEn == 1):
            minLines = self.kerDimY + self.poolKerDimY
        else:
            minLines = min(self.kerDimY + 1, linesPerChan)
        coeffLPB = chanPerBlock * self.kerDimY * self.kerDimX
        coeffSetSize = self.kerDimX * self.kerDimY
        outDataLineStride = bytesPerPixel * self.outputDimX
        outDataLineStride = ((outDataLineStride + 15) // 16) * 16
        outDataChanStride = outDataLineStride * self.totalOutputDimY

        if pingPongFmt[1] == 'I':
            # Assume interleaved output:
            outDataChanStride = outDataLineStride
            outDataLineStride = outDataLineStride * self.totalOutputDimZ

        bytesPerCoeffSet = coeffSetSize
        coeffChStrideIn = bytesPerCoeffSet * 2 * 8
        coeffChStrideOut = coeffChStrideIn * self.inputDimZ
        content = []
        relocInstance = []
        relocWorkBuffer = [baseAddress + 8*4, baseAddress + 16*4]
        relocInBlob = [baseAddress + 12*4]

        if pingPongFmt[0] == 'I':
            self.interleavedInput = 1

        if pingPongFmt[1] == 'I':
            self.interleavedOutput = 1

        #Line 0
        nextDescAddr = 0
        if not self.lastTile:
            nextDescAddr = baseAddress + (16 * 8)
            relocInstance = [baseAddress + 0]

        if self.useBias:
            relocInBlob.append(baseAddress + 22*4)
        if self.useScale:
            relocInBlob.append(baseAddress + 23*4)

        chemicalX = 0
        if(self.reluXEn):
            chemicalX = self.x
        elif(self.poolEn and self.poolType == 1):
            chemicalX = np.float16(1 / (self.poolKerDimX * self.poolKerDimY)).view(np.uint16)


        # If not set, ensure fields default to 0, not -1
        if self.poolKerDimX == 0:
            self.poolKerDimX = 1
        if self.poolKerDimY == 0:
            self.poolKerDimY = 1

        assert(minLines <= linesPerChan)

        sd = SerializedDescriptor("Conv")

        # Line 0
        sd.set_field("NextDesc", nextDescAddr)
        sd.set_field("Type", self.opType)    # TODO:
        sd.set_field("mode", self.mode)
        sd.set_field("rsvd_00", self.interleavedInput + (self.interleavedOutput << 1))
        sd.set_field("id", self.id)
        sd.set_field("it", self.interruptTrigger)
        sd.set_field("cm", 0)
        sd.set_field("dm", 0)
        sd.set_field("disaint", self.disableInt)
        sd.set_field("rsvd_02", 0)

        # Line 1
        sd.set_field("iDimY-1", self.inputDimY - 1 )
        sd.set_field("rsvd_10", self.topOutputJunk )
        sd.set_field("iDimX-1", self.inputDimX - 1 )
        sd.set_field("rsvd_11", self.bottomOutputJunk)
        sd.set_field("iChans-1", self.inputDimZ - 1)
        sd.set_field("rsvd_12", 0)
        sd.set_field("oChans-1", self.outputDimZ - 1)
        sd.set_field("interleaved", self.interleaved)

        # Line 2
        sd.set_field("ChRamBlk-1", chanPerBlock-1)
        sd.set_field("stride", self.stride-1)
        sd.set_field("InFw-1", self.kerDimX-1)
        sd.set_field("InFh-1", self.kerDimY-1)
        sd.set_field("PadType", self.padMode)
        sd.set_field("PadEnable", self.padEn)

        # Line 3
        sd.set_field("poolEn", self.poolEn)
        sd.set_field("poolKernelHeight-1", self.poolKerDimX-1)
        sd.set_field("poolKernelWidth-1", self.poolKerDimY-1)
        sd.set_field("avgPoolX", chemicalX)
        sd.set_field("poolType", self.poolType)

        # Line 4
        sd.set_field("dataBaseAddr", self.inputDataAddr)
        sd.set_field("t0", self.t0)
        sd.set_field("a0", self.a0)
        sd.set_field("a1", self.a1)
        sd.set_field("reluXEn", self.reluXEn)
        sd.set_field("reluEn", self.reluEn)

        # Line 5
        sd.set_field("dataChStr", inDataChanStride)
        sd.set_field("dataLnStr", inDataLineStride)

        # Line 6
        sd.set_field("coeffBaseAddr", self.coeffData)
        sd.set_field("coeffChStrOut", coeffChStrideOut)

        # Line 7
        sd.set_field("coeffChStrIn", coeffChStrideIn)
        sd.set_field("outLnStr", outDataLineStride)

        # Line 8
        sd.set_field("outBaseAddr", self.outputDataAddr)
        sd.set_field("outChStr", outDataChanStride)

        # Line 9
        sd.set_field("localLs", localLineStride)
        sd.set_field("localCs", localChanStride)
        sd.set_field("linesPerCh-1", linesPerChan - 1)
        sd.set_field("rsvd_92", self.sodGroup)
        sd.set_field("rud", self.reuseData)

        # Line A
        sd.set_field("minLines-1", minLines - 1)
        sd.set_field("rsvd_A0", self.sohGroup)
        sd.set_field("coeffLpb-1", coeffLPB - 1)
        sd.set_field("css-1", coeffSetSize- 1)
        sd.set_field("outputX", self.outputDimX)

        # Line B
        sd.set_field("biasBaseAddr", self.biasAddress)
        sd.set_field("scaleBaseAddr", self.scaleAddress)

        sd.set_pallete(None)


        content = sd.serialize()

        return content, relocInstance, relocWorkBuffer, relocInBlob

    def getContentForPooling(self, baseAddress, debug=False):
        noOfBlocks = 1 << self.mode
        sizeOfBlock = (128 * 1024) >> self.mode
        bytesPerPixel = 1 << (1 - self.dataMode)
        pixelsPerCMXLine = 128 // (bytesPerPixel * 8)
        inDataLineStride = bytesPerPixel * self.inputDimX
        inDataLineStride = ((inDataLineStride + 15) // 16) * 16
        inDataChanStride = inDataLineStride * self.totalInputDimY

        pingPongPair = getManualHwSchedule()

        pingPongDir, pingPongFmt, _ = pingPongPair[self.stageName]

        if pingPongFmt[0] == 'I':
            # Assume interleaved input:
            inDataChanStride = inDataLineStride
            inDataLineStride = inDataLineStride * self.totalInputDimZ

        localLineStride = (self.inputDimX + (pixelsPerCMXLine - 1)) // pixelsPerCMXLine
        chanPerBlock = self.inputDimZ // noOfBlocks
        availableBytesPerChan = sizeOfBlock // chanPerBlock
        bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel
        linesPerChan = availableBytesPerChan // bytesPerLine
        if(linesPerChan > self.inputDimY):
            linesPerChan = self.inputDimY
        localChanStride = linesPerChan * localLineStride
        # if(self.poolEn == 1):
        #     minLines = self.kerDimY + self.poolKerDimY
        # else:
        #     minLines = self.kerDimY + 1

        if self.inputDimX / self.stride <= 4:
            minLines = self.poolKerDimY + 2*self.stride + 3
        else:
            minLines = self.kerDimY + 1
            minLines = self.poolKerDimY + self.stride + 2

        outDataLineStride = bytesPerPixel * self.outputDimX
        outDataLineStride = ((outDataLineStride + 15) // 16) * 16
        outDataChanStride = outDataLineStride * self.outputDimY

        if pingPongFmt[1] == 'I':
            # Assume interleaved output:
            outDataChanStride = outDataLineStride
            outDataLineStride = outDataLineStride * self.totalOutputDimZ

        content = []
        relocInstance = []
        relocWorkBuffer = [baseAddress + 8*4, baseAddress + 16*4]
        relocInBlob = [baseAddress + 12*4]

        if pingPongFmt[0] == 'I':
            self.interleavedInput = 1

        if pingPongFmt[1] == 'I':
            self.interleavedOutput = 1



        if self.useBias:
            relocInBlob.append(baseAddress + 22*4)
        if self.useScale:
            relocInBlob.append(baseAddress + 23*4)

        chemicalX = 0
        if(self.reluXEn):
            chemicalX = self.x
        elif(self.poolEn and self.poolType == 1):
            chemicalX = np.float16(1 / (self.poolKerDimX * self.poolKerDimY)).view(np.uint16)


        #Line 0
        nextDescAddr = 0
        if not self.lastTile:
            nextDescAddr = baseAddress + (16 * 8)
            relocInstance = [baseAddress + 0]

        # If not set, ensure fields default to 0, not -1
        if self.poolKerDimX == 0:
            self.poolKerDimX = 1
        if self.poolKerDimY == 0:
            self.poolKerDimY = 1
        if self.kerDimX == 0:
            self.kerDimX = 1
        if self.kerDimY == 0:
            self.kerDimY = 1

        sd = SerializedDescriptor("Pool")

        # Line 0
        sd.set_field("NextDesc", nextDescAddr)
        sd.set_field("Type", HwDescOp.pooling_only.value)
        sd.set_field("mode", self.mode)
        sd.set_field("rsvd_00", self.interleavedInput + (self.interleavedOutput << 1))
        sd.set_field("id", self.id)
        sd.set_field("it", self.interruptTrigger)
        sd.set_field("cm", 0)
        sd.set_field("dm", 0)
        sd.set_field("disaint", self.disableInt)
        sd.set_field("rsvd_02", 0)

        # Line 1
        sd.set_field("iDimY-1", self.inputDimY - 1 )
        sd.set_field("rsvd_10", self.topOutputJunk )
        sd.set_field("iDimX-1", self.inputDimX - 1 )
        sd.set_field("rsvd_11", self.bottomOutputJunk)
        sd.set_field("iChans-1", self.inputDimZ - 1)
        sd.set_field("oChans-1", self.outputDimZ - 1)
        sd.set_field("interleaved", self.interleaved)

        # Line 2
        sd.set_field("ChRamBlk-1", chanPerBlock-1)
        sd.set_field("stride", self.stride-1)
        sd.set_field("InFw-1", self.kerDimX-1)
        sd.set_field("InFh-1", self.kerDimY-1)
        sd.set_field("PadType", self.padMode)
        sd.set_field("PadEnable", self.padEn)

        # Line 3
        sd.set_field("poolEn", self.poolEn)
        sd.set_field("poolKernelHeight-1", self.poolKerDimX-1)
        sd.set_field("poolKernelWidth-1", self.poolKerDimY-1)
        sd.set_field("avgPoolX", chemicalX)
        sd.set_field("poolType", self.poolType)

        # Line 4
        sd.set_field("dataBaseAddr", self.inputDataAddr)
        sd.set_field("t0", self.t0)
        sd.set_field("a0", self.a0)
        sd.set_field("a1", self.a1)
        sd.set_field("reluXEn", self.reluXEn)
        sd.set_field("reluEn", self.reluEn)

        # Line 5
        sd.set_field("dataChStr", inDataChanStride)
        sd.set_field("dataLnStr", inDataLineStride)

        # Line 6
        # Nothing needed Here

        # Line 7
        sd.set_field("outLnStr", outDataLineStride)

        # Line 8
        sd.set_field("outBaseAddr", self.outputDataAddr)
        sd.set_field("outChStr", outDataChanStride)

        # Line 9
        sd.set_field("localLs", localLineStride)
        sd.set_field("localCs", localChanStride)
        sd.set_field("linesPerCh-1", linesPerChan - 1)
        sd.set_field("rsvd_92", self.sodGroup)
        sd.set_field("rud", self.reuseData)

        # Line A
        sd.set_field("minLines-1", minLines - 1)
        sd.set_field("rsvd_A0", self.sohGroup)
        sd.set_field("outputX", self.outputDimX)

        # Line B
        sd.set_field("biasBaseAddr", self.biasAddress)
        sd.set_field("scaleBaseAddr", self.scaleAddress)

        sd.set_pallete(None)

        content = sd.serialize()

        return content, relocInstance, relocWorkBuffer, relocInBlob

    def getContentForFullyConnected(self, baseAddress):
        noOfBlocks = 1 << self.mode
        bytesPerCoefficient = 1 << (1 - self.dataMode)
        pixelsPerBlock = self.inputDimX // noOfBlocks
        inDataLineStride = 16
        inDataBlockStride = inDataLineStride * pixelsPerBlock
        outDataLineStride = 16
        outDataBlockStride = outDataLineStride * pixelsPerBlock
        localLineStride = 16
        localBlockStride = localLineStride * pixelsPerBlock
        vectorLPB = self.inputDimX // noOfBlocks
        vectStrideIn = bytesPerCoefficient * pixelsPerBlock * 8
        vectStrideOut = bytesPerCoefficient * self.totalDimX * 8
        content = []
        relocInstance = []
        relocWorkBuffer = [baseAddress + 8*4, baseAddress + 16*4]
        relocInBlob = [baseAddress + 12*4]

        nextDescAddr = 0
        if not self.lastTile:
            nextDescAddr = baseAddress + (16 * 8)
            relocInstance = [baseAddress + 0]

        # If not set, ensure fields default to 0, not -1

        if self.poolKerDimX <= 0:
            self.poolKerDimX = 1
        if self.poolKerDimY <= 0:
            self.poolKerDimY = 1

        if self.useBias:
            relocInBlob.append(baseAddress + 22*4)
        if self.useScale:
            relocInBlob.append(baseAddress + 23*4)

        sd = SerializedDescriptor("FCL")

        # Line 0
        sd.set_field("NextDesc", nextDescAddr)
        sd.set_field("Type", HwDescOp.fully_connected_convolution.value)
        sd.set_field("mode", self.mode)
        sd.set_field("rsvd_00", self.interleavedInput + (self.interleavedOutput << 1))
        sd.set_field("id", self.id)
        sd.set_field("it", self.interruptTrigger)
        sd.set_field("cm", 0)
        sd.set_field("dm", 0)
        sd.set_field("disaint", self.disableInt)
        sd.set_field("rsvd_02", 0)

        # Line 1
        sd.set_field("iDimX-1", self.inputDimX - 1 )
        sd.set_field("iChans-1", self.noOfVectors - 1)
        sd.set_field("oChans-1", self.noOfVectors - 1)

        # Line 2
        sd.set_field("ChRamBlk-1", pixelsPerBlock-1)

        # Line 3
        sd.set_field("actualOutChannels", self.actualOutChannels-1)
        sd.set_field("X", self.x)

        # Line 4
        sd.set_field("dataBaseAddr", self.inputDataAddr)
        sd.set_field("t0", self.t0)
        sd.set_field("a0", self.a0)
        sd.set_field("a1", self.a1)
        sd.set_field("reluXEn", self.reluXEn)
        sd.set_field("reluEn", self.reluEn)

        # Line 5
        sd.set_field("dataChStr", inDataBlockStride)
        sd.set_field("dataLnStr", inDataLineStride)

        # Line 6
        sd.set_field("vecStrOut", vectStrideOut)
        sd.set_field("vecBaseAddr", self.vectorData)

        # Line 7
        sd.set_field("vecStrIn", vectStrideIn)
        sd.set_field("outLnStr", outDataLineStride)

        # Line 8
        sd.set_field("outBaseAddr", self.outputDataAddr)
        sd.set_field("outChStr", outDataBlockStride)

        # Line 9
        sd.set_field("localLs", localLineStride)
        sd.set_field("localBs", localBlockStride)
        sd.set_field("rud", self.reuseData)

        # Line A
        sd.set_field("Acc", self.accumulate)
        sd.set_field("vecLPB-1", vectorLPB-1)
        sd.set_field("outputX", 1)

        # Line B
        sd.set_field("biasBaseAddr", self.biasAddress)
        sd.set_field("scaleBaseAddr", self.scaleAddress)

        sd.set_pallete(None)

        content = sd.serialize()

        return content, relocInstance, relocWorkBuffer, relocInBlob

def taps_hwck_to_cnnhw(data, new_c, new_k):
    def hwck_to_kchw(data):
        return np.transpose(data, (3, 2, 0, 1))

    fh, fw, c, k = data.shape
    newdata = np.zeros((new_k, new_c, fh, fw), dtype=np.float16)

    # If the newdata size is more than already allocated, then we dont
    # compile correctly, because pointers to buffers break.
    # From now on, overallocation is performed, which makes the
    # following assert obsolete (See get_buffer function in FileIO.py)
    # assert(newdata.size <= data.size)

    newdata[0:k, 0:c, :, :] = hwck_to_kchw(data)
    newdata = newdata.reshape(new_k, new_c * fh * fw).transpose()
    newdata = np.vstack(np.hsplit(newdata, new_k // 8))
    return newdata.reshape((new_k // 8, new_c, fh*fw, 8))

    # # Equivalent slower code
    # fh, fw, c, k = data.shape
    # newdata = np.zeros((new_k // 8, new_c, fh*fw, 8), dtype = np.float16)

    # # If the newdata size is more than already allocated, then we dont
    # # compile correctly, because pointers to buffers break.
    # assert(newdata.size <= data.size)

    # for outch in range(0, k):
    #     g = outch >> 3
    #     i = outch & 7
    #     for inch in range(0, c):
    #         for y in range(0, fh):
    #             for x in range(0, fw):
    #                 newdata[g, inch, y*fw + x, i] = data[y, x, inch, outch]
    # return newdata

def regroup_taps(taps, splits):
    """ When doing split over the input channels,
        it is necessary to rearrange the taps (which
        have been converted for the hardware).

        input: output of `taps_hwck_to_cnnhw`
        splits: how many pieces is the input depth split into
    """

    # Reshape by adding an extra axis
    group, in_depth, ker_surf, out_ch = taps.shape

    taps = np.reshape(taps,
        (group, splits, in_depth // splits, ker_surf, out_ch))

    taps_stack = []
    for sp in range(splits):
        for g in range(group):
            taps_stack.append(taps[g, sp, :, :, :])

    return np.reshape(np.stack(taps_stack), (group, in_depth, ker_surf, out_ch))

def calcOutputSize(iX, iY, kX, kY, kS, pad):
    if (0 == pad):
        iX -= (kX & 0xFE)
        iY -= (kY & 0xFE)
    if (kS > 1):
        if (0 != iX % kS):
            iX = (iX // kS) + 1
        else:
            iX = iX // kS
        if (0 != iY % kS):
            iY = (iY // kS) + 1
        else:
            iY = iY // kS
    return (iX, iY)

def checkModeForValidity(iX, iY, iZ, oX, oY, oZ, kX, kY, kS, dataType, coeffType, mode):
    CNNHW_INTERNAL_MEMORY_SIZE = (128 * 1024)
    COEFF_PER_WORD_VALUES = {'FP16': 1, 'U8F': 2, '4BIT': 4, '2BIT': 8, '1BIT': 16, '1BITD': 16}
    BYTES_PER_PIXEL = {'FP16': 2, 'U8F': 1}
    blocks = 1 << mode
    inChansPerBlock = iZ // blocks
    coeffSetSize = kX*kY

    bytesPerLine = ((iX * BYTES_PER_PIXEL[dataType] + 15) // 16) * 16
    linesPerChannel = min(CNNHW_INTERNAL_MEMORY_SIZE // (bytesPerLine * iZ), iY, 2**9)

    coeffPerWord = COEFF_PER_WORD_VALUES[coeffType]
    coeffLPB = (inChansPerBlock * coeffSetSize + coeffPerWord - 1) // coeffPerWord

    if iX > 2**12 or iY > 2**12 or iZ > 2**11 or oZ > 2**11:
        return False, 0
    if kX > 2**4 or kY > 2**4 or kS > 2**4:
        return False, 0
    if inChansPerBlock > 2**11:
        return False, 0
    if coeffLPB > 2**8:
        return False, 0
    # if ((oX // kS) <= 4) and (linesPerChannel <= (kY + 2 * (kS + 1) + 1)):
    #     return False, 0
    # if ((oX // kS) > 4) and (linesPerChannel <= (kY + kS + 1 + 1)):
    #     print("linesPerChannel", linesPerChannel, kY, kS)
    #     return False, 0
    return True, (iZ // blocks) * kX * kY + [0, 5, 11, 19, 31][mode]

def splitConvolution(iX, iY, iZ, oX, oY, oZ, kX, kY, kS, pad, dataType, coeffType, modes = [0, 1, 2, 3, 4]):
    #This function splits the convolution operation into uniform pieces
    coX, coY = calcOutputSize(iX, iY, kX, kY, kS, pad)
    solutions = []
    solution = None
    for mode in modes:
        ramBlocks = 1 << mode
        inC = ((iZ + ramBlocks - 1) // ramBlocks) * ramBlocks
        maxOc = 256 // ramBlocks
        outC = ((oZ +  7) // 8) * 8
        maxOc = min(maxOc, outC)
        for oChansPerDescr in reversed([8 * i for i in range(1, (maxOc // 8) + 1)]):
            valid, cost = checkModeForValidity(iX, iY, inC, oX, oY, oChansPerDescr, kX, kY, kS, dataType, coeffType, mode)
            if valid:
                fullDescriptors = outC // oChansPerDescr
                remOChans = outC % oChansPerDescr
                partDescriptors = 0
                if remOChans:
                    partDescriptors = 1
                tcost = fullDescriptors * cost + partDescriptors * cost
                solutions.append((mode, inC, outC, fullDescriptors + partDescriptors, oChansPerDescr, remOChans, tcost))
    if(solutions):
        minDesc = min(solutions, key = lambda x:x[3])[3]
        solutions = list(filter(lambda x:(x[3] == minDesc), solutions))
        minCost = min(solutions, key = lambda x:x[6])[6]
        solutions = list(filter(lambda x:x[6] == minCost, solutions))
        (mode, inC, outC, totalDescriptors, oChansPerDescr, remOChans, tcost) = solutions[0]
        tiles = [(oChansPerDescr, mode)] * (outC // oChansPerDescr)
        if outC % oChansPerDescr:
            tiles.append((remOChans, mode))

        # Sum the tiles. If it is more than oZ, go to the last tile and change it to mode 0
        totalOutChans = sum([outChans for outChans, _ in tiles])
        if totalOutChans > oZ:
            if len(tiles) > 1:
                newTiles = [t for t in tiles[:-1]]

                almostTotalOutChans = sum([outChans for outChans, _ in newTiles])
                newTiles.append((oZ - almostTotalOutChans, tiles[0][1]))
            else:
                newTiles = [(oZ, tiles[0][1])]

            tiles = newTiles

        solution = (inC, outC, tiles)
    return solution

def splitPooling(inOutChans):
    sub_layers = [(16, 4) for i in range(inOutChans // 16)]
    rem = inOutChans % 16
    if(rem != 0):
        sub_layers.append((rem, 4))

    return (inOutChans, inOutChans, sub_layers)


def splitFullyConnected(inW, inN, modes = [0, 1, 2, 3, 4]):
    solutions = []
    solution = None
    for mode in modes:
        ramblocks = 1 << mode
        maxW = ramblocks * 256
        maxN = 256 // ramblocks
        W = ((inW + ramblocks - 1) // ramblocks) * ramblocks
        N = ((inN + 7) // 8) * 8
        workW = min(W, maxW)
        workN = min(N, maxN)
        while workW >= ramblocks:
            countH = math.ceil(W / workW)
            countV = math.ceil(N / workN)
            cost = countH * countV * (workW // ramblocks + [0, 5, 11, 19, 31][mode])
            solutions.append((mode, W, N, workW, workN, countH, countV, cost))
            workW //= 2
    minCountH = min(solutions, key = lambda x:x[5])[5]
    zolutions = [sol for sol in solutions if sol[5] == minCountH]
    zolutions.sort(key = lambda x:x[7])
    if zolutions:
        mode, W, N, workW, maxN, countH, countV, cost = zolutions[0]
        tilesV = [[(workW, maxN, mode)] * countH] * countV
        solution = (W, N, tilesV)
    return solution
