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


from mvnctools.Models.HwDescriptor import HwDescriptor, HwDescriptorList
from mvnctools.Controllers.Soh import heightSolution, heightSolutionWithPooling
from mvnctools.Controllers.NCE import NCE_Scheduler
from mvnctools.Controllers.Parsers.Parser.Hw import *

def round_up(x, mult):
    return ((x + mult -1) // mult) * mult

def getPadding(inputSize, outputSize, kernelDimX, kernelDimY, kernelStride):
    valid_out_x = math.ceil((inputSize[2] - kernelDimX + 1) / kernelStride)
    valid_out_y = math.ceil((inputSize[3] - kernelDimY + 1) / kernelStride)

    pad_along_x = (outputSize[2] - 1) * kernelStride + kernelDimX - inputSize[2]
    pad_along_y = (outputSize[3] - 1) * kernelStride + kernelDimY - inputSize[3]
    pad_left = pad_along_x // 2
    pad_right = pad_along_x - pad_left
    pad_top = pad_along_y // 2
    pad_bottom = pad_along_y - pad_top
    return (outputSize[2] != valid_out_x or outputSize[3] != valid_out_y, pad_left, pad_right, pad_top, pad_bottom)

def getStrides(inputSize, outputSize, bytesPerPixel):
    inputStrideWidth = bytesPerPixel
    inputStrideHeight = ((inputSize[2] * inputStrideWidth + 15) // 16) * 16
    inputStrideZ = inputStrideHeight * inputSize[3]
    outputStrideWidth = bytesPerPixel
    outputStrideHeight = ((outputSize[2] * outputStrideWidth + 15) // 16) * 16
    outputStrideZ = outputStrideHeight * outputSize[3]

    return (inputStrideWidth, inputStrideHeight, inputStrideZ), (outputStrideWidth, outputStrideHeight, outputStrideZ)

def trueRowAdaptationB0(maxOutputLines, pad_left, pad_right, pad_top, pad_bottom, inputSize, kernelHeight):

    if pad_bottom:
        max_size = 2048
    elif (pad_left or pad_right): 
        max_size = 4096
    else:
        max_size = round_up(inputSize[3], 16) * (maxOutputLines-1 + kernelHeight-1)
    maxOutputLines = max_size//round_up(inputSize[3], 16) -kernelHeight + 2

    return maxOutputLines

def getTensorSize(inputTensor):
    return inputTensor.getShape()

def compile_conv(stage):

    bytesPerPixel = 2# np.dtype(stage.getWeights().dtype).itemsize

    # Basic checks
    assert len(stage.inputTensors) == 1
    assert len(stage.outputTensors) == 1
    assert stage.kernelWidth == stage.kernelHeight
    assert stage.strideWidth == stage.strideHeight

    # Get input and output tensors
    inputTensor = stage.inputTensors[0]
    outputTensor = stage.outputTensors[0]
    inputSize = getTensorSize(inputTensor)
    outputSize = getTensorSize(outputTensor)

    # Get solution and splits
    solution = stage.getSolution()
    splits = stage.getSplitOverC()
    # solution[2].sort(key = lambda tup: tup[1])

    completeHwDescList = []

    # Default for relu
    reluOnShaveAccumulation = False
    reluNegSlope = 0
    reluPosSlope = 1

    padEnable, pad_left, pad_right, pad_top, pad_bottom = getPadding(inputSize, outputSize, stage.kernelWidth, stage.kernelHeight, stage.strideWidth)
    # Cicle input channel splits
    for sodIndex in range(splits):
        newInputDimZ, newOutputDimZ, tiles = solution
        # This situation may arise situations like pool-> convolution, when pool create a 
        # multiple of 16 in Z input tensor for the conv
        if inputTensor.getTopEncloserRecursive().getShape()[1] != newInputDimZ*splits:
            newInputDimZ = inputTensor.getTopEncloserRecursive().getShape()[1] // splits
        if outputTensor.getTopEncloserRecursive().getShape()[1] > newOutputDimZ:
            newOutputDimZ = outputTensor.getTopEncloserRecursive().getShape()[1]
        #Calculate in/out strides
        inputStride, outputStride = getStrides(inputSize, outputSize, bytesPerPixel)

        # Culculate buffers Size
        inputBufferSize = newInputDimZ * inputStride[2]
        outputBufferSize = newOutputDimZ * outputStride[2]
        tapsBufferSize = newInputDimZ * newOutputDimZ * stage.kernelWidth * stage.kernelHeight * bytesPerPixel
        biasBufferSize = newOutputDimZ * bytesPerPixel if stage.biasEnabled() is not None else 0

        heightSplits = []
        # Calculate the maximum number of output lines
        if stage.isStreaming()  and stage.isSplitOverH():
            maxOutputChannelsInDescr = max(tiles, key=lambda x: x[0])[0]
            bytesPerFullDepthSlice = bytesPerPixel * maxOutputChannelsInDescr * round_up(outputSize[3], 8)
            maxOutputLines = stage.getCMXforStreaming() // bytesPerFullDepthSlice
        else:
            maxOutputLines = outputSize[2]

        # Get the number of split over height
        if isinstance(stage, HwConvolutionPooling):
            # For conv3x3s1p1 and fused 2x2s2 pooling, we need 4 extra lines
            # Also, for this specific case, the maxOutputLines is doubled, because the output
            # is reduced by a factor of 2
            heightSplits = heightSolutionWithPooling(inputSize[2], stage.kernelHeight, stage.strideHeight, stage.paddingHeight, maxOutputLines)
        else:
            # For convolution without fused pooling
            # The following is not correct for convolution. We cannot have selective zero padding
            # pad = (stage.kernelHeight // 2 if pad_top > 0 else 0, stage.kernelHeight // 2 if pad_bottom > 0 else 0)
            assert(stage.paddingWidth == stage.paddingHeight)
            assert(stage.paddingHeight == 0 or stage.paddingHeight == (stage.kernelHeight // 2))
            pad = (stage.paddingHeight, stage.paddingHeight)
            heightSplits = heightSolution(inputSize[2], stage.kernelHeight, stage.strideHeight, pad, maxOutputLines)


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

                conv = HwDescriptor(0, 0, 1, 0, mode, tileIndex, lastTile, stage.getName())
                conv.topOutputJunk = junkBefore
                conv.bottomOutputJunk = junkAfter

                if round_up(inputSize[3], 8) * inputSize[2] * (newInputDimZ) * 2 < 128*1024:
                    if tileIndex == 0:
                        conv.reuseData = stage.rud
                    else:
                        if tiles[tileIndex-1][1] != mode:
                            conv.reuseData = 0
                        else:
                            conv.reuseData = 1

                # Input in interleaved
                if inputTensor.getLayout() == (0, 2, 1, 3):
                    conv.setupInterleaved(True)
                    conv.setupInterleavedInput(True)

                    split_by_input_sz = sodIndex * newInputDimZ * inputStride[1]
                    split_by_height_sz = (newInputDimZ * splits) * inputStride[1] * inputStartIndex
                    offset = split_by_input_sz + split_by_height_sz
                    conv.setupInput(offset, inputSize[3], trueInputNeeded, newInputDimZ, inputSize[2], (newInputDimZ * splits))
                else:
                    split_by_input_sz = sodIndex * newInputDimZ * inputStride[1] * inputSize[2]
                    split_by_height_sz = inputStride[1] * inputStartIndex
                    offset = split_by_input_sz + split_by_height_sz
                    conv.setupInput(offset, inputSize[3], trueInputNeeded, newInputDimZ, inputSize[2], (newInputDimZ * splits))
                
                if stage.isConcat():
                    totalOutChans = stage.totalOutChans
                else:
                    totalOutChans = sum([outChans for outChans, _ in tiles])

                if newOutputDimZ > totalOutChans and newOutputDimZ == outputTensor.getTopEncloserRecursive().shape[1]:
                    totalOutChans = max(newOutputDimZ, totalOutChans)

                # output in interleaved
                if outputTensor.getLayout() == (0, 2, 1, 3):
                    conv.setupInterleavedOutput(True)
                    conv.setupOutput(outputStartIndex * totalOutChans * outputStride[1] + outChanOffset * outputStride[1], outputSize[3], outputSize[2], outChans, outputSize[2], totalOutChans)
                else:
                    conv.setupOutput(bytesPerPixel*outputStartIndex*round_up(outputSize[3], 8) + outChanOffset * outputStride[2], outputSize[3], outputSize[2], outChans, outputSize[2], totalOutChans)

                taps_processed = sum([a for a, _ in tiles[:tileIndex]])
                # taps_total = sum([a for a, _ in tiles])
                taps_total = newOutputDimZ

                split_by_input_taps_sz = bytesPerPixel * taps_total * newInputDimZ * stage.kernelWidth * stage.kernelHeight
                conv.setupConvolutionCoefficients(0,
                        sodIndex * split_by_input_taps_sz +
                        bytesPerPixel * taps_processed * newInputDimZ * stage.kernelWidth * stage.kernelHeight,
                        stage.kernelWidth, stage.kernelHeight, stage.strideWidth)
                if padEnable:
                    conv.setupPadding(0)

                # if sohIndex == 0 and (stage.bias is not None):
                if sodIndex == 0 and stage.biasEnabled():
                    conv.setupBias(outChanOffset * bytesPerPixel)

                if isinstance(stage, HwConvolutionPooling) and (sodIndex == 0):
                    # If we have fused pooling and more that one depth groups, then
                    # we need not to fuse the pooling
                    if stage.type == Pooling.Type.MAX:
                        poolType = 0
                    else:
                        raise ValueError('Average pooling fusion with conv not supported')

                    conv.setupConvolutionPooling(poolType, stage.pooling_kernelWidth, stage.pooling_kernelHeight)

                # Since we run this in hardware, we need to convert postops (and whatever else) that are
                # supported by the hardware
                postOp = [x for x in stage.getPostOp() if isinstance(x, ReLU)]
                if len(postOp) > 1:
                    raise ValueError("Only one post operation is supported. Found {}".format(postOp))
                elif len(postOp) == 1:
                    # General form of hardware Parametrized ReLU (PRelu):
                    #                     / a0*x, if x < t0
                    # f(x; t0, a0, a1) = |
                    #                     \ a1*x, if x >= t0
                    if hasattr(stage.post_op[0], 'negativeSlope'):
                        post_param1 = stage.post_op[0].negativeSlope
                    else:
                        post_param1 = 0
                    if splits > 1:
                        reluOnShaveAccumulation = True
                        reluNegSlope = post_param1
                    else:
                        reluOnShaveAccumulation = False
                        reluNegSlope = 0
                        if post_param1 == 0.0:
                            conv.setupRelu(0, 0, 1)
                        else:
                            # Check that the scaling with be 10 bits signed integer
                            assert(abs(int(round(1 / post_param1))) < 2 ** 9)
                            stage.setReLUScale(post_param1)
                            conv.setupRelu(0, 1, int(round(1 / post_param1)))
                            conv.setupScale(outChanOffset * bytesPerPixel)

                outChanOffset += outChans
                conv.sohGroup = sohIndex
                conv.sodGroup = sodIndex
                sohGroup.pushDescriptor(conv)

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

    # Convert taps to hardware
    weights = stage.getWeights()
    taps = taps_hwck_to_cnnhw(np.transpose(weights.data, (2,3,1,0)), newInputDimZ * splits, newOutputDimZ)
    if splits > 1:
        taps = regroup_taps(taps, splits)

    if stage.biasEnabled():
        bias = stage.getBias().data
    else:
        bias = None

    # TODO: Implement ME!!
    scale = None

    #associate taps to the layer
    weights.data = taps
    weights.shape = (newOutputDimZ, newInputDimZ * splits, weights.shape[2], weights.shape[3])

    concatOffset, finalOutputBufferSize = stage.getConcatParameters()

    if finalOutputBufferSize == 0:
        finalOutputBufferSize = outputBufferSize

    solution = (inputStride, outputStride, \
                (newInputDimZ, newOutputDimZ, concatOffset, reluOnShaveAccumulation, reluNegSlope, reluPosSlope), \
                (inputBufferSize, finalOutputBufferSize, tapsBufferSize, biasBufferSize), finalHwDescList, taps, bias, scale)

    return solution


def compile_pooling(stage):

        bytesPerPixel = 2

        # Basic checks
        assert len(stage.inputTensors) == 1
        assert len(stage.outputTensors) == 1
        assert stage.kernelWidth == stage.kernelHeight
        assert stage.strideWidth == stage.strideHeight

        # Get input and output tensors
        inputTensor = stage.inputTensors[0]
        outputTensor = stage.outputTensors[0]
        inputSize = getTensorSize(inputTensor)
        outputSize = getTensorSize(outputTensor)


        splits = 1 # which means no splits
        completeHwDescList = []

        padEnable, pad_left, pad_right, pad_top, pad_bottom = getPadding(inputSize, outputSize, stage.kernelWidth, stage.kernelHeight, stage.strideWidth)
        solution = stage.getSolution()

        for sodIndex in range(splits):
            newInputDimZ, newOutputDimZ, tiles = solution
            if inputTensor.getTopEncloserRecursive().getShape()[1] != newInputDimZ*splits:
                newInputDimZ = inputTensor.getTopEncloserRecursive().getShape()[1] // splits
            if outputTensor.getTopEncloserRecursive().getShape()[1] > newOutputDimZ:
                newOutputDimZ = outputTensor.getTopEncloserRecursive().getShape()[1]
            #Calculate in/out strides
            inputStrideX = bytesPerPixel
            inputStrideY = ((inputSize[3] * inputStrideX + 15) // 16) * 16
            inputStrideZ = inputStrideY * inputSize[2]

            outputStrideX = bytesPerPixel
            outputStrideY = ((outputSize[3] * outputStrideX + 15) // 16) * 16
            outputStrideZ = outputStrideY * outputSize[2]

            # Culculate buffers Size
            inputBufferSize = newInputDimZ * inputStrideZ
            outputBufferSize = newOutputDimZ * outputStrideZ
            tapsBufferSize = 0
            biasBufferSize = 0

            heightSplits = []
            # Calculate the maximum number of output lines
            if stage.isStreaming()  and stage.isSplitOverH():
                maxOutputChannelsInDescr = max(tiles, key=lambda x: x[0])[0]
                bytesPerFullDepthSlice = bytesPerPixel * maxOutputChannelsInDescr * round_up(outputSize[3], 8)
                maxOutputLines = stage.getCMXforStreaming() // bytesPerFullDepthSlice
            else:
                maxOutputLines = outputSize[2]

            if padEnable == 1:
                maxOutputLines = trueRowAdaptationB0(maxOutputLines, pad_left, pad_right, pad_top, pad_bottom, inputSize, stage.kernelHeight)

            pad = (stage.kernelHeight // 2 if pad_top > 0 else 0, stage.kernelHeight // 2 if pad_bottom > 0 else 0)
            heightSplits = heightSolution(inputSize[2], stage.kernelHeight, stage.strideHeight, pad, maxOutputLines)

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

                    pool = HwDescriptor(0, 0, 1, 0, mode, tileIndex, lastTile, stage.getName())
                    pool.topOutputJunk = junkBefore
                    pool.bottomOutputJunk = junkAfter

                    chansSoFar = sum([outChans for outChans, _ in tiles[0:tileIndex]])
                    if stage.isConcat():
                        totalOutChans = stage.totalOutChans
                    else:
                        totalOutChans = sum([outChans for outChans, _ in tiles])

                    # Input in interleaved
                    if inputTensor.getLayout() == (0, 2, 1, 3):
                        pool.setupInterleaved(True)
                        pool.setupInterleavedInput(True)

                        split_by_input_sz = chansSoFar * inputStrideY
                        split_by_height_sz = inputSize[1] * inputStrideY * inputStartIndex
                        offset = split_by_input_sz + split_by_height_sz
                        pool.setupInput(offset, inputSize[3], trueInputNeeded, outChans, inputSize[2], newInputDimZ)
                    else:
                        split_by_input_sz = chansSoFar * inputStrideY * inputSize[2]
                        split_by_height_sz = inputStrideY * inputStartIndex
                        offset = split_by_input_sz + split_by_height_sz
                        pool.setupInput(offset, inputSize[3], trueInputNeeded, outChans, inputSize[2], newInputDimZ)

                    # output in interleaved
                    if outputTensor.getLayout() == (0, 2, 1, 3):
                        pool.setupInterleavedOutput(True)
                        pool.setupOutput(outputStartIndex * totalOutChans * outputStrideY + outChanOffset * outputStrideY, outputSize[3], outputSize[2], outChans, outputSize[2], totalOutChans)
                    else:
                        pool.setupOutput(bytesPerPixel*outputStartIndex*round_up(outputSize[3], 8) + outChanOffset * outputStrideZ, outputSize[3], outputSize[2], outChans, outputSize[2], totalOutChans)

                    if stage.type == Pooling.Type.MAX:
                        poolType = 0
                    else:
                        poolType = 1
                    pool.setupPooling(poolType, stage.kernelWidth, stage.kernelHeight, stage.strideWidth)
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

        concatOffset, finalOutputBufferSize = stage.getConcatParameters()

        if finalOutputBufferSize == 0:
            finalOutputBufferSize = outputBufferSize

        solution = ((inputStrideX, inputStrideY, inputStrideZ), (outputStrideX, outputStrideY, outputStrideZ), \
                    (newInputDimZ, newOutputDimZ, concatOffset, False, 0.0, 1.0), \
                    (inputBufferSize, finalOutputBufferSize, tapsBufferSize, biasBufferSize), finalHwDescList, None, None, None)
        return solution

def compile_fc(stage):


    bytesPerPixel = 2# np.dtype(stage.getWeights().dtype).itemsize

    # Basic checks
    assert len(stage.inputTensors) == 1
    assert len(stage.outputTensors) == 1

    # Get input and output tensors
    inputTensor = stage.inputTensors[0]
    outputTensor = stage.outputTensors[0]
    inputSize = getTensorSize(inputTensor)
    outputSize = getTensorSize(outputTensor)

    # Get solution and splits
    solution = stage.getSolution()
    splits = stage.getSplitOverC()

    newInputDimZ, newOutputDimZ, tiles = solution

    assert(inputSize[1] % splits == 0)

    completeHwDescList = []

    # Default for relu
    reluOnShaveAccumulation = False
    reluNegSlope = 0
    reluPosSlope = 1

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
    if stage.biasEnabled():
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
        actualOutputChannels = outputSize[1]
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
            fc = HwDescriptor(0, 0, 1, 0, mode, genTileIndex, genLastTile, stage.getName())
            fc.setupInputFullyConnected(inputOffset * inputStrideZ, accum, newInputDimZ, maxW)
            fc.setupOutputFullyConnected(outputOffset * outputStrideZ)
            fc.setupVectorsFullyConnected(0, tapsOffset * 2, maxN)

            # Store the actual output in the descriptor
            fc.actualOutChannels = descriptorOutputChannels

            if lastSubTile:
                postOp = [x for x in stage.getPostOp() if type(x) == ReLU]
                if len(postOp) > 1:
                    raise ValueError("Only one post operation is supported. Found {}".format(postOp))
                elif len(postOp) == 1:
                    fc.setupRelu(0, 0, 1)

            if stage.biasEnabled():
                fc.setupBias(outputOffset * bytesPerPixel)
            hwDescList.pushDescriptor(fc)
            genTileIndex += 1
            inputOffset += maxW
        outputOffset += maxN

    # Convert taps to hardware
    weights = stage.getWeights()
    taps = taps_hwck_to_cnnhw(np.transpose(weights.data, (0, 1, 3, 2)), newInputDimZ * splits, newOutputDimZ)

    #associate taps to the layer
    weights.data = taps
    weights.shape = (newOutputDimZ, newInputDimZ * splits, weights.shape[0], weights.shape[1])

    concatOffset, finalOutputBufferSize = stage.getConcatParameters()

    if finalOutputBufferSize == 0:
        finalOutputBufferSize = outputBufferSize

    solution = ((inputStrideX, inputStrideY, inputStrideZ), (outputStrideX, outputStrideY, outputStrideZ), \
                (newInputDimZ, newOutputDimZ, concatOffset, False, 0.0, 1.0), \
                (inputBufferSize, finalOutputBufferSize, tapsBufferSize, biasBufferSize), hwDescList, taps, None, None)

    return solution


def hwCompile(stage):

    if not stage.isHW:
        return None, None, None, None, None, None, None, None     # Fallback to SW

    if type(stage) in [HwConvolution, HwConvolutionPooling]:
        return compile_conv(stage)
    elif isinstance(stage, HwPooling):
        return compile_pooling(stage)
    elif isinstance(stage, HwFC):
        return compile_fc(stage)
    else:
        raise ValueError("Layer {} of type {} not support for HW compilation".format(stage.getName(), type(stage)))



def taps_hwck_to_cnnhw(data, new_c, new_k):
    def hwck_to_kchw(data):
        return np.transpose(data, (3, 2, 0, 1))

    fh, fw, c, k = data.shape
    newdata = np.zeros((new_k, new_c, fh, fw), dtype=np.float16)

    newdata[0:k, 0:c, :, :] = hwck_to_kchw(data)
    newdata = newdata.reshape(new_k, new_c * fh * fw).transpose()
    newdata = np.vstack(np.hsplit(newdata, new_k // 8))
    return newdata.reshape((new_k // 8, new_c, fh*fw, 8))

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

