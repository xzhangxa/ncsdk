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

from collections import OrderedDict

# Computes the output dimension (x or y) based on the input size, kernel size, stride and padding
def calcOutputSize(inputSize, kernelSize, stride, pad):
    # if pad == 0:
    #     inputSize -= (kernelSize >> 1) << 1

    # if stride > 1:
    #     if inputSize % stride != 0:
    #         inputSize = (inputSize // stride) + 1
    #     else:
    #         inputSize = inputSize // stride

    # return inputSize

    padBefore, padAfter = pad

    return (inputSize - kernelSize + padBefore + padAfter) // stride + 1

def inputLinesForOutputLines(inputSize, kernelSize, stride, pad, output):
    padBefore, padAfter = pad

    # Output format: [outputStartIndex, outputEndIndex)
    outputStartIndex, outputEndIndex = output

    totalOutputSize = calcOutputSize(inputSize, kernelSize, stride, pad)
    # assert(outputStart >= 0 and totalOutputSize)

    # Negative value encodes the padding
    inputStartIndex = -padBefore + outputStartIndex * stride
    inputEndIndex = -padBefore + (outputEndIndex - 1) * stride + kernelSize

    if inputStartIndex < 0:
        """Negative inputStartIndex means that we use
           the original padding"""

        inputLinesBefore = 0
        inputStartIndex = 0
        if outputStartIndex == 0:
            junkOutputBefore = 0
        else:
            junkOutputBefore = outputStartIndex
    else:
        """Non-negative inputStartIndex means that we either
           have no padding, or we are in the middle of the image"""

        # We reduce the inputStartIndex to the smallest non-negative integer
        inputLinesBefore = inputStartIndex
        while inputLinesBefore >= stride:
            inputLinesBefore -= stride

        # Compute the junkOutputBefore
        junkOutputBefore = (inputLinesBefore + padBefore) // stride

    if inputEndIndex > inputSize:
        """Larger inputEndIndex means that we use
           the original padding at the bottom of the image"""
        paddingUsed = inputEndIndex - inputSize
        inputLinesAfter = 0
        inputEndIndex = inputSize

        """The hardware will continue to compute output lines,
           until the kernel is just inside the padded image."""
        junkOutputAfter = 0
        while paddingUsed + stride <= padAfter:
            paddingUsed += stride
            junkOutputAfter += 1
    else:
        """This value of inputEndIndex means that we either
           have no padding, or we are in the middle of the image"""

        inputLinesAfter = 0

        # Count how many kernels fit with the provided padding
        paddingUsed = 0
        junkOutputAfter = 0
        while paddingUsed + stride <= padAfter:
            paddingUsed += stride
            junkOutputAfter += 1

    """Note:
       * [inputStartIndex, inputEndIndex): is the original range,
         without account for splits.
       * inputLinesBefore: specifies how many elements we need to
         remove from inputStartIndex to get the correct starting
         point.
       * junkOutputBefore: With starting point inputStartIndex-inputLinesBefore,
         this value contains the junk lines we need to discard.
       * [outputStartIndex, outputEndIndex): is the output range
         we are interested in generating, without the extra junk
         that might be there.
       * junkOutputBefore: is the junk contained before the
         outputStartIndex.
       * junkOutputAfter: is the junk contained after the
         outputEndIndex.
         The output generated by the hardware is:
         [outputStartIndex-outputStartIndex, outputEndIndex+junkOutputAfter)."""

    # print("New computation")
    # print(    OrderedDict([
    #     ('inputStartIndex', inputStartIndex),
    #     ('inputEndIndex', inputEndIndex),
    #     ('inputLinesBefore', inputLinesBefore),
    #     ('inputLinesAfter', inputLinesAfter),
    #     ('outputStartIndex', outputStartIndex),
    #     ('outputEndIndex', outputEndIndex),
    #     ('junkOutputBefore', junkOutputBefore),
    #     ('junkOutputAfter', junkOutputAfter)
    # ]))
    # print()

    # Interval format [inputStartIndex, inputEndIndex)
    return ((inputStartIndex, inputEndIndex),
             inputLinesBefore, inputLinesAfter,
            (outputStartIndex, outputEndIndex),
             junkOutputBefore, junkOutputAfter)

def maximizeOutput(inputSize, kernelSize, stride, pad, output, maxOutputSliceLines):
    outputSize = calcOutputSize(inputSize, kernelSize, stride, pad)
    outputStartIndex, outputEndIndex = output
    _, _, _, _, \
    junkOutputBefore, junkOutputAfter = inputLinesForOutputLines(inputSize, kernelSize, stride, pad, (outputStartIndex, outputEndIndex))
    totalOutputSlice = junkOutputBefore + (outputEndIndex - outputStartIndex) + junkOutputAfter

    # print(inputSize)
    # print(totalOutputSlice)

    def isValid(totalOutputSlice, maxOutputSliceLines, outputEndIndex, outputSize):
        return totalOutputSlice <= maxOutputSliceLines and outputEndIndex <= outputSize

    extraLines = 0
    while not isValid(totalOutputSlice, maxOutputSliceLines, outputEndIndex + extraLines, outputSize):
        extraLines -= 1
        _, _, _, _, \
        junkOutputBefore, junkOutputAfter = inputLinesForOutputLines(inputSize, kernelSize, stride, pad, (outputStartIndex, outputEndIndex + extraLines))
        totalOutputSlice = junkOutputBefore + (outputEndIndex + extraLines - outputStartIndex) + junkOutputAfter

    return outputEndIndex + extraLines + (not isValid(totalOutputSlice, maxOutputSliceLines, outputEndIndex, outputSize))

def arange(size, step):
    left = list(range(0, size, step))
    right = list(left[1:] + [size])
    return zip(left, right)

def heightSolution(inputSize, kernelSize, stride, pad, maxOutputLines):
    outputSize = calcOutputSize(inputSize, kernelSize, stride, pad)
    # print("heightSolution| outputSize: {}".format(outputSize))

    outputStartIndex = 0
    heightSol = list()

    while True:
        prevOutputStartIndex = outputStartIndex
        outputEndIndex = min(outputSize, outputStartIndex + maxOutputLines)

        if outputEndIndex - outputStartIndex <= 0:
            break

        (inputStartIndex, inputEndIndex), \
        inputLinesBefore, inputLinesAfter, _, \
        junkOutputBefore, junkOutputAfter = inputLinesForOutputLines(inputSize, kernelSize, stride, pad, (outputStartIndex, outputEndIndex))

        # print("pad", pad)
        # print("(inputStartIndex, inputEndIndex) =", (inputStartIndex, inputEndIndex))
        # print("Junk:", junkOutputBefore, junkOutputAfter)

        newOutputEndIndex = maximizeOutput(inputSize, kernelSize, stride, pad, (outputStartIndex, outputEndIndex), maxOutputLines)
        # print(newOutputEndIndex)

        # Recompute the (inputStartIndex, inputEndIndex) for the updated newOutputEndIndex
        (inputStartIndex, inputEndIndex), \
        inputLinesBefore, inputLinesAfter, \
        (outputStartIndex, outputEndIndex), \
        junkOutputBefore, junkOutputAfter = inputLinesForOutputLines(inputSize, kernelSize, stride, pad, (outputStartIndex, newOutputEndIndex))

        heightSol.append((inputLinesBefore + inputEndIndex - inputStartIndex + inputLinesAfter,
           junkOutputBefore + outputEndIndex - outputStartIndex + junkOutputAfter,
           junkOutputBefore, junkOutputAfter,
           inputStartIndex - inputLinesBefore, inputEndIndex + inputLinesAfter,
           outputStartIndex, outputEndIndex))

        outputStartIndex = outputEndIndex

        if prevOutputStartIndex == outputStartIndex:
            """ If CMX is very small, it is possible for the output to contain
                only junk lines. This means that we cannot have progress, therefore
                we will end up in an infinite loop."""
            raise Exception("Available CMX memory is not enough to generate a single proper line of output")

    return heightSol


def heightSolutionWithPooling(inputSize, kernelSize, stride, pad, maxOutputLines):

    # This is very specific case for 3x3p1s1 convlution, followed by 2x2s2 pooling
    # with even height.
    assert(kernelSize == 3 and stride == 1 and pad == 1)
    assert(inputSize % 2 == 0)

    # For this specific case, the outputSize is:
    outputSize = inputSize // 2
    if outputSize > maxOutputLines:
        maxOutputLines = maxOutputLines if (maxOutputLines % 2 == 1) else (maxOutputLines - 1)

    # print("maxOutputLines:", maxOutputLines)
    # print("outputSize:", outputSize)

    # print(list(range(0, outputSize, maxOutputLines)))

    inputStartIndex, outputStartIndex = 0, 0

    heightSol = list()
    while True:
        inputEndIndex = min(inputStartIndex + 2*maxOutputLines, inputSize)
        outputEndIndex = min(outputStartIndex + maxOutputLines, outputSize)

        trueInputNeeded = inputEndIndex - inputStartIndex
        outputWithJunk = outputEndIndex - outputStartIndex
        junkBefore = 1 if outputStartIndex > 0 else 0
        junkAfter = 1 if outputEndIndex < outputSize else 0

        outputStartIndex += junkBefore
        outputEndIndex -= junkAfter

        heightSol.append((trueInputNeeded, outputWithJunk, junkBefore, junkAfter, inputStartIndex, inputEndIndex, outputStartIndex, outputEndIndex))
        # print("trueInputNeeded, outputWithJunk, junkBefore, junkAfter, inputStartIndex, inputEndIndex, outputStartIndex, outputEndIndex",
            # trueInputNeeded, outputWithJunk, junkBefore, junkAfter, inputStartIndex, inputEndIndex, outputStartIndex, outputEndIndex)

        inputStartIndex = inputEndIndex - 4
        outputStartIndex = outputEndIndex - 1

        if outputEndIndex >= outputSize:
            break

    return heightSol

def main():
    inputSize, kernelSize, stride, pad = 112, 3, 2, (0, 1)
    maxOutputLines = 30

    print(heightSolution(inputSize, kernelSize, stride, pad, maxOutputLines))

# main()


def mmain():

    # inputSize, kernelSize, stride, pad = 224, 7, 2, (3, 3)
    inputSize, kernelSize, stride, pad = 224, 3, 2, (1, 1)
    maxOutputLines = 36

    outputSize = calcOutputSize(inputSize, kernelSize, stride, pad)

    outputStartIndex = 0
    heightSol = list()

    while True:
        outputEndIndex = min(outputSize, outputStartIndex + maxOutputLines)

        if outputEndIndex - outputStartIndex <= 0:
            break

        (inputStartIndex, inputEndIndex), \
        inputLinesBefore, inputLinesAfter, _, \
        junkOutputBefore, junkOutputAfter = inputLinesForOutputLines(inputSize, kernelSize, stride, pad, (outputStartIndex, outputEndIndex))

        newOutputEndIndex = maximizeOutput(inputSize, kernelSize, stride, pad, (outputStartIndex, outputEndIndex), maxOutputLines)

        # Recompute the (inputStartIndex, inputEndIndex) for the updated newOutputEndIndex
        (inputStartIndex, inputEndIndex), \
        inputLinesBefore, inputLinesAfter, \
        (outputStartIndex, outputEndIndex), \
        junkOutputBefore, junkOutputAfter = inputLinesForOutputLines(inputSize, kernelSize, stride, pad, (outputStartIndex, newOutputEndIndex))

        heightSol.append((inputLinesBefore + inputEndIndex - inputStartIndex + inputLinesAfter,
           junkOutputBefore + outputEndIndex - outputStartIndex + junkOutputAfter,
           junkOutputBefore, junkOutputAfter,
           inputStartIndex - inputLinesBefore, inputEndIndex + inputLinesAfter,
           outputStartIndex, outputEndIndex))

        print("New computation")
        print(    OrderedDict([
            ('inputStartIndex', inputStartIndex),
            ('inputEndIndex', inputEndIndex),
            ('inputLinesBefore', inputLinesBefore),
            ('inputLinesAfter', inputLinesAfter),
            ('outputStartIndex', outputStartIndex),
            ('outputEndIndex', outputEndIndex),
            ('junkOutputBefore', junkOutputBefore),
            ('junkOutputAfter', junkOutputAfter)
        ]))
        print()
        # quit()

        outputStartIndex = outputEndIndex


    print(heightSol)

    return heightSol

# mmain()