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

#!/usr/bin/env python

from mvnctools.Controllers.ProductionSchedulers import conf
from mvnctools.Controllers.ProductionSchedulers import googlenet, googlenet_best, yolo_tiny, vgg16, googlenet_ssd
from collections import OrderedDict
import mvnctools.Controllers.Globals as GLOBALS


def get_null_terminating_name(x):
    """Input is a bytes class"""
    return (x.split(b'\0', 1)[0]).decode('utf-8')


class PingPongSchedule:
    def __init__(self, ordered_dict=OrderedDict(), enable_perm=False):
        self.ordered_dict = ordered_dict
        self.enable_perm = enable_perm

    def __getitem__(self, key):
        try:
            # Return only the first 3 elements of the tuple
            if self.getDict(key) is not False:
                # Hw config
                return self.ordered_dict[key][1:4]
            else:
                # SW Fallback config
                return self.ordered_dict[key]
        except KeyError:
            print("Falling back to default ping-pong pair")
            self.ordered_dict[key] = conf()
            return self.__getitem__(key)

    def getDict(self, key):
        try:
            return self.ordered_dict[key]
        except KeyError:
            self.ordered_dict[key] = conf()
            return self.ordered_dict[key]

    def isStreamed(self, key):
        try:
            return self.getDict(key)[0][0].lower() == 's'
        except KeyError:
            print("Falling back to non-streaming")
            return False

    def isStreamedAndSplit(self, key):
        try:
            return self.isStreamed(key) and self.getDict(key)[0][1].lower() == 's'
        except KeyError:
            print("Falling back to non-streaming")
            return False

    def _inCmx_(self, key, src_dst, letter):
        return self.__getitem__(key)[0][src_dst] == letter

    def sourceInCmxA(self, key):
        return self._inCmx_(key, 0, 'L')

    def sourceInCmxAuxA(self, key):
        return self._inCmx_(key, 0, 'l')

    def sourceInCmxB(self, key):
        return self._inCmx_(key, 0, 'R')

    def sourceInCmxAuxB(self, key):
        return self._inCmx_(key, 0, 'r')

    def sourceInCmx(self, key):
        inA = self.sourceInCmxA(key) or self.sourceInCmxAuxA(key)
        inB = self.sourceInCmxB(key) or self.sourceInCmxAuxB(key)
        return inA or inB

    def destinationInCmxA(self, key):
        return self._inCmx_(key, 1, 'L')

    def destinationInCmxAuxA(self, key):
        return self._inCmx_(key, 1, 'l')

    def destinationInCmxB(self, key):
        return self._inCmx_(key, 1, 'R')

    def destinationInCmxAuxB(self, key):
        return self._inCmx_(key, 1, 'r')

    def destinationInCmx(self, key):
        inA = self.sourceInCmxA(key) or self.sourceInCmxAuxA(key)
        inB = self.sourceInCmxB(key) or self.sourceInCmxAuxB(key)
        return inA or inB

    def unloadCmxBuffer(self, key):
        if len(self.getDict(key)[1]) < 3:
            return False
        return self.getDict(key)[1][2].lower() == 'u'

    def overwriteInput(self, key):
        if len(self.getDict(key)[1]) < 3:
            return False
        return self.getDict(key)[1][2].lower() == 'o'

    def cmxForStreaming(self, key):
        """Returns CMX size in bytes. The size is for one of the buffers.
           In total, double the returned size is required."""
        if self.getDict(key) is not False:      # Not needed for Software Fallback
            return 1024 * self.getDict(key)[4][1]

    def streamingCmxPos(self, key):
        return self.getDict(key)[4][0]


def detectPermutationOrder(stage_list):
    # Get the relative order between the layers:
    d = OrderedDict()
    for i, s in enumerate(stage_list):
        if isinstance(s.name, str):
            name = s.name
        else:
            name = get_null_terminating_name(s.name)
        d[name] = (i, True if name in pingPongPair.ordered_dict else False)
    # print(d)

    # Extract the basic blocks. A basic block is a permutation unit,
    # assuming that software layers are interweaven with hardware ones.
    basicBlocks = [[]]
    for k, v in d.items():
        if not v[1]:
            basicBlocks[-1].append((k, v[0]))
        else:
            basicBlocks.append([(k, v[0])])
    # print(basicBlocks)

    permutedBasicBlocks = []
    noneDict = {}
    for b in basicBlocks:
        head = b[0][0]
        if head in pingPongPair.ordered_dict:
            noneDict[head] = b
            permutedBasicBlocks.append([(None, None)])
        else:
            permutedBasicBlocks.append(b)
    # print(permutedBasicBlocks)
    # print(noneDict)

    # Copy the pingPong dict and keep only the relevant info.
    # In particular, keep only layers that exist in the provided network.
    copiedPingPong = OrderedDict()
    for k, v in pingPongPair.ordered_dict.items():
        if k in noneDict:
            copiedPingPong[k] = v

    # Patch the None placeholders with the permuted info
    cnt = 0
    for idx, b in enumerate(permutedBasicBlocks):
        head = b[0][0]
        if head is None:
            key = list(copiedPingPong.items())[cnt][0]
            permutedBasicBlocks[idx] = noneDict[key]
            cnt += 1
    # print(permutedBasicBlocks)

    return permutedBasicBlocks

# SS -> Enable streaming and split over height if necessary.
# SX -> Enable streaming, but don't split over height. We need to make sure
#       that every hardware descriptor fits in CMX.
# When using any of the above two options, the data should move DDR->DDR.
# X? -> Don't stream, and don't split over height. '?' is "don't care".
#
# R -> Output to right part of CMX.
# r -> Output to auxiliary right part of CMX.
# L -> Output to left part of CMX.
# l -> Output to auxiliary left part of CMX.
# U -> Unload CMX, i.e. return data from CMX to DDR (blocking DMA)
# O -> Overwrite input that exists in CMX. Can be used only like 'RrO'.

pingPongPair = PingPongSchedule()


def ppInit(choice):
    global pingPongPair

    schedules = {
        "googlenet": [googlenet, False],
        "googlenet_best": [googlenet_best, True],
        "yolo_tiny": [yolo_tiny, False],
        "vgg": [vgg16, False],
        "googlenet_ssd": [googlenet_ssd, False]
    }

    if choice not in [None, ""]:
        sch = schedules[choice]
        pingPongPair = PingPongSchedule(sch[0], enable_perm=sch[1])


def getManualHwSchedule():
    global pingPongPair
    return pingPongPair
