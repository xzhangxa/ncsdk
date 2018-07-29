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

from ctypes import c_uint32
from mvnctools.Controllers.EnumController import throw_error
from mvnctools.Models.EnumDeclarations import ErrorTable
from enum import Enum
import numpy as np

from collections import OrderedDict

HEX = '0x{:08X}'



class HwDescOp(Enum):
    """
    These are the enums used in the hardware to identify layers.
    """
    convolution = 0
    convolution_with_pooling = 1
    fully_connected_convolution = 2
    pooling_only = 4

class SerializedDescriptor:
    def __init__(self, type_):
        self.layout = OrderedDict()

        descriptor_header = OrderedDict([
            # Line 0
            ("NextDesc", [32, None, HEX]),
            ("Type", [3, None]),
            ("mode", [3, None]),
            ("rsvd_00", [2, None]),
            ("id", [4, None]),
            ("it", [4, None]),
            ("cm", [3, None]),
            ("dm", [1, None]),
            ("disaint", [1, None]),
            ("rsvd_02", [11, None]),
        ])
        self.layout.update(descriptor_header)

        if type_ == "Conv":
            main_layout = OrderedDict([
                # Line 1
                ("iDimY-1", [12, None]),
                ("rsvd_10", [4, None]),
                ("iDimX-1", [12, None]),
                ("rsvd_11", [4, None]),
                ("iChans-1", [11, None]),
                ("rsvd_12", [5, None]),
                ("oChans-1", [11, None]),
                ("rsvd_13", [3, None]),
                ("interleaved", [2, None]),

                # Line 2
                ("ChRamBlk-1", [11, None]),
                ("rsvd_20", [5, None]),
                ("stride", [4, None]),
                ("rsvd_21", [12, None]),
                ("InFw-1", [4, None]),
                ("InFh-1", [4, None]),
                ("rsvd_22", [19, None]),
                ("PadType", [4, None]),
                ("PadEnable", [1, None]),

                # Line 3
                ("poolEn", [1, None]),
                ("rsvd7", [15, None]),
                ("poolKernelHeight-1", [8, None]),
                ("poolKernelWidth-1", [8, None]),
                ("avgPoolX", [16, None]),
                ("rsvd8", [15, None]),
                ("poolType", [1, None]),

                #Line 4
                ("dataBaseAddr", [32, None, HEX]),
                ("t0", [10, None]),
                ("a0", [10, None]),
                ("a1", [10, None]),
                ("reluXEn", [1, None]),
                ("reluEn", [1, None]),

                # Line 5
                ("dataChStr", [32, None]),
                ("dataLnStr", [32, None]),

                # Line 6
                ("coeffBaseAddr", [32, None, HEX]),
                ("coeffChStrOut", [32, None]),

                # Line 7
                ("coeffChStrIn", [32, None]),
                ("outLnStr", [32, None]),

                # Line 8
                ("outBaseAddr", [32, None, HEX]),
                ("outChStr", [32, None]),

                # Line 9
                ("localLs", [9, None]),
                ("rsvd_90", [7, None]),
                ("localCs", [13, None]),
                ("rsvd_91", [3, None]),
                ("linesPerCh-1", [9, None]),
                ("rsvd_92", [22, None]),
                ("rud", [1, None]),

                # Line 10
                ("minLines-1", [9, None]),
                ("rsvd_A0", [23, None]),
                ("coeffLpb-1", [8, None]),
                ("css-1", [8, None]),
                ("outputX", [12, None]),
                ("rsvd_A1", [4, None]),

                # Line 11
                ("biasBaseAddr", [32, None, HEX]),
                ("scaleBaseAddr", [32, None, HEX]),
            ])

        elif type_ == "Pool":
            main_layout = OrderedDict([

                # Line 1
                ("iDimY-1", [12, None]),
                ("rsvd_10", [4, None]),
                ("iDimX-1", [12, None]),
                ("rsvd_11", [4, None]),
                ("iChans-1", [11, None]),
                ("rsvd_12", [5, None]),
                ("oChans-1", [11, None]),
                ("rsvd_13", [3, None]),
                ("interleaved", [2, None]),

                # Line 2
                ("ChRamBlk-1", [11, None]),
                ("rsvd_20", [5, None]),
                ("stride", [4, None]),
                ("rsvd_21", [12, None]),
                ("InFw-1", [4, None]),
                ("InFh-1", [4, None]),
                ("rsvd_22", [19, None]),
                ("PadType", [4, None]),
                ("PadEnable", [1, None]),

                # Line 3
                ("poolEn", [1, None]),
                ("rsvd7", [15, None]),
                ("poolKernelHeight-1", [8, None]),
                ("poolKernelWidth-1", [8, None]),
                ("avgPoolX", [16, None]),
                ("rsvd8", [15, None]),
                ("poolType", [1, None]),

                # Line 4
                ("dataBaseAddr", [32, None, HEX]),
                ("t0", [10, None]),
                ("a0", [10, None]),
                ("a1", [10, None]),
                ("reluXEn", [1, None]),
                ("reluEn", [1, None]),

                # Line 5
                ("dataChStr", [32, None]),
                ("dataLnStr", [32, None]),

                # Line 6
                ("rsvd_60", [32, None]),
                ("rsvd_61", [32, None]),

                # Line 7
                ("rsvd_70", [32, None]),
                ("outLnStr", [32, None]),

                # Line 8
                ("outBaseAddr", [32, None, HEX]),
                ("outChStr", [32, None]),

                # Line 9
                ("localLs", [9, None]),
                ("rsvd_90", [7, None]),
                ("localCs", [13, None]),
                ("rsvd_91", [3, None]),
                ("linesPerCh-1", [9, None]),
                ("rsvd_92", [22, None]),
                ("rud", [1, None]),

                # Line 10
                ("minLines-1", [9, None]),
                ("rsvd_A0", [23, None]),
                ("rsvd_A1", [8, None]),
                ("rsvd_A2", [8, None]),
                ("outputX", [12, None]),
                ("rsvd_A3", [4, None]),

                # Line 11
                ("biasBaseAddr", [32, None, HEX]),
                ("scaleBaseAddr", [32, None, HEX]),
            ])

        elif type_ == "FCL":
            main_layout = OrderedDict([

                # Line 1
                ("iDimX-1",  [12, None]),      # InputWidth
                ("rsvd0",    [20, None]),
                ("iChans-1", [8, None]),     # Vectors
                ("rsvd1",    [8, None]),
                ("oChans-1", [8, None]),      # vectors2
                ("rsvd2",    [8, None]),

                # Line 2
                ("ChRamBlk-1",   [9, None]),   # dataPerRamBlock
                ("rsvd3",        [23, None]),
                ("rsvd4",        [32, None]),

                # Line 3
                ("rsvd5",                [1, None]),
                ("actualOutChannels",    [8, None]),  # Custom Info
                ("rsvd5_",                [23, None]),
                ("X",                    [16, None]),
                ("rsvd6",                [16, None]),

                #Line 4
                ("dataBaseAddr",     [32, None, HEX]),
                ("t0",               [10, None]),
                ("a0",               [10, None]),
                ("a1",               [10, None]),
                ("reluXEn",          [1, None]),
                ("reluEn",           [1, None]),

                # Line 5
                ("dataChStr", [32, None]),
                ("dataLnStr", [32, None]),

                # Line 6
                ("vecBaseAddr", [32, None, HEX]),
                ("vecStrOut", [32, None, HEX]),

                # Line 7
                ("vecStrIn", [32, None]),
                ("outLnStr", [32, None]),

                # Line 8
                ("outBaseAddr", [32, None, HEX]),
                ("outChStr", [32, None]),

                # Line 9
                ("localLs", [9, None]),
                ("rsvd7", [7, None]),
                ("localBs", [13, None]),
                ("rsvd8", [3, None]),
                ("rsvd9", [31, None]),
                ("rud", [1, None]),

                # Line 10
                ("rsvd10", [16, None]),
                ("Acc", [1, None]),
                ("rsvd11", [15, None]),
                ("vecLPB-1", [8, None]),
                ("rsvd12", [8, None]),
                ("outputX", [12, None]),
                ("rsvd12_", [4, None]),

                # Line 11
                ("biasBaseAddr", [32, None, HEX]),
                ("scaleBaseAddr", [32, None, HEX]),
            ])
        else:
            assert 0, "Invalid Descriptor."

        self.layout.update(main_layout)

        pallete_fields = OrderedDict([
            # Line 12
            ("p0", [16, None]),
            ("p1", [16, None]),
            ("p2", [16, None]),
            ("p3", [16, None]),

            ("p4", [16, None]),
            ("p5", [16, None]),
            ("p6", [16, None]),
            ("p7", [16, None]),

            ("p8", [16, None]),
            ("p9", [16, None]),
            ("pA", [16, None]),
            ("pB", [16, None]),

            ("pC", [16, None]),
            ("pD", [16, None]),
            ("pE", [16, None]),
            ("pF", [16, None]),
        ])

        self.layout.update(pallete_fields)

    def set_field(self, field, value):
        # print(field, value)
        assert field in self.layout, "No such field in Descriptor"
        assertBitSize(value, self.layout[field][0])
        self.layout[field][1] = value

    def set_pallete(self, arr_16_elements):
        """
        helper function to set the palletized weights. If None, fill with zeroes.
        :param arr_16_elements: ordered 16 element array to populate pallete.
        :return: N/A
        """

        if arr_16_elements == None:
            arr_16_elements = [0]*16
        else:
            assert len(arr_16_elements) == 16, "Pallete not fully set."
        self.set_field("p0", arr_16_elements[0])
        self.set_field("p1", arr_16_elements[1])
        self.set_field("p2", arr_16_elements[2])
        self.set_field("p3", arr_16_elements[3])
        self.set_field("p4", arr_16_elements[4])
        self.set_field("p5", arr_16_elements[5])
        self.set_field("p6", arr_16_elements[6])
        self.set_field("p7", arr_16_elements[7])
        self.set_field("p8", arr_16_elements[8])
        self.set_field("p9", arr_16_elements[9])
        self.set_field("pA", arr_16_elements[10])
        self.set_field("pB", arr_16_elements[11])
        self.set_field("pC", arr_16_elements[12])
        self.set_field("pD", arr_16_elements[13])
        self.set_field("pE", arr_16_elements[14])
        self.set_field("pF", arr_16_elements[15])


    def print(self):
        prev_line = -1
        this_line = 0
        bit_count = 0
        for x in self.layout:
            if prev_line != this_line:
                # New Line
                print("XLine " + str(this_line), end=": ")
                prev_line = this_line
                bit_count = 0
            if len(self.layout[x]) == 3:
                print(x, self.layout[x][2].format(self.layout[x][1]), end='. ')
            else:
                print(x, self.layout[x][1],end='. ')
            bit_count += self.layout[x][0]
            if bit_count >= 64:
                # End of Line
                this_line+=1
                print("")

        print("")   # Newline

    def deserialize(self, desc):
        """
        Takes in a hardware descriptor and creates an instance of this class with it.
        Currently only supports Conv.

        :param desc: Descriptor, can include 0x and whitespace.
        :return: N/A
        """
        # Clean Descriptor
        desc = desc.replace("0x","")
        desc = ''.join(desc.split()) # Strip whitespace characters

        lines = []

        x = 0
        step = 8
        while x < len(desc):                            # Each 'half-line' is 32 bits, or 8 hex values.
            hex_ = desc[x:x+step]                       # We iterate over each of these half-lines.
            bin_ = bin(int(hex_,16))[2:].zfill(step*4)  # Convert to binary.
            lines.append(bin_)                          # and add to a larger list.
            x+=step

        char_count = 0

        for field in self.layout:
            field_bits = self.layout[field][0]          # How many bits required for field.
            idx = (char_count)//32                      # Index of Half-line in line list.
            target_line = lines[idx]
            tmp_cc = char_count - ((char_count//32)*32)
                                                        # We then index into the half-line from the other direction.
            end = 32-tmp_cc
            start = 32-(tmp_cc + field_bits)

            cut_value = target_line[start:end].zfill(field_bits)      # Strip the part we want and pad with 0s if required.
            char_count += field_bits                    # Increment pointer

            self.set_field(field, int(cut_value,2))     # Populate field


    def serialize(self):
        lines = self.serialize_lines()
        bk = []
        for x in lines:
            bk.append(c_uint32(x))

        return bk


    def serialize_lines(self, debug=False):
        prev_line = -1
        this_line = 0
        bit_count = 0

        cnt = []
        dsc = 0

        for index, x in enumerate(self.layout):
            if prev_line != this_line:
                # New Line
                prev_line = this_line
                bit_count = 0
                dsc = 0

            if self.layout[x][1] is None and "rsvd" in x:
                self.layout[x][1] = 0
                if debug:
                    print("Warning: Reserved Field Defaulted to Zero")
            elif self.layout[x][1] is None:
                print("Error: Required Field not populated:", x)
                quit()

            dsc += (self.layout[x][1] << bit_count)
            bit_count += self.layout[x][0]

            if bit_count >= 32 or (index == len(self.layout) - 1):
                # End of Line
                this_line+=1
                cnt.extend([dsc])

        return cnt


def assertBitSize(value, allowed_bits):
    """
    Ensures that when written to a field, does not overflow boundaries
    :param value: The value we want to enter into the space
    :param allowed_bits: size of the space for this field, in bits.
    :return: N/A
    """

    # print("Values allowed 0-"+str(2**(allowed_bits)), ". Value: ", value)
    if(type(value) not in [int, bool, np.int64, np.uint16, np.uint32, np.int32, np.uint64, np.int16]):
        print(type(value))
        throw_error(ErrorTable.HardwareConfigurationError, "field is not int")
    if 2**(allowed_bits) <= value:
        throw_error(ErrorTable.HardwareConfigurationError,"field overflow")
    if 0 > value:
        throw_error(ErrorTable.HardwareConfigurationError,"field underflow")
