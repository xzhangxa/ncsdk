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

import collections
import ctypes

import numpy as np
from mvnctools.Controllers.FileIO import *


def byte_size(item):
    """
    :param item: variable to assess
    :return: size of item in bytes when passed to blob file.
    """
    if type(item) is bytes:
        return len(item)
    elif type(item) is np.ndarray:
        return item.flatten().shape[0] * get_numpy_element_byte_size(item) # TODO: Calculate correct bytesize for buffers
    else:
        return ctypes.sizeof(item)


class Value:
    """
    An object representing a singular value
    """
    def __init__(self, val):
        self.val = val

class BinaryData:
    """
    An object representing a contiguous array of data
    """
    def __init__(self, val):
        self.val = val #ctypes.c_uint(99)


class StableLink:
    """
    An offset to another item, relative to this object's position.
    This object does not go through a relocation table.
    """
    def __init__(self, offset):
        self.val = offset


class WeakLink:
    """
    An offset to another item, relative to this object's position.
    This will point to an entry in a relocation table.
    """
    def __init__(self, offset, location):
        """
        :param offset:   Offset from start of buffer
        :param location: Which buffer
        :return:
        """
        self.val = offset
        self.locale = location


class WeakLinkManager:
    def __init__(self):
        self.link_array = []
        self.bss_array = []
        self.blob_array = []
        self.io_array = []

    def new(self, offset, location):
        w = WeakLink(offset, location)
        self.link_array.append(w)
        if location.value == MemoryIndex.blob.value:
            self.blob_array.append(w)
        elif location.value >= MemoryIndex.workbuffer.value:
            # print("This is Work", w.val)
            self.bss_array.append(w)
        else:
            self.io_array.append(w)
            # print("THIS IS NOT BLOB OR WORK", location.value, MemoryIndex.blob.value, MemoryIndex.workbuffer.value)
            pass
        return w

    def index(self, wl):
        if wl in self.blob_array:
            return self.blob_array.index(wl)
        elif wl in self.bss_array:
            return self.bss_array.index(wl)
        else:
            return wl.val.value
            # return -1


class Placeholder:
    """
    An object that cannot be populated right now.
    """
    def __init__(self, val):
        self.val = val

def align_to(number, to_alignment):
    if to_alignment == 0: return  number    # No alignment
    return number + to_alignment - number % to_alignment


class Container:
    """
    An encapsulation object
    """
    def __init__(self, align=0):
        self.attr = collections.OrderedDict()
        self.size = 0
        self.aligned_size = 0
        self.meld_count = 0
        self.align = align

    def push(self, label, item, pre_align=0):
        """
        Add an object to the writable layout
        :param label - human readable description of the item's purpose
        :param item  - item to push
        """
        self.size += byte_size(item.val)
        self.size += byte_size(item.locale) if hasattr(item, 'locale') else 0
        self.aligned_size = align_to(self.size, self.align)
        self.attr[label] = item

    def meld(self, subcontainer):
        """
        Consume another container's objects.
        :param subcontainer - item to meld/consume
        """
        self.meld_count += 1
        self.size += subcontainer.size
        self.aligned_size = align_to(self.size, self.align)

        # Pad so that end is aligned
        for x in range(subcontainer.aligned_size - subcontainer.size):
            subcontainer.attr["PAD"+str(x)] = Value(ctypes.c_uint8(0))

        for label, item in subcontainer.attr.items():
            self.attr[str(self.meld_count) + label] = item  # Prefix the label with a meld index.

    def update(self, label, item):
        """
        Replace a placeholder and put in real values.
        :param label - human readable description of the item's purpose
        :param item  - item to update
        """
        assert label in self.attr, "Cannot change field that does not exist yet."
        assert type(self.attr[label]) is Placeholder, "Can only update placeholder values." + str(type(item)) + str(label)
        self.attr[label] = item

    def print(self, include_filters=None):
        """
        Print each value in the container. Recursive.
        TODO: Exclude Filters
        """
        print("\n_______________________")
        count = 0
        for key, value in self.attr.items():
            if include_filters is not None:
                for f in include_filters:
                    if f in key:
                        print(count, key, ": ", value.val)
            else:
                print(count, key, ": ", value.val)
            count += byte_size(value.val)
        print("_______________________\n")

    def write(self, f):
        for key, value in self.attr.items():
            if isinstance(value, Value) or isinstance(value, StableLink) or isinstance(value, WeakLink):
                if isinstance(value.val, ctypes._SimpleCData) or isinstance(value.val, bytes):
                    f.write(value.val)
                    if hasattr(value,'locale'):
                        f.write(value.locale)
                else:
                    # print("Not a valid Object to write", value.val, type(value.val) )
                    pass
            elif isinstance(value, BinaryData):
                f.write(value.val)
            elif isinstance(value, Container):
                # print("Subcontainer: ", key, value)
                # value.write(f)
                pass
            else:
                assert 0 ,"Unidentified Type" + str(key) +" "+ str(type(value))
