#!/usr/bin/env python3

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


from functools import reduce
from copy import copy
import numpy as np
import uuid

def addCoordinates(c1, c2):
    return tuple([sum(x) for x in zip(c1, c2)])

class Tensor():
    ORIGIN = (0, 0, 0, 0)
    def __init__(self, shape):
        # Support only 4D tensors. No particular reason for this
        # limitation (in terms of Tensor support).

        shape = tuple(shape)

        assert(len(shape) == 4)

        self.__ID = uuid.uuid4()
        self.shape = shape
        self.placedTensors = []
        self.enclosure = None
        self.relativeCoordinates = Tensor.ORIGIN
        self.proposed_shapes = []

    def __deepcopy__(self, memo):
        """
            NetworkX performs a deep copy in several operations (e.g. edge contraction).
            We do not want this, this we may end up having multiple copies of an object.
        """
        return copy(self)

    def getShape(self):
        return self.shape

    def setDatatype(self, dtype):
        self.dtype = dtype

    def proposeShape(self, shape):
        self.proposed_shapes.append(shape)

    def getDatatype(self):
        return np.float16
        return self.dtype

    def setLayout(self, layout):
        self.layout = layout

    def getLayout(self):
        return self.layout

    def storePlacedTensor(self, smallerTensor):
        # TODO: Check that there is no intersection with other tensors
        # that have already been placed.

        self.placedTensors.append(smallerTensor)

    def getTopEncloserRecursive(self):
        if self.enclosure:
            enc = self.enclosure.getTopEncloserRecursive()

            return self.enclosure.getTopEncloserRecursive()
        else:
            return self

    def getTopEncloser(self):
        if self.enclosure:
            return self.getTopEncloserRecursive()
        else:
            return None

    def getAbsolutePosition(self):
        if self.enclosure:
            enclosureCoords = self.enclosure.getAbsolutePosition()
            return addCoordinates(enclosureCoords, self.relativeCoordinates)

        else:
            return self.relativeCoordinates

    def __shapeInCanonicalLayout(self, shape):
        '''For a shape given in a canonical format, convert it to
           according to the (non-canonical) layout provided. The
           result is a new shape whose layout is canonical'''

        assert(len(shape) == len(self.layout))

        permutation = [self.layout.index(i) for i in self.layout]
        newShape = tuple([shape[self.layout[i]] for i in permutation])
        return newShape

    def __getDistanceFromOrigin(self, absoluteCoordinate, topTensorShape):
        def prod(x):
            if x:
                return reduce(lambda a,b: a*b, x, 1)
            else:
                return 1

        absPos = self.__shapeInCanonicalLayout(absoluteCoordinate)
        reverseCanonicalShape = tuple(reversed(self.__shapeInCanonicalLayout(topTensorShape)))

        dist = 0
        for idx, coord in enumerate(reversed(absPos)):
            dist += coord * prod(reverseCanonicalShape[0:idx])

        return dist

    def getDistanceFromOrigin(self):
        topEncloser = self.getTopEncloserRecursive()
        topTensorShape = topEncloser.getShape()
        return self.__getDistanceFromOrigin(self.getAbsolutePosition(), topTensorShape)

    def getStrideInElements(self, axis):
        topEncloser = self.getTopEncloserRecursive()
        topTensorShape = topEncloser.getShape()

        # Increase the dimension in the axis of interest, to
        # ensure we will not fall off the edge.
        increment = [0] * len(self.shape)
        increment[axis] = 1
        increment = tuple(increment)
        topTensorShape = addCoordinates(topTensorShape, increment)

        tensorOrigin = self.getAbsolutePosition()
        tensorOriginNeighbour = addCoordinates(tensorOrigin, increment)

        d1 = self.__getDistanceFromOrigin(tensorOrigin, topTensorShape)
        d2 = self.__getDistanceFromOrigin(tensorOriginNeighbour, topTensorShape)

        return d2 - d1

    def getStrideInBytes(self, axis):
        return np.dtype(self.dtype).itemsize * self.getStrideInElements(axis)

    def place(self, largerTensor, topCornerInLargerTensor):
        """
        Place this tensor inside another.

        arguments:
        @ largerTensor: Tensor which to place this tensor in.
        @ topCornerInLargerTensor: position placed (co-ord)
        """
        # For the placement to be valid, larger and smaller tensors
        # must have the same layout
        assert(self.getLayout() == largerTensor.getLayout())

        # Check that the current tensor fits into the larger tensor
        # Start and End element of the larger tensor follows [, ) convention.
        # assert (largerTensor.getShape() >= tuple([sum(x) for x in zip(self.getShape(),topCornerInLargerTensor)]))

        self.relativeCoordinates = topCornerInLargerTensor
        largerTensor.storePlacedTensor(self)
        assert(self.enclosure is None)
        self.enclosure = largerTensor

    def resolve(self):
        rt = ResolvedTensor(self)
        return rt

    def pprint(self, recursively=False):
        UNDEF = "<undefined>"
        layout = self.layout if hasattr(self, "layout") else UNDEF
        dist = self.getDistanceFromOrigin() if hasattr(self, "layout") else UNDEF
        """
        Pretty Print the Buffer
        """
        print("""
                X       ID = {}
                X
                X       Shape = {}
                X       Layout = {}
                X       Distance from Origin = {}
               X X
             XX   XX
          XXX       XXX
        XX    Tensor   XX
            """.format(self.ID, self.shape, layout, dist)
              )

        if recursively:
            if self.enclosure:
                print()
                print('Enclosing Tensor:')
                self.enclosure.pprint(recursively)

    # ID must be read-only
    @property
    def ID(self):
        return self.__ID

    def reshape(self, shape):
        # Check that the reshape is acceptable by numpy
        self.shape = shape

    def reorder(self, layout):
        """
        """
        self.shape = tuple([self.shape[i] for i in layout])


class UnpopulatedTensor(Tensor):
    def place(self, largerTensor, topCornerInLargerTensor):
        assert(isinstance(largerTensor, UnpopulatedTensor))
        super().place(largerTensor, topCornerInLargerTensor)

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

class PopulatedTensor(Tensor):
    def __init__(self, data, isFirstAxisBatchNumber=False):
        # Check that the data being loaded matches the layout
        self.isFirstAxisBatchNumber = isFirstAxisBatchNumber
        self.dtype = data.dtype
        self.data = data

        # Canonicalize shape to match the dimension of origin
        origShape = self.data.shape
        assert(len(origShape) <= len(Tensor.ORIGIN))

        diff = len(Tensor.ORIGIN) - len(origShape)
        if isFirstAxisBatchNumber:
            newShape = tuple(origShape[0]) + tuple([1] * (diff-1)) + origShape[1:]
        else:
            newShape = tuple([1] * diff) + origShape

        self.data = self.data.reshape(newShape)

        # Set the canonical layout
        self.setLayout(tuple(range(len(Tensor.ORIGIN))))

        super().__init__(self.data.shape)

    def place(self, largerTensor, topCornerInLargerTensor):
        assert(isinstance(largerTensor, UnpopulatedTensor))
        super().place(largerTensor, topCornerInLargerTensor)

    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        self.shape = shape

    # def __canonicalizeShape(self, shape):

class ResolvedTensor():
    """
        A Resolved Tensor is an read-only representation of a tensor
    """
    def __init__(self, tensor):
        self.__topID = tensor.getTopEncloserRecursive().ID
        self.__original_tensor = tensor
        self.__dimensions = tensor.shape
        self.__strides = [tensor.getStrideInBytes(axis) for axis in range(len(tensor.shape))]
        self.__layout = tensor.getLayout()
        self.__dtype = tensor.dtype

        if isinstance(tensor, PopulatedTensor):
            self.__name = None
            self.__opaque = True

            # Apply the layout. Convert from canonical format to the specified one.
            self.__data = tensor.data.transpose(self.__layout)
        else:
            self.__name = tensor.getName()
            self.__opaque = False
            self.__data = np.zeros(tensor.shape)
        self.__local_offset = tensor.getDistanceFromOrigin()

    def getTopEncloserRecursive(self):
        return self.original_tensor.getTopEncloserRecursive()

    # From here on is read-only decorations
    @property
    def original_tensor(self):
        return self.__original_tensor

    @property
    def name(self):
        return self.__name

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def strides(self):
        return self.__strides

    @property
    def layout(self):
        return self.__layout

    @property
    def dtype(self):
        return self.__dtype

    @property
    def data(self):
        return self.__data

    @property
    def local_offset(self):
        return self.__local_offset

    @property
    def topID(self):
        return self.__topID

    @property
    def opaque(self):
        """
            A "transparent" resolved tensor is one that contains positional information only.
            An "opaque" tensor is one that has data inside it.
        """
        return self.__opaque
