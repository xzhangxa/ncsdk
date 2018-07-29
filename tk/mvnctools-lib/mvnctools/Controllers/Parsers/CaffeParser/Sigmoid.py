#!/usr/bin/env python3

from mvnctools.Controllers.Parsers.Parser.Sigmoid import Sigmoid

def load(obj, parsedNetworkObj):
    # This operation supports only one inputTensor and one
    # outputTensor
    assert(len(obj.bottom) == 1)
    assert(len(obj.top) == 1)

    x = Sigmoid(obj.name, obj.bottom, obj.top)

    return [x]