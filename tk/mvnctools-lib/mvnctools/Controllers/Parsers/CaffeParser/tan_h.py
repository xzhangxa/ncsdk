#!/usr/bin/env python3

from mvnctools.Controllers.Parsers.Parser.tan_h import TanH

def load(obj, parsedNetworkObj):

    x = TanH(obj.name, obj.bottom, obj.top)

    return [x]