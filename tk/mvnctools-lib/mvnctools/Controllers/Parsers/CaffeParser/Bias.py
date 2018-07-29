#!/usr/bin/env python3

import numpy as np
from mvnctools.Controllers.Parsers.Parser.Bias import Bias


def load(obj, parsedNetworkObj):
    x = Bias(obj.name, obj.bottom, obj.top)

    return [x]
