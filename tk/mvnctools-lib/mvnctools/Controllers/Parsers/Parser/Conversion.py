#!/usr/bin/env python3

from .Layer import Layer

import mvnctools.Models.Layouts as Layouts
from mvnctools.Controllers.TensorFormat import TensorFormat

class Conversion(Layer):
    def __init__(self, *args):
        super().__init__(*args)
        self.formatPool = getDefaultConversionLayerFormatPool()

    def setLayouts(self, inputLayout, outputLayout):
        self.inputLayout = inputLayout
        self.outputLayout = outputLayout

    def deriveFormatPool(self, format1, format2):
        self.formatPool = [(getTensorFormatFromLayout(format1),
                           getTensorFormatFromLayout(format2))]


def getDefaultConversionLayerFormatPool():

    # Set the supported layouts
    chmin = TensorFormat(Layouts.NHWC, (1, 2, 3))
    pln = TensorFormat(Layouts.NCHW, (1, 2, 3))
    ri = TensorFormat(Layouts.NHCW, (1, 2, 3))

    formatPool = [
        (None, chmin),
        (None, ri),
        (None, pln),
    ]

    return formatPool

def getTensorFormatFromLayout(tensorFormat):

    tfCMConversion = TensorFormat(Layouts.NHWC, (1, 2, 3))
    tfPLConversion = TensorFormat(Layouts.NCHW, (1, 2, 3))
    tfIVConversion = TensorFormat(Layouts.NHCW, (1, 2, 3))

    if tensorFormat.layout == tfCMConversion.layout:
        return tfCMConversion
    elif tensorFormat.layout == tfIVConversion.layout:
        return tfIVConversion
    elif tensorFormat.layout == tfPLConversion.layout:
        return tfPLConversion
