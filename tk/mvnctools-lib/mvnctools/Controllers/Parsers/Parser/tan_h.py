#!/usr/bin/env python3

from mvnctools.Controllers.TensorFormat import TensorFormat
import mvnctools.Models.Layouts as Layouts
from .Layer import Layer


class TanH(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]