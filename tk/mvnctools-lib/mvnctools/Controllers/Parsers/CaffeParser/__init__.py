# #!/usr/bin/env python3

import mvnctools.Controllers.Parsers.CaffeParser.Pooling
import mvnctools.Controllers.Parsers.CaffeParser.Bias
import mvnctools.Controllers.Parsers.CaffeParser.Convolution
import mvnctools.Controllers.Parsers.CaffeParser.Input
import mvnctools.Controllers.Parsers.CaffeParser.ReLU
import mvnctools.Controllers.Parsers.CaffeParser.Concat
import mvnctools.Controllers.Parsers.CaffeParser.Slice
import mvnctools.Controllers.Parsers.CaffeParser.Eltwise
import mvnctools.Controllers.Parsers.CaffeParser.ELU
import mvnctools.Controllers.Parsers.CaffeParser.PReLU
import mvnctools.Controllers.Parsers.CaffeParser.LRN
import mvnctools.Controllers.Parsers.CaffeParser.InnerProduct
import mvnctools.Controllers.Parsers.CaffeParser.Softmax
import mvnctools.Controllers.Parsers.CaffeParser.Sigmoid
import mvnctools.Controllers.Parsers.CaffeParser.BatchNorm
import mvnctools.Controllers.Parsers.CaffeParser.Scale
import mvnctools.Controllers.Parsers.CaffeParser.Reshape
import mvnctools.Controllers.Parsers.CaffeParser.Dropout
import mvnctools.Controllers.Parsers.CaffeParser.permute
import mvnctools.Controllers.Parsers.CaffeParser.Normalize
import mvnctools.Controllers.Parsers.CaffeParser.PriorBox
import mvnctools.Controllers.Parsers.CaffeParser.DetectionOutput
import mvnctools.Controllers.Parsers.CaffeParser.Flatten
import mvnctools.Controllers.Parsers.CaffeParser.Deconvolution
import mvnctools.Controllers.Parsers.CaffeParser.tan_h
import mvnctools.Controllers.Parsers.CaffeParser.crop