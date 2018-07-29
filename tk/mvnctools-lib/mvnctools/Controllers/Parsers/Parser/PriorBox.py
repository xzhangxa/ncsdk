
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

import numpy as np

from .Layer import Layer
import mvnctools.Models.Layouts as Layouts
from mvnctools.Controllers.TensorFormat import TensorFormat

class PriorBox(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCMInput = TensorFormat(Layouts.NHCW, (1, 2, 3))
        tfCMOutput = TensorFormat(Layouts.NHWC, (2,))
        self.formatPool = [(tfCMInput, tfCMOutput)]

    def load_parameters(self, prior_box_param):
        #TODO prettfy this. Not enough time to do that for the 2.04 release.
        self.params = np.array((
                   len(prior_box_param.min_size),
                   len(prior_box_param.max_size),
                   len(prior_box_param.aspect_ratio),
                   len(prior_box_param.variance),
                   prior_box_param.flip,
                   prior_box_param.clip),
                   dtype=np.dtype("<f4"))
        self.params = np.append(self.params, prior_box_param.min_size[0:])
        self.params = np.append(self.params, prior_box_param.max_size[0:])
        self.params = np.append(self.params, prior_box_param.aspect_ratio[0:])
        self.params = np.append(self.params, prior_box_param.variance[0:])

        if (prior_box_param.HasField("step_w") and
                prior_box_param.HasField("step_h")):
            # We don't check for both step and step_h/step_h being set
            # because caffe should yeld an error before this.
            self.params = np.append(self.params, prior_box_param.step_w)
            self.params = np.append(self.params, prior_box_param.step_h)
        elif (prior_box_param.HasField("step")):
            self.params = np.append(self.params, prior_box_param.step)
            self.params = np.append(self.params, prior_box_param.step)
        else:
            self.params = np.append(self.params, 0)
            self.params = np.append(self.params, 0)
        self.params = np.append(self.params, prior_box_param.offset)
        self.params = self.params.astype(dtype = np.dtype("<f4"))
