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

#!/usr/bin/env python

from collections import OrderedDict

def conf(opMode='SS', memPath='DD', storageOrder='II', streamingConf=('L', 256), rud=0):
    """Define CMX size in KB"""
    return (opMode, memPath, storageOrder, rud, streamingConf)

googlenet = OrderedDict([
    # Init
    ('conv1/7x7_s2',            conf('SS', 'DD', 'PI')),
    ('pool1/3x3_s2',            conf('SX', 'DD', 'II')),
    ('conv2/3x3_reduce',        conf('SS', 'DD', 'II')),
    ('conv2/3x3',               conf('SS', 'DD', 'II')),

    # FirstInception
    ('pool2/3x3_s2',            conf('SX', 'DD', 'II')),
    ('inception_3a/1x1',        conf('SS', 'DD', 'II')),
    ('inception_3a/3x3_reduce', conf('SS', 'DD', 'II')),
    ('inception_3a/3x3',        conf('SS', 'DD', 'II')),
    ('inception_3a/5x5_reduce', conf('SS', 'DD', 'II')),
    ('inception_3a/5x5',        conf('SS', 'DD', 'II')),
    ('inception_3a/pool',       conf('SX', 'DD', 'II')),
    ('inception_3a/pool_proj',  conf('SS', 'DD', 'II')),

    # SecondInception
    ('inception_3b/1x1',        conf('SS', 'DD', 'II')),
    ('inception_3b/3x3_reduce', conf('SS', 'DD', 'II')),
    ('inception_3b/3x3',        conf('SS', 'DD', 'II')),
    ('inception_3b/5x5_reduce', conf('SS', 'DD', 'II')),
    ('inception_3b/5x5',        conf('SS', 'DD', 'II')),
    ('inception_3b/pool',       conf('SX', 'DD', 'II')),
    ('inception_3b/pool_proj',  conf('SS', 'DD', 'II')),

    # ThirdInception
    ('pool3/3x3_s2',            conf('SX', 'DD', 'II')),
    ('inception_4a/1x1',        conf('XX', 'DL', 'II')),
    ('inception_4a/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_4a/3x3',        conf('XX', 'RL', 'II')),
    ('inception_4a/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_4a/5x5',        conf('XX', 'RL', 'II')),
    ('inception_4a/pool',       conf('XX', 'DR', 'II')),
    ('inception_4a/pool_proj',  conf('XX', 'RLU', 'II')),

    # FourthInception
    ('inception_4b/1x1',        conf('XX', 'DL', 'II')),
    ('inception_4b/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_4b/3x3',        conf('XX', 'RL', 'II')),
    ('inception_4b/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_4b/5x5',        conf('XX', 'RL', 'II')),
    ('inception_4b/pool',       conf('XX', 'DR', 'II')),
    ('inception_4b/pool_proj',  conf('XX', 'RLU', 'II')),

    # FifthInception
    ('inception_4c/1x1',        conf('XX', 'DL', 'II')),
    ('inception_4c/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_4c/3x3',        conf('XX', 'RL', 'II')),
    ('inception_4c/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_4c/5x5',        conf('XX', 'RL', 'II')),
    ('inception_4c/pool',       conf('XX', 'DR', 'II')),
    ('inception_4c/pool_proj',  conf('XX', 'RLU', 'II')),

    # SixthInception
    ('inception_4d/1x1',        conf('XX', 'DL', 'II')),
    ('inception_4d/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_4d/3x3',        conf('XX', 'RL', 'II')),
    ('inception_4d/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_4d/5x5',        conf('XX', 'RL', 'II')),
    ('inception_4d/pool',       conf('XX', 'DR', 'II')),
    ('inception_4d/pool_proj',  conf('XX', 'RLU', 'II')),

    # SeventhInception
    ('inception_4e/1x1',        conf('SS', 'DD', 'II')),
    ('inception_4e/3x3_reduce', conf('SS', 'DD', 'II')),
    ('inception_4e/3x3',        conf('SS', 'DD', 'II')),
    ('inception_4e/5x5_reduce', conf('SS', 'DD', 'II')),
    ('inception_4e/5x5',        conf('SS', 'DD', 'II')),
    ('inception_4e/pool',       conf('SX', 'DD', 'II')),
    ('inception_4e/pool_proj',  conf('SS', 'DD', 'II')),

    # EighthInception
    ('pool4/3x3_s2',            conf('SX', 'DD', 'II')),
    ('inception_5a/1x1',        conf('XX', 'DL', 'II')),
    ('inception_5a/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_5a/3x3',        conf('XX', 'RL', 'II')),
    ('inception_5a/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_5a/5x5',        conf('XX', 'RL', 'II')),
    ('inception_5a/pool',       conf('XX', 'DR', 'II')),
    ('inception_5a/pool_proj',  conf('XX', 'RLU', 'II')),

    # NinthInception
    ('inception_5b/1x1',        conf('XX', 'DL', 'II')),
    ('inception_5b/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_5b/3x3',        conf('XX', 'RL', 'II')),
    ('inception_5b/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_5b/5x5',        conf('XX', 'RL', 'II')),
    ('inception_5b/pool',       conf('XX', 'DR', 'II')),
    ('inception_5b/pool_proj',  conf('XX', 'RLU', 'II')),

    # End
    ('pool5/7x7_s1',            conf('XX', 'DL', 'II')),
    ('loss3/classifier',        conf('XX', 'LRU', 'II')),
])

googlenet_best = OrderedDict([
    # Init
    ('conv1/7x7_s2',            conf('SS', 'DD', 'PI')),
    ('pool1/3x3_s2',            conf('XX', 'DL', 'II')),
    ('conv2/3x3_reduce',        conf('SS', 'LD', 'II', ('R', 60))),

    # Select this:
    ('conv2/3x3_a',             conf('XX', 'DR', 'II')),
    ('pool2/3x3_s2_a',          conf('SX', 'RD', 'II', ('L', 200))),
    ('conv2/3x3_b',             conf('XX', 'DR', 'II')),
    ('pool2/3x3_s2_b',          conf('SX', 'RD', 'II', ('L', 200))),
    ('conv2/3x3_c',             conf('XX', 'DR', 'II')),
    ('pool2/3x3_s2_c',          conf('SX', 'RD', 'II', ('L', 200))),

    # # Or this
    # ('conv2/3x3',               conf('SS', 'DD', 'II')),
    # ('pool2/3x3_s2',            conf('SS', 'DD', 'II')),

    # FirstInception
    ('inception_3a/1x1',        conf('SX', 'DD', 'II', ('L', 256))),
    ('inception_3a/3x3_reduce', conf('XX', 'DR', 'II')),
    ('inception_3a/3x3',        conf('SX', 'RD', 'II', ('L', 256))),
    ('inception_3a/5x5_reduce', conf('XX', 'DR', 'II')),
    ('inception_3a/5x5',        conf('SX', 'RD', 'II', ('L', 256))),
    ('inception_3a/pool',       conf('XX', 'DR', 'II')),
    ('inception_3a/pool_proj',  conf('SX', 'RD', 'II', ('L', 256))),

    # SecondInception
    ('inception_3b/pool',       conf('SS', 'DD', 'II', ('L', 256))),
    ('inception_3b/pool_proj',  conf('XX', 'DL', 'IP')),
    ('pool3/3x3_s2_d',          conf('XX', 'LR', 'PP')),
    ('inception_3b/3x3_reduce', conf('SS', 'DD', 'II', ('L', 256))),
    ('inception_3b/3x3',        conf('XX', 'DL', 'IP')),
    ('pool3/3x3_s2_b',          conf('XX', 'LR', 'PP')),
    ('inception_3b/5x5_reduce', conf('XX', 'DL', 'II')),
    ('inception_3b/5x5',        conf('XX', 'Ll', 'IP')),
    ('pool3/3x3_s2_c',          conf('XX', 'lR', 'PP')),
    ('inception_3b/1x1',        conf('XX', 'DL', 'II')),
    ('pool3/3x3_s2_a',          conf('XX', 'LR', 'IP')),

    # ThirdInception
    ('inception_4a/1x1',        conf('XX', 'RL', 'PP')),
    ('inception_4a/3x3_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_4a/3x3',        conf('XX', 'rL', 'PP')),
    ('inception_4a/5x5_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_4a/5x5',        conf('XX', 'rL', 'PP')),
    ('inception_4a/pool',       conf('XX', 'RrO', 'PP')),
    ('inception_4a/pool_proj',  conf('XX', 'rL', 'PP')),

    # FourthInception
    ('inception_4b/pool',       conf('XX', 'Ll', 'PP')),
    ('inception_4b/pool_proj',  conf('XX', 'lR', 'PP')),
    ('inception_4b/5x5_reduce', conf('XX', 'Ll', 'PP')),
    ('inception_4b/5x5',        conf('XX', 'lR', 'PP')),
    ('inception_4b/3x3_reduce', conf('XX', 'Ll', 'PP')),
    ('inception_4b/3x3',        conf('XX', 'lR', 'PP')),
    ('inception_4b/1x1',        conf('XX', 'LR', 'PP')),

    # FifthInception
    ('inception_4c/1x1',        conf('XX', 'RL', 'PP')),
    ('inception_4c/3x3_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_4c/3x3',        conf('XX', 'rL', 'PP')),
    ('inception_4c/5x5_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_4c/5x5',        conf('XX', 'rL', 'PP')),
    ('inception_4c/pool',       conf('XX', 'RrO', 'PP')),
    ('inception_4c/pool_proj',  conf('XX', 'rL', 'PP')),

    # SixthInception
    ('inception_4d/pool',       conf('XX', 'Ll', 'PP')),
    ('inception_4d/pool_proj',  conf('XX', 'lR', 'PP')),
    ('inception_4d/5x5_reduce', conf('XX', 'Ll', 'PP')),
    ('inception_4d/5x5',        conf('XX', 'lR', 'PP')),
    ('inception_4d/3x3_reduce', conf('XX', 'Ll', 'PP')),
    ('inception_4d/3x3',        conf('XX', 'lR', 'PP')),
    ('inception_4d/1x1',        conf('XX', 'LR', 'PP')),

    # SeventhInception
    ('inception_4e/1x1',        conf('XX', 'Rr', 'PP')),
    ('pool4/3x3_s2_a',          conf('XX', 'rL', 'PP')),
    ('inception_4e/3x3_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_4e/3x3',        conf('XX', 'rl', 'PP')),
    ('pool4/3x3_s2_b',          conf('XX', 'lL', 'PP')),
    ('inception_4e/5x5_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_4e/5x5',        conf('XX', 'rl', 'PP')),
    ('pool4/3x3_s2_c',          conf('XX', 'lL', 'PP')),
    ('inception_4e/pool',       conf('XX', 'RrO', 'PP')),
    ('inception_4e/pool_proj',  conf('XX', 'rl', 'PP')),
    ('pool4/3x3_s2_d',          conf('XX', 'lL', 'PP')),

    # EighthInception
    ('inception_5a/pool',       conf('XX', 'Ll', 'PP')),
    ('inception_5a/pool_proj',  conf('XX', 'lR', 'PP')),
    ('inception_5a/5x5_reduce', conf('XX', 'Ll', 'PP')),
    ('inception_5a/5x5',        conf('XX', 'lR', 'PP')),
    ('inception_5a/3x3_reduce', conf('XX', 'Ll', 'PP')),
    ('inception_5a/3x3',        conf('XX', 'lR', 'PP')),
    ('inception_5a/1x1',        conf('XX', 'LR', 'PP')),

    # NinthInception
    ('inception_5b/1x1',        conf('XX', 'RL', 'PP')),
    ('inception_5b/3x3_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_5b/3x3',        conf('XX', 'rL', 'PP')),
    ('inception_5b/5x5_reduce', conf('XX', 'Rr', 'PP')),
    ('inception_5b/5x5',        conf('XX', 'rL', 'PP')),
    ('inception_5b/pool',       conf('XX', 'RrO', 'PP')),
    ('inception_5b/pool_proj',  conf('XX', 'rL', 'PP')),

    # End
    ('pool5/7x7_s1',            conf('XX', 'LR', 'PI')),
    ('loss3/classifier',        conf('XX', 'RLU', 'II')),
])

yolo_tiny = OrderedDict([
    # ('conv1',     conf('SS', 'DD', 'PI')),
    ('pool1',     conf('SS', 'DD', 'PI')),

    # ('conv2',     conf('SS', 'DD', 'II')),
    ('pool2',     conf('SS', 'DD', 'II')),

    # ('conv3',     conf('SS', 'DD', 'II')),
    ('pool3',     conf('SS', 'DD', 'II')),

    # ('conv4',     conf('SS', 'DD', 'II')),
    ('pool4',     conf('SS', 'DD', 'II')),

    # ('conv5',     conf('SS', 'DD', 'II')),
    ('pool5',     conf('SS', 'DD', 'II')),

    # ('conv6',     conf('SS', 'DD', 'II')),
    ('pool6',     conf('SS', 'DD', 'II')),

    # ('conv7',     conf('SS', 'DD', 'II')),
    ('scale7',     conf('SS', 'DD', 'II')),

    # ('conv8',     conf('SS', 'DD', 'II')),
    ('scale8',     conf('SS', 'DD', 'II')),

    ('fc9',       conf('SS', 'DD', 'II')),
])

vgg16 = OrderedDict([
    ('conv1_1',     conf('SS', 'DD', 'PI')),
    ('conv1_2',     conf('SS', 'DD', 'II')),
    ('pool1',       conf('SS', 'DD', 'II')),

    ('conv2_1',     conf('SS', 'DD', 'II')),
    ('conv2_2',     conf('SS', 'DD', 'II')),
    ('pool2',       conf('SS', 'DD', 'II')),

    ('conv3_1',     conf('SS', 'DD', 'II')),
    ('conv3_2',     conf('SS', 'DD', 'II')),
    ('conv3_3',     conf('SS', 'DD', 'II')),
    ('pool3',       conf('SS', 'DD', 'II')),

    ('conv4_1',     conf('SS', 'DD', 'II')),
    ('conv4_2',     conf('SS', 'DD', 'II')),
    ('conv4_3',     conf('SS', 'DD', 'II')),
    ('pool4',       conf('SS', 'DD', 'II')),

    ('conv5_1',     conf('SS', 'DD', 'II')),
    ('conv5_2',     conf('SS', 'DD', 'II')),
    ('conv5_3',     conf('SS', 'DD', 'II')),
    ('pool5',       conf('SS', 'DD', 'II')),

    ('fc6',         conf('SS', 'DD', 'II')),
    ('fc7',         conf('XX', 'DD', 'II')),
    ('fc8',         conf('XX', 'DD', 'II')),
])


googlenet_ssd = OrderedDict([
    ('copy_data',                               conf('XX', 'DD', 'PI')),
    ('conv1/7x7_s2',                            conf('SS', 'DD', 'II')),
    ('pool1/3x3_s2',                            conf('SS', 'DD', 'II')),
    ('pool1/norm1',                             conf('XX', 'DD', 'II')),
    ('conv2/3x3_reduce',                        conf('SS', 'DD', 'II')),
    ('conv2/3x3',                               conf('SS', 'DD', 'II')),
    ('conv2/norm2',                             conf('XX', 'DD', 'II')),
    ('pool2/3x3_s2',                            conf('SS', 'DD', 'II')),
    ('inception_3a/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_3a/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_3a/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_3a/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_3a/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_3a/pool',                       conf('XX', 'DR', 'II')),
    ('inception_3a/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('inception_3b/1x1',                        conf('SS', 'DD', 'II')),
    ('inception_3b/3x3_reduce',                 conf('SS', 'DD', 'II')),
    ('inception_3b/3x3',                        conf('SS', 'DD', 'II')),
    ('inception_3b/5x5_reduce',                 conf('SS', 'DD', 'II')),
    ('inception_3b/5x5',                        conf('SS', 'DD', 'II')),
    ('inception_3b/pool',                       conf('SS', 'DD', 'II')),
    ('inception_3b/pool_proj',                  conf('SS', 'DD', 'II')),
    ('pool3/3x3_s2',                            conf('SS', 'DD', 'II')),
    ('inception_4a/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_4a/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4a/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_4a/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4a/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_4a/pool',                       conf('XX', 'DR', 'II')),
    ('inception_4a/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('inception_4b/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_4b/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4b/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_4b/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4b/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_4b/pool',                       conf('XX', 'DR', 'II')),
    ('inception_4b/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('inception_4c/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_4c/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4c/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_4c/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4c/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_4c/pool',                       conf('XX', 'DR', 'II')),
    ('inception_4c/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('inception_4d/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_4d/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4d/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_4d/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4d/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_4d/pool',                       conf('XX', 'DR', 'II')),
    ('inception_4d/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('inception_4e/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_4e/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4e/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_4e/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_4e/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_4e/pool',                       conf('XX', 'DR', 'II')),
    ('inception_4e/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('pool4/3x3_s2',                            conf('SS', 'DD', 'II')),
    ('inception_5a/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_5a/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_5a/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_5a/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_5a/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_5a/pool',                       conf('XX', 'DR', 'II')),
    ('inception_5a/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('inception_5b/1x1',                        conf('XX', 'DL', 'II')),
    ('inception_5b/3x3_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_5b/3x3',                        conf('XX', 'RL', 'II')),
    ('inception_5b/5x5_reduce',                 conf('XX', 'DR', 'II')),
    ('inception_5b/5x5',                        conf('XX', 'RL', 'II')),
    ('inception_5b/pool',                       conf('XX', 'DR', 'II')),
    ('inception_5b/pool_proj',                  conf('XX', 'RLU', 'II')),
    ('conv6_1',                                 conf('SS', 'DD', 'II')),
    ('conv6_2',                                 conf('SS', 'DD', 'II')),
    ('conv7_1',                                 conf('SS', 'DD', 'II')),
    ('conv7_2',                                 conf('SS', 'DD', 'II')),
    ('conv8_1',                                 conf('SS', 'DD', 'II')),
    ('conv8_2',                                 conf('SS', 'DD', 'II')),
    ('inception_3b/output_norm',                conf('XX', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_loc',       conf('SS', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_loc_perm',  conf('XX', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_loc_flat',  conf('XX', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_conf',      conf('SS', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_conf_perm', conf('XX', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_conf_flat', conf('XX', 'DD', 'II')),
    ('inception_3b/output_norm_mbox_priorbox',  conf('XX', 'DD', 'II')),
    ('inception_4e/output_mbox_loc',            conf('SS', 'DD', 'II')),
    ('inception_4e/output_mbox_loc_perm',       conf('XX', 'DD', 'II')),
    ('inception_4e/output_mbox_loc_flat',       conf('XX', 'DD', 'II')),
    ('inception_4e/output_mbox_conf',           conf('SS', 'DD', 'II')),
    ('inception_4e/output_mbox_conf_perm',      conf('XX', 'DD', 'II')),
    ('inception_4e/output_mbox_conf_flat',      conf('XX', 'DD', 'II')),
    ('inception_4e/output_mbox_priorbox',       conf('XX', 'DD', 'II')),
    ('conv6_2_mbox_loc',                        conf('SS', 'DD', 'II')),
    ('conv6_2_mbox_loc_perm',                   conf('XX', 'DD', 'II')),
    ('conv6_2_mbox_loc_flat',                   conf('XX', 'DD', 'II')),
    ('conv6_2_mbox_conf',                       conf('SS', 'DD', 'II')),
    ('conv6_2_mbox_conf_perm',                  conf('XX', 'DD', 'II')),
    ('conv6_2_mbox_conf_flat',                  conf('XX', 'DD', 'II')),
    ('conv6_2_mbox_priorbox',                   conf('XX', 'DD', 'II')),
    ('conv7_2_mbox_loc',                        conf('SS', 'DD', 'II')),
    ('conv7_2_mbox_loc_perm',                   conf('XX', 'DD', 'II')),
    ('conv7_2_mbox_loc_flat',                   conf('XX', 'DD', 'II')),
    ('conv7_2_mbox_conf',                       conf('SS', 'DD', 'II')),
    ('conv7_2_mbox_conf_perm',                  conf('XX', 'DD', 'II')),
    ('conv7_2_mbox_conf_flat',                  conf('XX', 'DD', 'II')),
    ('conv7_2_mbox_priorbox',                   conf('XX', 'DD', 'II')),
    ('conv8_2_mbox_loc',                        conf('SS', 'DD', 'II')),
    ('conv8_2_mbox_loc_perm',                   conf('XX', 'DD', 'II')),
    ('conv8_2_mbox_loc_flat',                   conf('XX', 'DD', 'II')),
    ('conv8_2_mbox_conf',                       conf('SS', 'DD', 'II')),
    ('conv8_2_mbox_conf_perm',                  conf('XX', 'DD', 'II')),
    ('conv8_2_mbox_conf_flat',                  conf('XX', 'DD', 'II')),
    ('conv8_2_mbox_priorbox',                   conf('XX', 'DD', 'II')),
    ('mbox_conf_reshape',                       conf('XX', 'DD', 'II')),
    ('mbox_conf_softmax',                       conf('XX', 'DD', 'II')),
    ('mbox_conf_flatten',                       conf('XX', 'DD', 'II')),
    ('detection_out',                           conf('XX', 'DD', 'II'))
])