# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

from detection import models


class CustomFCOSConfig(models.FCOSConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet18"
        # self.max_epoch = 36
        # self.lr_decay_stages = [24, 32]
        self.fpn_in_channels = [128, 256, 512]


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "fcos_res18_coco_3x_800size_37dot6_adab0136.pkl"
)
def fcos_res18_coco_3x_800size(**kwargs):
    r"""
    FCOS trained from COCO dataset.
    `"FCOS" <https://arxiv.org/abs/1904.01355>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = CustomFCOSConfig()
    cfg.backbone_pretrained = False
    return models.FCOS(cfg, **kwargs)


Net = models.FCOS
Cfg = CustomFCOSConfig
