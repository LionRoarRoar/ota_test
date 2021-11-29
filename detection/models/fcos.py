# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

# import sys
# print(sys.path)
# sys.path.append('../models')
from . import resnet_model as resnet
#
# import resnet_model as resnet
from detection import layers


class FCOS(M.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.anchor_generator = layers.AnchorPointGenerator(
            cfg.num_anchors,
            strides=self.cfg.stride,
            offset=self.cfg.anchor_offset,
        )
        self.point_coder = layers.PointCoder()

        self.in_features = cfg.in_features

        # ----------------------- build backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.backbone_norm), pretrained=cfg.backbone_pretrained
        )
        del bottom_up.fc

        # ----------------------- build FPN ----------------------------- #
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=cfg.fpn_in_features,
            out_channels=cfg.fpn_out_channels,
            norm=cfg.fpn_norm,
            top_block=layers.LastLevelP6P7(
                cfg.fpn_top_in_channel, cfg.fpn_out_channels, cfg.fpn_top_in_feature
            ),
            strides=cfg.fpn_in_strides,
            channels=cfg.fpn_in_channels,
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # ----------------------- build FCOS Head ----------------------- #
        self.head = layers.PointHead(cfg, feature_shapes)

        self.OTA = dict(
            REG_WEIGHT=1.5,
            SINKHORN_EPS=0.1,
            SINKHORN_ITER=50,
            TOP_CANDIDATES=20,
        )

        self.sinkhorn = SinkhornDistance(eps=self.OTA["SINKHORN_EPS"],
                                         max_iter=self.OTA["SINKHORN_ITER"])

    def preprocess_image(self, image):
        padded_image = layers.get_padded_tensor(image, 32, 0.0)
        normed_image = (
                               padded_image
                               - np.array(self.cfg.img_mean, dtype="float32")[None, :, None, None]
                       ) / np.array(self.cfg.img_std, dtype="float32")[None, :, None, None]
        return normed_image

    def forward(self, image, im_info, gt_boxes=None):
        image = self.preprocess_image(image)
        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets, box_ctrness = self.head(features)

        box_logits_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, self.cfg.num_classes)
            for _ in box_logits
        ]
        box_offsets_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 4) for _ in box_offsets
        ]
        box_ctrness_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 1) for _ in box_ctrness
        ]

        anchors_list = self.anchor_generator(features)

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_box_ctrness = F.concat(box_ctrness_list, axis=1)

        if self.training:
            gt_labels, gt_offsets, gt_ctrness = self.get_ground_truth(
                anchors_list, gt_boxes, im_info[:, 4].astype("int32"), all_level_box_logits, all_level_box_offsets, all_level_box_ctrness
            )

            all_level_box_logits = all_level_box_logits.reshape(-1, self.cfg.num_classes)
            all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)
            all_level_box_ctrness = all_level_box_ctrness.flatten()

            gt_labels = gt_labels.flatten()
            gt_offsets = gt_offsets.reshape(-1, 4)
            gt_ctrness = gt_ctrness.flatten()

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.sum()
            sum_ctr = gt_ctrness[fg_mask].sum()
            # add detach() to avoid syncing across ranks in backward
            num_fg = layers.all_reduce_mean(num_fg).detach()
            sum_ctr = layers.all_reduce_mean(sum_ctr).detach()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            loss_cls = layers.sigmoid_focal_loss(
                all_level_box_logits[valid_mask],
                gt_targets[valid_mask],
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum() / F.maximum(num_fg, 1)

            loss_bbox = (
                          layers.iou_loss(
                                all_level_box_offsets[fg_mask],
                                gt_offsets[fg_mask],
                                box_mode="ltrb",
                                loss_type=self.cfg.iou_loss_type,
                            ) * gt_ctrness[fg_mask]
                        ).sum() / F.maximum(sum_ctr, 1e-5) * self.cfg.loss_bbox_weight

            loss_ctr = layers.binary_cross_entropy(
                all_level_box_ctrness[fg_mask],
                gt_ctrness[fg_mask],
            ).sum() / F.maximum(num_fg, 1)

            total = loss_cls + loss_bbox + loss_ctr
            loss_dict = {
                "total_loss": total,
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_ctr": loss_ctr,
            }
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert image.shape[0] == 1

            all_level_anchors = F.concat(anchors_list, axis=0)
            pred_boxes = self.point_coder.decode(
                all_level_anchors, all_level_box_offsets[0]
            )
            pred_boxes = pred_boxes.reshape(-1, 4)

            scale_w = im_info[0, 1] / im_info[0, 3]
            scale_h = im_info[0, 0] / im_info[0, 2]
            pred_boxes = pred_boxes / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0
            )
            clipped_boxes = layers.get_clipped_boxes(
                pred_boxes, im_info[0, 2:4]
            ).reshape(-1, 4)
            pred_score = F.sqrt(
                F.sigmoid(all_level_box_logits) * F.sigmoid(all_level_box_ctrness)
            )[0]
            return pred_score, clipped_boxes

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_num_gts,
                            all_level_box_logits, all_level_box_offsets, all_level_box_ctrness):
        labels_list = []
        offsets_list = []
        ctrness_list = []
        assigned_units = []
        all_level_anchors = F.concat(anchors_list, axis=0)
        for bid in range(batched_gt_boxes.shape[0]):
            gt_boxes = batched_gt_boxes[bid, :batched_num_gts[bid]]

            box_cls_per_image,  box_delta_per_image, box_iou_per_image = \
                all_level_box_logits[bid], all_level_box_offsets[bid], all_level_box_ctrness[bid]

            offsets = self.point_coder.encode(
                all_level_anchors, F.expand_dims(gt_boxes[:, :4], axis=1)
            )

            object_sizes_of_interest = F.concat([
                F.broadcast_to(
                    F.expand_dims(mge.tensor(size, dtype="float32"), axis=0),
                    (anchors_i.shape[0], 2)
                )
                for anchors_i, size in zip(anchors_list, self.cfg.object_sizes_of_interest)
            ], axis=0)
            max_offsets = F.max(offsets, axis=2)
            is_cared_in_the_level = (
                    (max_offsets >= F.expand_dims(object_sizes_of_interest[:, 0], axis=0))
                    & (max_offsets <= F.expand_dims(object_sizes_of_interest[:, 1], axis=0))
            )

            if self.cfg.center_sampling_radius > 0:
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2
                is_in_boxes = []
                for stride, anchors_i in zip(self.cfg.stride, anchors_list):
                    radius = stride * self.cfg.center_sampling_radius
                    center_boxes = F.concat([
                        F.maximum(gt_centers - radius, gt_boxes[:, :2]),
                        F.minimum(gt_centers + radius, gt_boxes[:, 2:4]),
                    ], axis=1)
                    center_offsets = self.point_coder.encode(
                        anchors_i, F.expand_dims(center_boxes, axis=1)
                    )
                    is_in_boxes.append(F.min(center_offsets, axis=2) > 0)
                is_in_boxes = F.concat(is_in_boxes, axis=1)
            else:
                is_in_boxes = F.min(offsets, axis=2) > 0

            gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            # FIXME: use repeat instead of broadcast_to
            areas = F.broadcast_to(F.expand_dims(gt_area, axis=1), offsets.shape[:2])
            areas[~is_cared_in_the_level] = float("inf")
            areas[~is_in_boxes] = float("inf")

            match_indices = F.argmin(areas, axis=0)
            gt_boxes_matched = gt_boxes[match_indices]
            anchor_min_area = F.indexing_one_hot(areas, match_indices, axis=0)

            labels = gt_boxes_matched[:, 4].astype("int32")
            labels[anchor_min_area == float("inf")] = 0
            offsets = self.point_coder.encode(all_level_anchors, gt_boxes_matched[:, :4])

            left_right = offsets[:, [0, 2]]
            top_bottom = offsets[:, [1, 3]]
            ctrness = F.sqrt(
                F.maximum(F.min(left_right, axis=1) / F.max(left_right, axis=1), 0)
                * F.maximum(F.min(top_bottom, axis=1) / F.max(top_bottom, axis=1), 0)
            )

            # labels_list.append(labels)
            # offsets_list.append(offsets)
            # ctrness_list.append(ctrness)

            # Do OTA
            num_gt = len(batched_gt_boxes)
            num_anchor = len(offsets)
            shape = (num_gt, num_anchor, -1)

            gt_labels = labels.flatten()
            gt_offsets = offsets.reshape(-1, 4)

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.sum()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            loss_cls = layers.sigmoid_focal_loss(
                box_cls_per_image.unsqueeze(0).expand(shape),
                gt_targets[valid_mask].expand(shape),
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum(axis=-1)

            loss_cls_bg = layers.sigmoid_focal_loss(
                box_cls_per_image,
                F.zeros_like(box_cls_per_image),
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum(axis=-1)

            ious, loss_delta = layers.iou_loss(
                box_delta_per_image.unsqueeze(0),
                gt_offsets[fg_mask],
                box_mode="ltrb",
                loss_type=self.cfg.iou_loss_type,
                return_type="ious_iouloss"
            )

            loss = loss_cls + self.cfg.loss_bbox_weight * loss_delta + 1e6 * (1 - is_in_boxes.astype(float))

            # Performing Dynamic k Estimation
            topk_ious, _ = F.topk(ious * is_in_boxes.astype(float), self.OTA['TOP_CANDIDATES'])
            mu = ious.new_ones(num_gt + 1)
            mu[:-1] = F.clip(topk_ious.sum(1).int(), lower=1).astype(float)
            mu[-1] = num_anchor - mu[:-1].sum()
            nu = ious.new_ones(num_anchor)
            loss = F.concat([loss, loss_cls_bg.unsqueeze(0)], axis=0)

            # Solving Optimal-Transportation-Plan pi via Sinkhorn-Iteration.
            _, pi = self.sinkhorn(mu, nu, loss)

            # Rescale pi so that the max pi for each gt equals to 1.
            rescale_factor, _ = pi.max(axis=1)
            pi = pi / rescale_factor.unsqueeze(1)

            max_assigned_units, matched_gt_inds = F.max(pi, axis=0)
            gt_classes_i = labels.new_ones(num_anchor) * self.cfg.num_classes
            fg_mask = matched_gt_inds != num_gt
            gt_classes_i[fg_mask] = labels[matched_gt_inds[fg_mask]]
            labels_list.append(gt_classes_i)
            assigned_units.append(max_assigned_units)

            box_target_per_image = offsets.new_zeros((num_anchor, 4))
            box_target_per_image[fg_mask] = \
                offsets[matched_gt_inds[fg_mask], F.arange(num_anchor)[fg_mask]]
            offsets_list.append(box_target_per_image)

            gt_ious_per_image = ious.new_zeros((num_anchor, 1))
            gt_ious_per_image[fg_mask] = ious[matched_gt_inds[fg_mask],
                                              F.arange(num_anchor)[fg_mask]].unsqueeze(1)
            ctrness_list.append(gt_ious_per_image)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
            F.stack(ctrness_list, axis=0).detach(),
        )


class SinkhornDistance(M.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = F.ones_like(mu)
        v = F.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (F.log(
                    nu + 1e-8) - F.logsumexp(self.M(C, u, v).transpose(-2, -1), axis=-1)) + v
            u = self.eps * \
                (F.log(
                    mu + 1e-8) - F.logsumexp(self.M(C, u, v), axis=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = F.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = F.sum(
            pi * C, axis=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


class FCOSConfig:
    # pylint: disable=too-many-statements
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.backbone_norm = "FrozenBN"
        self.backbone_freeze_at = 2
        self.fpn_norm = None
        self.fpn_in_features = ["res3", "res4", "res5"]
        self.fpn_in_strides = [8, 16, 32]
        self.fpn_in_channels = [512, 1024, 2048]
        self.fpn_out_channels = 256
        self.fpn_top_in_feature = "p5"
        self.fpn_top_in_channel = 256

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 80
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        # ----------------------- net cfg ------------------------- #
        self.stride = [8, 16, 32, 64, 128]
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]

        self.num_anchors = 1
        self.anchor_offset = 0.5

        self.object_sizes_of_interest = [
            [-1, 64], [64, 128], [128, 256], [256, 512], [512, float("inf")]
        ]
        self.center_sampling_radius = 1.5
        self.class_aware_box = False
        self.cls_prior_prob = 0.01

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.iou_loss_type = "giou"
        self.loss_bbox_weight = 1.0
        self.num_losses = 4

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        self.train_image_max_size = 1333

        self.basic_lr = 0.01 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 54
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [42, 50]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.6



# from detection.tools.utils import  import_from_file
# current_network = import_from_file('../configs/fcos_res18_coco_3x_800size.py')
# cfg = current_network.Cfg()
# cfg.backbone_pretrained = False
# model = current_network.Net(cfg)