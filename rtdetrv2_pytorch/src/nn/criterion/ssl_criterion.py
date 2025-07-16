import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision

from ...misc import box_ops
from ...misc import dist_utils
from ...core import register
from .det_criterion import DetCriterion


@register()
class SSLCriterion(DetCriterion):
    def forward(self, outputs, targets, **kwargs):
        if key == "loss_box_reg_mask" or key == "loss_rpn_box_reg_mask":
            # pseudo bbox regression <- 0
            loss_dict[key] = record_dict[key] * 0
        elif key.endswith('_mask') and 'da' in key:
            # mask+da loss <- 0
            loss_dict[key] = record_dict[key] * 0
        elif key == 'loss_classifier_mask' or key == 'loss_objectness_mask':
            # pseudo classification and objectness <- 1
            loss_dict[key] = record_dict[key] * cfg.MODEL.PSEUDO_LABEL_LAMBDA
        else:  # supervised loss
            loss_dict[key] = record_dict[key] * 1

        # in RPNLossCopmutation
        # <...>
        if use_pseudo_labeling_weight=='prob':
            weight = F.sigmoid(objectness[sampled_inds])
            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds], reduction='none'
            )
            objectness_loss = torch.mean(objectness_loss*weight)
