from sybilx.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb


@register_object("weighted_focal_loss", "loss")
def get_weighted_focal_loss(model_output, batch, model, args):
    """
    Focal loss = self.alpha * (1-pt)**self.gamma * BCE_loss * torch.stack([weights,weights],1).float()
    source: 'Focal Loss for Dense Object Detection' https://arxiv.org/pdf/1708.02002.pdf
    original implementation: https://detectron2.readthedocs.io/en/latest/_modules/fvcore/nn/focal_loss.html#sigmoid_focal_loss
    """
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output["logit"]

    # If using the same 'y' as them, then weight loss by second target variable (time to incidence)
    # weights = batch['y'][:,1].float()
    # batch['y'] = batch['y'][:,0].long()
    batch['y'] = batch['y'].long()

    ce_loss = F.cross_entropy(logit, batch['y'], reduction='none')
    pt = torch.exp(-ce_loss)
    loss = args.wfl_alpha * (1-pt)**args.wfl_gamma * ce_loss # * weights
    loss = torch.mean(loss)
    logging_dict["cross_entropy_loss"] = ce_loss.detach()
    logging_dict["focal_loss"] = loss.detach()
    predictions["probs"] = F.softmax(logit, dim=-1).detach()
    predictions["golds"] = batch["y"]
    predictions["preds"] = predictions["probs"].argmax(axis=-1).reshape(-1)
    return loss, logging_dict, predictions
