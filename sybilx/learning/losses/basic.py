from modules.utils.shared import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb


@register_object("cross_entropy", "loss")
def get_cross_entropy_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output["logit"]
    loss = F.cross_entropy(logit, batch["y"].long())
    logging_dict["cross_entropy_loss"] = loss.detach()
    predictions["probs"] = F.softmax(logit, dim=-1).detach()
    predictions["golds"] = batch["y"]
    return loss, logging_dict, predictions


@register_object("survival", "loss")
def get_survival_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output["logit"]
    y_seq, y_mask = batch["y_seq"], batch["y_mask"]
    loss = F.binary_cross_entropy_with_logits(
        logit, y_seq.float(), weight=y_mask.float(), reduction="sum"
    ) / torch.sum(y_mask.float())
    logging_dict["survival_loss"] = loss.detach()
    predictions["probs"] = torch.sigmoid(logit).detach()
    predictions["golds"] = batch["y"]
    predictions["censors"] = batch["time_at_event"]
    return loss, logging_dict, predictions


@register_object("ordinal_cross_entropy", "loss")
def get_ordinal_ce_loss(model_output, batch, model, args):
    """
    Computes cross-entropy loss

    If batch contains they key 'has_y', the cross entropy loss will be computed for samples where batch['has_y'] = 1
    Expects model_output to contain 'logit'

    Returns:
        loss: cross entropy loss
        l_dict (dict): dictionary containing cross_entropy_loss detached from computation graph
        p_dict (dict): dictionary of model predictions and ground truth labels (preds, probs, golds)
    """
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    logit = model_output["logit"]
    yseq = batch["yseq"]
    ymask = batch["ymask"]

    loss = F.binary_cross_entropy_with_logits(
        logit, yseq.float(), weight=ymask.float(), reduction="sum"
    ) / torch.sum(ymask.float())

    probs = F.logsigmoid(logit)  # log_sum to add probs
    probs = probs.unsqueeze(1).repeat(1, len(args.rank_thresholds), 1)
    probs = torch.tril(probs).sum(2)
    probs = torch.exp(probs)

    p_dict["probs"] = probs.detach()
    preds = probs > 0.5  # class = last prob > 0.5
    preds = preds.sum(-1)
    p_dict["preds"] = preds
    p_dict["golds"] = batch["y"]

    return loss * args.ce_loss_lambda, l_dict, p_dict
