from sybilx.utils.registry import register_object
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
    predictions["preds"] = predictions["probs"].argmax(axis=-1).reshape(-1)
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


@register_object("source_discrimination", "loss")
def discriminator_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    d_output = model.discriminator(model_output, batch)
    loss = (
        F.cross_entropy(d_output["logit"], batch[args.adv_key].long())
        * args.adv_loss_lambda
    )
    logging_dict["discrim_loss"] = loss.detach()
    predictions["discrim_probs"] = F.softmax(d_output["logit"], dim=-1).detach()
    predictions["discrim_golds"] = batch[args.adv_key]

    if model.reverse_discrim_loss:
        loss = -loss

    return loss, logging_dict, predictions


@register_object("device_discrimination", "loss")
def device_discriminator_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    d_output = model.discriminator(model_output, batch)
    loss = (
        F.cross_entropy(d_output["logit"], batch["device"].long())
        * args.adv_loss_lambda
    )
    logging_dict["device_loss"] = loss.detach()
    predictions["device_probs"] = F.softmax(d_output["device_logit"], dim=-1).detach()
    predictions["device_golds"] = batch["device"]

    if model.reverse_discrim_loss:
        loss = -loss

    return loss, logging_dict, predictions


@register_object("thickness_discrimination", "loss")
def thickness_discriminator_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    d_output = model.discriminator(model_output, batch)
    loss = (
        F.cross_entropy(d_output["logit"], batch["slice_thickness"].long())
        * args.adv_loss_lambda
    )
    logging_dict["thickness_loss"] = loss.detach()
    predictions["thickness_probs"] = F.softmax(
        d_output["thickness_logit"], dim=-1
    ).detach()
    predictions["thickness_golds"] = batch["slice_thickness"]

    if model.reverse_discrim_loss:
        loss = -loss

    return loss, logging_dict, predictions
