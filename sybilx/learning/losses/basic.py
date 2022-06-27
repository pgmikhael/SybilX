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

    p_dict["logits"] = logit.detach()
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


@register_object("device_thickness_discrimination", "loss")
def device_discriminator_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    d_output = model.discriminator(model_output, batch)
    device_loss = (
        F.cross_entropy(d_output["device_logit"], batch["device"].long())
        * args.adv_loss_lambda
    )
    logging_dict["device_loss"] = device_loss.detach()
    predictions["device_probs"] = F.softmax(d_output["device_logit"], dim=-1).detach()
    predictions["device_golds"] = batch["device"]

    thick_loss = (
        F.cross_entropy(d_output["thickness_logit"], batch["slice_thickness"].long())
        * args.adv_loss_lambda
    )
    logging_dict["thickness_loss"] = thick_loss.detach()
    predictions["thickness_probs"] = F.softmax(
        d_output["thickness_logit"], dim=-1
    ).detach()
    predictions["thickness_golds"] = batch["slice_thickness"]

    loss = thick_loss + device_loss
    if model.reverse_discrim_loss:
        loss = -loss

    return loss, logging_dict, predictions

@register_object("risk_factor_loss", "loss")
def get_risk_factor_loss(model_output, batch, model, args):
    total_loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()
    for idx, key in enumerate(args.risk_factor_keys):
        logit = model_output["{}_logit".format(key)]
        gold_rf = batch["risk_factors"][idx]
        is_rf_known = (torch.sum(gold_rf, dim=-1) > 0).unsqueeze(-1).float()

        gold = torch.argmax(gold_rf, dim=-1).contiguous().view(-1)

        loss = (
                F.cross_entropy(logit, gold, reduction="none") * is_rf_known
                    ).sum() / max(1, is_rf_known.sum())
        total_loss += loss
        logging_dict["{}_loss".format(key)] = loss.detach()

        probs = F.softmax(logit, dim=-1).detach()
        predictions["{}_probs".format(key)] = probs.detach()
        predictions["{}_golds".format(key)] = gold.detach()
        predictions["{}_risk_factor".format(key)] = batch["risk_factors"][idx]
        preds = torch.argmax(probs, dim=-1).view(-1)

    return total_loss * args.primary_loss_lambda, logging_dict, predictions

@register_object("survivalv2", "loss")
def get_survival_lossv2(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    y_seq, y_mask = batch["y_seq"], batch["y_mask"]
    loss = 0
    for logitname in ["logit", "risk_factor_logit"]: #img_logit
        logit = model_output["logit"]
        subloss = F.binary_cross_entropy_with_logits(
        logit, y_seq.float(), weight=y_mask.float(), reduction="sum"
        ) / torch.sum(y_mask.float())
        loss += subloss
        logging_dict["survival_{}loss".format(logitname.split("logit")[0]) ] = loss.detach()
    predictions["probs"] = torch.sigmoid(logit).detach()
    predictions["golds"] = batch["y"]
    predictions["censors"] = batch["time_at_event"]
    predictions["survival_loss"] = loss.detach()
    return loss, logging_dict, predictions
