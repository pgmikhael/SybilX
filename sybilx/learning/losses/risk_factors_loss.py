from sybilx.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb

@register_object("risk_factor_loss", 'loss')
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
        # preds = torch.argmax(probs, dim=-1).view(-1)

    return total_loss * args.primary_loss_lambda, logging_dict, predictions