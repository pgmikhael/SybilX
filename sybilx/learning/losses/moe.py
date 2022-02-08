from modules.utils.shared import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb


@register_object("moe_survival", "loss")
def get_cross_entropy_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    losses, probs = [], []
    for i in range(args.num_experts):
        logit = model_output["logit_{}".format(i)]
        y_seq, y_mask = batch["y_seq"], batch["y_mask"]
        losses.append(
            F.binary_cross_entropy_with_logits(
                logit, y_seq.float(), weight=y_mask.float(), reduction="none"
            ).sum(-1)
            / torch.sum(y_mask.float())
        )
        probs.append(F.sigmoid(logit).detach())
    total_loss = (torch.vstack(losses).T * model_output["moe_weight"]).sum()
    probs = torch.stack(probs).mean(0)
    logging_dict["survival_loss"] = total_loss.detach()
    predictions["probs"] = probs.detach()
    predictions["golds"] = batch["y"]
    predictions["censors"] = batch["time_at_event"]
    return total_loss, logging_dict, predictions
