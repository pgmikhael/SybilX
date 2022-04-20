from sybilx.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

@register_object("guided_attention", "loss")
def get_annotation_loss(model_output, batch, model, args):
    total_loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()

    (
        B,
        _,
        N,
        H,
        W,
    ) = model_output["activ"].shape

    batch_mask = batch["has_annotation"]

    for attn_num in [1, 2]:

        side_attn = -1
        if model_output.get("image_attention_{}".format(attn_num), None) is not None:
            if len(batch["image_annotations"].shape) == 4:
                batch["image_annotations"] = batch["image_annotations"].unsqueeze(1)

            # resize annotation to 'activ' size
            annotation_gold = F.interpolate(
                batch["image_annotations"], (N, H, W), mode="area"
            )
            annotation_gold = annotation_gold * batch_mask[:, None, None, None, None]

            # renormalize scores
            mask_area = annotation_gold.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            mask_area[mask_area == 0] = 1
            annotation_gold /= mask_area

            # reshape annotation into 1D vector
            annotation_gold = annotation_gold.view(B, N, -1).float()

            # get mask over annotation boxes in order to weigh
            # non-annotated scores with zero when computing loss
            annotation_gold_mask = (annotation_gold > 0).float()

            num_annotated_samples = (annotation_gold.view(B * N, -1).sum(-1) > 0).sum()
            num_annotated_samples = max(1, num_annotated_samples)

            pred_attn = (
                model_output["image_attention_{}".format(attn_num)]
                * batch_mask[:, None, None]
            )
            kldiv = (
                F.kl_div(pred_attn, annotation_gold, reduction="none")
                * annotation_gold_mask
            )

            # sum loss per volume and average over batches
            loss = kldiv.sum() / num_annotated_samples
            logging_dict["image_attention_loss_{}".format(attn_num)] = loss.detach()
            total_loss += args.image_attention_loss_lambda * loss

            # attend to cancer side
            cancer_side_mask = (batch["cancer_laterality"][:, :2].sum(-1) == 1).float()[
                :, None
            ]  # only one side is positive
            cancer_side_gold = (
                batch["cancer_laterality"][:, 1].unsqueeze(1).repeat(1, N)
            )  # left side (seen as lung on right) is positive class
            num_annotated_samples = max(N * cancer_side_mask.sum(), 1)
            side_attn = torch.exp(model_output["image_attention_{}".format(attn_num)])
            side_attn = side_attn.view(B, N, H, W)
            side_attn = torch.stack(
                [
                    side_attn[:, :, :, : W // 2].sum((2, 3)),
                    side_attn[:, :, :, W // 2 :].sum((2, 3)),
                ],
                dim=-1,
            ) # [(B, N, 1), (B, N, 1)] -> (B, N, 2)
            side_attn_log = F.log_softmax(side_attn, dim=-1).transpose(1, 2) # (B, 2, N)

            loss = (
                F.cross_entropy(side_attn_log, cancer_side_gold, reduction="none")
                * cancer_side_mask
            ).sum() / num_annotated_samples
            logging_dict[
                "image_side_attention_loss_{}".format(attn_num)
            ] = loss.detach()
            total_loss += args.image_attention_loss_lambda * loss

        if model_output.get("volume_attention_{}".format(attn_num), None) is not None:
            # find size of annotation box per slice and normalize
            annotation_gold = batch["annotation_areas"].float() * batch_mask[:, None]

            if N != args.num_images:
                annotation_gold = F.interpolate(
                    annotation_gold.unsqueeze(1), (N), mode="linear", align_corners=True
                )[:, 0]
            area_per_slice = annotation_gold.sum(-1).unsqueeze(-1)
            area_per_slice[area_per_slice == 0] = 1
            annotation_gold /= area_per_slice

            num_annotated_samples = (annotation_gold.sum(-1) > 0).sum()
            num_annotated_samples = max(1, num_annotated_samples)

            # find slices with annotation
            annotation_gold_mask = (annotation_gold > 0).float()

            pred_attn = (
                model_output["volume_attention_{}".format(attn_num)]
                * batch_mask[:, None]
            )
            kldiv = (
                F.kl_div(pred_attn, annotation_gold, reduction="none")
                * annotation_gold_mask
            )  # B, N
            loss = kldiv.sum() / num_annotated_samples

            logging_dict["volume_attention_loss_{}".format(attn_num)] = loss.detach()
            total_loss += args.volume_attention_loss_lambda * loss

            if isinstance(side_attn, torch.Tensor):
                # attend to cancer side
                cancer_side_mask = (
                    batch["cancer_laterality"][:, :2].sum(-1) == 1
                ).float()  # only one side is positive
                cancer_side_gold = batch["cancer_laterality"][
                    :, 1
                ]  # left side (seen as lung on right) is positive class
                num_annotated_samples = max(cancer_side_mask.sum(), 1)

                pred_attn = torch.exp(
                    model_output["volume_attention_{}".format(attn_num)]
                )
                side_attn = (side_attn * pred_attn.unsqueeze(-1)).sum(1)
                side_attn_log = F.log_softmax(side_attn, dim=-1)

                loss = (
                    F.cross_entropy(side_attn_log, cancer_side_gold, reduction="none")
                    * cancer_side_mask
                ).sum() / num_annotated_samples
                logging_dict[
                    "volume_side_attention_loss_{}".format(attn_num)
                ] = loss.detach()
                total_loss += args.volume_attention_loss_lambda * loss

    return total_loss * args.annotation_loss_lambda, logging_dict, predictions

@register_object("guided_attention_2d", "loss")
def get_2d_annotation_loss(model_output, batch, model, args):
    loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()
    
    B, C, H, W = model_output["activ_2d"].shape

    batch_mask = batch["projection_has_annotation"]

    if model_output.get("image_attention_2d", None) is not None:
        if len(batch["projection_image_annotations"].shape) == 3:
            batch["projection_image_annotations"] = batch["projection_image_annotations"].unsqueeze(1)

        # resize annotation to 'activ' size
        annotation_gold = F.interpolate(batch["projection_image_annotations"], (H, W), mode="area")
        annotation_gold = annotation_gold * batch_mask[:, None, None, None] # (B, 1, H, W)

        # renormalize scores
        mask_area = annotation_gold.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1) # (B, 1, 1, 1)
        mask_area[mask_area == 0] = 1
        annotation_gold /= mask_area # (B, 1, H, W)

        # reshape annotation into 1D vector 
        annotation_gold = annotation_gold.view(B, -1).float() # (B, H * W)

        # get mask over annotation boxes in order to weigh
        # non-annotated scores with zero when computing loss
        annotation_gold_mask = (annotation_gold > 0).float() # (B, H * W)

        num_annotated_samples = batch_mask.sum() # tensor(int), dim=0
        num_annotated_samples = max(1, num_annotated_samples)

        pred_attn = model_output["image_attention_2d"] * batch_mask[:, None] # (B, H * W)
        kldiv = F.kl_div(pred_attn, annotation_gold, reduction="none") * annotation_gold_mask # (B, H * W)

        if args.right_annotation_loss_lambda is not None:
            cancer_side_mask = (batch["cancer_laterality"][:, :2].sum(-1) == 1).float()[:, None]  # (B, 1), only one side is positive
            cancer_side_gold = batch["cancer_laterality"][:, 1] # (B, 1), left side (seen as lung on right) is positive class
            left_mask = cancer_side_gold & cancer_side_mask
            right_mask = (~cancer_side_gold) & cancer_side_mask
            kldiv[right_mask] *= args.right_annotation_loss_lambda
            kldiv[left_mask] *= 1 - args.right_annotation_loss_lambda

        # sum loss per slice and average over batches
        loss = kldiv.sum() / num_annotated_samples # tensor(int), dim=0
        logging_dict["image_attention_loss_2d"] = loss.detach()
    
    return loss * args.annotation_loss_lambda_2d, logging_dict, predictions 


@register_object("guided_side_attention_2d", "loss")
def get_2d_side_annotation_loss(model_output, batch, model, args):
    loss, logging_dict, predictions = 0, OrderedDict(), OrderedDict()

    B, C, H, W = model_output["activ_2d"].shape

    if model_output.get("image_attention_2d", None) is not None:

        # attend to cancer side
        cancer_side_mask = (batch["cancer_laterality"][:, :2].sum(-1) == 1).float()[:, None]  # (B, 1), only one side is positive
        cancer_side_gold = batch["cancer_laterality"][:, 1] # (B, 1), left side (seen as lung on right) is positive class
        num_annotated_samples = max(cancer_side_mask.sum(), 1)
        side_attn = torch.exp(model_output["image_attention_2d"]) # (B, H * W)
        side_attn = side_attn.view(B, H, W) # (B, H, W)
        # sum across the H and then the W, so we find which lung side the attn was on
        side_attn = torch.stack([side_attn[:, :, : W // 2].sum((1, 2)), side_attn[:, :, W // 2 :].sum((1, 2)),], dim=-1,) # (B, 2)
        side_attn_log = F.log_softmax(side_attn, dim=-1) # .transpose(1, 2) # TODO: check this

        loss = (F.cross_entropy(side_attn_log, cancer_side_gold, reduction="none") * cancer_side_mask).sum() / num_annotated_samples
        logging_dict["image_side_attention_loss_2d"] = loss.detach()
        
    return loss * args.image_side_attention_loss_lambda_2d, logging_dict, predictions
