import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
import copy
from sybilx.utils.registry import register_object, get_object
from sybilx.lightning.base import Base
from sybilx.models.adversary import AlignmentMLP, MultiAlignmentMLP


@register_object("domain_adapter", "lightning")
class DomainAdaptation(Base):
    """
    PyTorch Lightning module used as base for running training and test loops

    Args:
        args: argparser Namespace
    """

    def __init__(self, args):
        super().__init__(args)
        self.discriminator = AlignmentMLP(args)
        self.reverse_discrim_loss = False

    def step(self, batch, batch_idx, optimizer_idx):
        logged_output = OrderedDict()

        if optimizer_idx == 0 or optimizer_idx is None:
            self.reverse_discrim_loss = True
            model_output = self.model(batch["x"], batch)

        elif optimizer_idx == 1:
            self.reverse_discrim_loss = False
            with torch.no_grad():
                model_output = self.model(batch["x"], batch)
        else:
            print("Got invalid optimizer_idx! optimizer_idx =", optimizer_idx)

        loss, logging_dict, predictions = self.compute_loss(model_output, batch)
        predictions = self.store_in_predictions(predictions, batch)
        predictions = self.store_in_predictions(predictions, model_output)

        logged_output["loss"] = loss
        logged_output.update(logging_dict)
        logged_output["preds_dict"] = predictions

        if (
            (self.args.log_gen_image)
            and (self.trainer.is_global_zero)
            and (batch_idx == 0)
            and (self.current_epoch % 100 == 0)
        ):
            self.log_image(model_output, batch)

        return logged_output

    def configure_optimizers(self):
        """
        Obtain optimizers and hyperparameter schedulers for model

        """
        enc_params = [param for param in self.model.parameters() if param.requires_grad]
        disc_params = [
            param for param in self.discriminator.parameters() if param.requires_grad
        ]

        enc_optimizer = get_object(self.args.optimizer, "optimizer")(
            enc_params, self.args
        )

        discriminator_args = copy.deepcopy(self.args)
        discriminator_args.lr = self.args.adv_lr
        disc_optimizer = get_object(self.args.optimizer, "optimizer")(
            disc_params, discriminator_args
        )

        enc_scheduler = {
            "scheduler": get_object(self.args.scheduler, "scheduler")(
                enc_optimizer, self.args
            ),
            "monitor": self.args.monitor,
            "interval": "epoch",
            "frequency": 1,
        }

        disc_scheduler = {
            "scheduler": get_object(self.args.scheduler, "scheduler")(
                disc_optimizer, self.args
            ),
            "monitor": self.args.monitor,
            "interval": "epoch",
            "frequency": self.args.num_adv_steps,
        }

        return [enc_optimizer, disc_optimizer], [enc_scheduler, disc_scheduler]


@register_object("multi_domain_adapter", "lightning")
class MultiDomainAdaptation(DomainAdaptation):
    """
    PyTorch Lightning module used as base for running training and test loops

    Args:
        args: argparser Namespace
    """

    def __init__(self, args):
        super().__init__(args)
        self.discriminator = MultiAlignmentMLP(args)
