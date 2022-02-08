from modules.utils.shared import register_object
from pytorch_lightning.callbacks import StochasticWeightAveraging


@register_object("swa", "callback")
class SWA(StochasticWeightAveraging):
    def __init__(self, args) -> None:
        if "." in args.swa_epoch:
            swa_epoch = float(args.swa_epoch)
        else:
            swa_epoch = int(args.swa_epoch)

        super().__init__(
            swa_epoch_start=swa_epoch,
            swa_lrs=args.swa_lr,
            annealing_epochs=args.swa_annealing_epochs,
            annealing_strategy=args.swa_annealing_strategy,
            avg_fn=None,
        )
