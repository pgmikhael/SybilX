import os
from modules.utils.shared import  register_object
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary

# TODO: add args for various callbacks -- currently hardcoded

@register_object('checkpointer', 'callback')
class Checkpoint(ModelCheckpoint):
    def __init__(self, args) -> None:
        super().__init__(        
            monitor=args.monitor,
            dirpath= os.path.join(args.model_save_dir, args.experiment_name),
            mode='min' if 'loss' in args.monitor else 'max',
            filename= '{}'.format(args.experiment_name) + '{epoch}',
            every_n_epochs =1
            )

@register_object('lr_monitor', 'callback')
class LRMonitor(LearningRateMonitor):
    def __init__(self, args) -> None:
        super().__init__(
            logging_interval='step'
            )

@register_object('rich_model_summary', 'callback')
class ModelSummary(RichModelSummary):
    def __init__(self, args) -> None:
        super().__init__(
            max_depth=1
            )
