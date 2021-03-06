# type: ignore

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from sybilx.models.sybil import SybilNet
from sybilx.serie import Serie

# data
import sybilx.datasets.nlst
import sybilx.datasets.mgh
import sybilx.datasets.taiwan

# augmentation
import sybilx.augmentations.rawinput
import sybilx.augmentations.tensor

# loader
import sybilx.loaders.image_loaders

# lightning
import sybilx.lightning.base

# optimizers
import sybilx.learning.optimizers.basic

# scheduler
import sybilx.learning.schedulers.basic

# losses
import sybilx.learning.losses.basic
import sybilx.learning.losses.guided_attention

# metrics
import sybilx.learning.metrics.basic
import sybilx.learning.metrics.survival

# callbacks
import sybilx.callbacks.basic
import sybilx.callbacks.swa


# branch-specific
import sybilx.datasets.joint_nlst_mgh
import sybilx.lightning.domain_adapter
import sybilx.models.adversary

__all__ = ["Sybil", "Serie"]
