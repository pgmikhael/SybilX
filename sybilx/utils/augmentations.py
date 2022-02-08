from sybilx.augmentations.basic import ToTensor
from sybilx.utils.registry import get_object


def get_augmentations(image_augmentations, tensor_augmentations, args):
    """
    Args:
        image_augmentations: augmentations as list of tuples [(augmentation_name, dict(augmentation_kwargs) )]
        tensor_augmentations: augmentations as list of tuples [(augmentation_name, dict(augmentation_kwargs) )]
        args
    Returns:
        augmentations: list of augmentation instances
    """
    augmentations = []
    augmentations = _add_augmentations(augmentations, image_augmentations, args)
    augmentations.append(ToTensor())
    augmentations = _add_augmentations(augmentations, tensor_augmentations, args)
    return augmentations


def _add_augmentations(augmentations, new_augmentations, args):
    """
    Iterates through augmentations (list of tuples), parses name and kwargs, and returns list of augmentation objects for given registry
    """
    for trans in new_augmentations:
        name = trans[0]
        kwargs = trans[1]
        augmentations.append(get_object(name, "augmentation")(args, kwargs))

    return augmentations
