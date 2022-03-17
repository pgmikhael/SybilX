import torch
from sybilx.augmentations.abstract import Abstract_augmentation
from sybilx.utils.registry import register_object

@register_object("project", "augmentation")
class ProjectCT(Abstract_augmentation):
    """
    Projects 3D CT volume to approximate X-Ray
    """

    def __init__(self, args, kwargs):
        super(ProjectCT, self).__init__()
        assert len(kwargs) == 0

    def __call__(self, input_dict, sample=None):
        volume = input_dict["input"]
        mask = input_dict["mask"]

        img = torch.flipud(torch.mean(volume, dim=2))
        mask = torch.flipud(torch.mean(mask, dim=2))

        input_dict["input"] = img
        input_dict["mask"] = mask

        return input_dict

