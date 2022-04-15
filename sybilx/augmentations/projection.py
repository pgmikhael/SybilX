import numpy as np
from scipy.ndimage import binary_dilation
from sybilx.augmentations.abstract import Abstract_augmentation
from sybilx.utils.registry import register_object

def project_simple(volume, agg_func=np.mean):
    return np.flipud(agg_func(volume, axis=1)).copy()

def project_campo(volume, beta=0.85):
    """Applies projection from Campo, M.I., Pascau, J. and Estpar, R.S.J.: "Emphysema quantification on simulated X-rays through deep learning techniques."

    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6239425/pdf/nihms-982449.pdf

    Note: 
      intensities of volume has to be in Hounsfield Units
    """
    N = volume.shape[1]

    drr = ( np.exp(beta*((np.maximum(volume, -1024) + 1024)/1000)) ).sum(axis=1) / N

    # also flipud
    drr = np.flipud(drr)

    return drr

@register_object("project_ct", "augmentation")
class ProjectCT(Abstract_augmentation):
    """
    Projects 3D CT volume to approximate X-Ray
    """

    def __init__(self, args, kwargs):
        super(ProjectCT, self).__init__()
        assert len(kwargs) == 0 or len(kwargs) == 2
        if len(kwargs) == 2:
            assert kwargs["method"] in ("mean", "mean-invert", "sum", "campo")
            assert kwargs["annotation_method"] in ("mean", "sum", "max", "mean-threshold", "max-threshold", "binary-grow-1", "binary-grow-2")
            self.method = kwargs["method"]
            self.annotation_method = kwargs["annotation_method"]
        else:
            self.method = "mean"
            self.annotation_method = "mean"

        self.set_cachable(self.method, self.annotation_method)

    def __call__(self, input_dict, sample=None):
        volume = input_dict["input"]
        mask = input_dict["mask"]

        if self.method == "mean":
            img = project_simple(volume, agg_func=np.mean)
        elif self.method == "sum":
            img = project_simple(volume, agg_func=np.sum)
        elif self.method == "mean-invert":
            volume = volume.max() - volume
            img = project_simple(volume, agg_func=np.mean)
        else:
            assert self.method == "campo"
            img = project_campo(volume)

        if self.annotation_method == "mean":
            mask = project_simple(mask, agg_func=np.mean)
        elif self.annotation_method == "sum":
            mask = project_simple(mask, agg_func=np.sum)
        elif self.annotation_method == "max":
            mask = project_simple(mask, agg_func=np.max)
        elif self.annotation_method == "mean-threshold":
            mask = project_simple(mask, agg_func=np.mean)
            mask[mask != 0] = 1
        elif self.annotation_method == "max-threshold":
            mask = project_simple(mask, agg_func=np.max)
            mask[mask != 0] = 1
        elif self.annotation_method == "binary-grow-1":
            mask = project_simple(mask, agg_func=np.max)
            mask = binary_dilation(mask).astype(int) # expand 1 pixel in every direction (including z+ and z-)
            mask[mask != 0] = 1

        else:
            assert self.annotation_method == "binary-grow-2"
            mask = project_simple(mask, agg_func=np.max)
            mask = binary_dilation(mask, iterations=2).astype(int)
            mask[mask != 0] = 1


        input_dict["input"] = img
        input_dict["mask"] = mask

        return input_dict

@register_object("project_ct_cheat", "augmentation")
class ProjectCTCheat(Abstract_augmentation):
    """
    Projection of CTs that "cheats" by multiplying annotated areas up, so they're more visible to the encoder
    """

    def __init__(self, args, kwargs):
        super(ProjectCT, self).__init__()
        assert len(kwargs) == 1
        self.scale = kwargs["scale"]
        self.set_cachable(self.scale)

    def __call__(self, input_dict, sample=None):
        volume = input_dict["input"]
        mask = input_dict["mask"]

        if mask.any():
            volume += self.scale * (volume * mask)

        img = project_campo(volume)
        mask = project_simple(mask, agg_func=np.max)
        mask[mask != 0] = 1

        input_dict["input"] = img
        input_dict["mask"] = mask

        return input_dict
