import numpy as np
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
            assert kwargs["annotation_method"] in ("mean", "sum", "max", "threshold")
            self.method = kwargs["method"]
            self.annotation_method = kwargs["annotation_method"]
        else:
            self.method = "mean"
            self.annotation_method = "mean"

        self.set_cachable()

    def __call__(self, input_dict, sample=None):
        volume = input_dict["input"]
        mask = input_dict["mask"]

        import pdb; pdb.set_trace()
        if self.method == "mean":
            img = project_simple(volume, agg_func=np.mean)
        if self.method == "sum":
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
        else:
            assert self.annotation_method == "threshold"
            mask = project_simple(mask, agg_func=np.mean)
            epsilon = 1e-6
            mask[mask > epsilon] = 1
            mask[mask <= epsilon] = 0

        input_dict["input"] = img
        input_dict["mask"] = mask

        return input_dict

