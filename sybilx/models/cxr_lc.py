import torch.nn as nn
import torchvision
import torch
import pretrainedmodels
from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.models.pooling_layer import MultiAttentionPool
from sybilx.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer
from sybilx.utils.registry import register_object


@register_object("cxrlc", "model")
class ChestXRayLungCancer(nn.Module):
    def __init__(self, args):
        super(ChestXRayLungCancer, self).__init__()

        self.hidden_dim = 512
 
        # encoder = torchvision.models.inception_v3(pretrained=True) # older model 
        model_name = 'inceptionv4'
        encoder = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

        # this is refactored from fastai.vision.learner.create_head and fastai.layers.LinBnDrop
        lin_ftrs = [1536, self.hidden_dim, 32]
        bns = [True] + [True]*len(lin_ftrs[1:])
        ps = [0.75]
        ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)- 1) # 2) + [None]
        
        # pool layers
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1))
 
        layers = [nn.Flatten()]
  
        for n_in, n_out, batch_norm, p, activation_fn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_in))
            layers.append(nn.Dropout(p))
            layers.append(nn.Linear(n_in, n_out))
            if activation_fn is not None: 
                layers.append(activation_fn)
        
        # Image 
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.custom_head = nn.Sequential(*layers)

        # Age, Sex, Smoking Status
        risk_factors_layers = [nn.Linear(14, 32), nn.ReLU(), nn.Dropout(p=args.dropout), nn.Linear(32, 32), nn.ReLU()] # change input dim to 7 if risk factors changes
        self.risk_factors_mlp = nn.Sequential(*risk_factors_layers)

        # final MLP
        final_layers = [nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p=args.dropout), nn.Linear(32, 2), nn.ReLU()]
        self.final_mlp = nn.Sequential(*final_layers)

    def forward(self, x, batch = None):
        output = {}
        risk_factors_hidden = self.risk_factors_mlp( batch['risk_factors'].float() ) # TODO: age, sex, smoking  (currently all risk factors)
        image_hidden = self.image_encoder( x )
        image_hidden = self.pool(image_hidden)
        image_hidden = self.custom_head(image_hidden)
        output["hidden"] = torch.cat( [risk_factors_hidden, image_hidden], dim=1 )
        output["logit"] = self.final_mlp( output["hidden"] )
        return output



class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)
