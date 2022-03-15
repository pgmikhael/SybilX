import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet50, resnet152, vit_l_32, convnext_base

from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.utils.registry import register_object


class AttentionPool2D(nn.Module):
    """Simple 2D version of Sybil's Attention Pool."""
    def __init__(self, **kwargs):
        super(AttentionPool2D, self).__init__()
        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        #X dim: B, C, W, H
        output = {}

        B, C, W, H = x.size()
        x = x.view(B, C, W*H)
        attention_scores = self.attention_fc(x.transpose(1,2)) #B, WH , 1
    
        output['image_attention'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, -1) 

        x = x * attention_scores #B, C, WH
        x = torch.sum(x, dim=-1)
        output['hidden'] = x.view(B, C)
        return output

@register_object("sybilxray_inception", "model")
class SybilXrayInception(nn.Module):
    def __init__(self, args):
        super(SybilXrayInception, self).__init__()

        self.hidden_dim = 512 # TODO: args?

        self.image_encoder = self.get_image_encoder(args)

        self.pool = AttentionPool2D(num_chan=512) # output of image encoder?

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            self.hidden_dim, args, max_followup=args.max_followup
        )

    def get_image_encoder():
        encoder = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        return nn.Sequential(*list(encoder.children())[:-2])

    def forward(self, x, batch = None):
        output = {}
        x = self.image_encoder(x)
        pool_output = self.aggregate_and_classify(x)
        output["activ"] = x
        output.update(pool_output)

        return output

    def aggregate_and_classify(self, x):
        pool_output = self.pool(x)

        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"])

        return pool_output

@register_object("sybilxray_r50", "model")
class SybilXrayR50(nn.Module):
    def get_image_encoder():
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

@register_object("sybilxray_r152", "model")
class SybilXrayR152(nn.Module):
    def get_image_encoder():
        encoder = resnet152(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

@register_object("sybilxray_vit", "model")
class SybilXrayViT(nn.Module):
    def get_image_encoder():
        encoder = vit_l_32(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

@register_object("sybilxray_convnext", "model")
class SybilXrayConvNext(nn.Module):
    def get_image_encoder():
        encoder = convnext_base(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])