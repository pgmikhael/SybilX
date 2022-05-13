import copy
import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet50

from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.utils.loading import get_lightning_model
from sybilx.utils.registry import register_object, get_object


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
        attention_scores = self.attention_fc(x.transpose(1,2)) # (B, WH, C) -> (B, WH, 1)
                                                               # (B, 25, 1536) - > (B, 25 , 1)
    
        output['image_attention_2d'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, -1)
        # (B, WH, 1) -> (B, 1, WH) -> (B, WH)

        attention_scores = self.softmax(attention_scores.transpose(1,2)) #B, 1, WH

        x = x * attention_scores #(B, C, WH) * (B, 1 WH) -> B, C, WH
        x = torch.sum(x, dim=-1) # (B, C)
        output['hidden'] = x.view(B, C)
        return output


@register_object("simple_sybilx_r50", "model")
class SimpleSybilXR50(nn.Module):
    def __init__(self, args):
        super(SimpleSybilXR50, self).__init__()
        self.args = args
        # encode 
        self.image_encoder = self.get_image_encoder()
        # pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # predict
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, args.num_classes)
        
    def get_image_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 2048

    def forward(self, x, batch = None):
        output = {}

        # encode
        encoded_image = self.image_encoder(x)
        output["activ_2d"] = encoded_image       
        # pool
        output["hidden"] = self.avg_pool(encoded_image).squeeze(2).squeeze(2)
        # predict
        output['hidden'] = self.lin1(output["hidden"])
        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output["logit"] = self.fc(output["hidden"])

        return output


@register_object("simple_sybilx_random", "model")
class SimpleSybilXR50Random(SimpleSybilXR50):
    def __init__(self, args):
        super(SimpleSybilXR50Random, self).__init__(args)

    def get_image_encoder(self):
        encoder = resnet50(pretrained=False)
        return nn.Sequential(*list(encoder.children())[:-2])


@register_object("simple_sybilx_r50_attn_global", "model")
class SimpleSybilXR50AttnGlobal(nn.Module):
    def __init__(self, args):
        super(SimpleSybilXR50AttnGlobal, self).__init__()
        self.args = args
        # encode 
        self.image_encoder = self.get_image_encoder()
        # attention
        self.attn = AttentionPool2D(num_chan=self.ENCODER_OUTPUT_DIM)
        # pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # predict
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM*2, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, args.num_classes)
    

    def get_image_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 2048

    def forward(self, x, batch = None):
        output = {}

        # encode
        encoded_image = self.image_encoder(x)
        output["activ_2d"] = encoded_image
        # pool
        output["hidden"] = self.avg_pool(encoded_image).squeeze(2).squeeze(2)

        # attn
        attn_output = self.attn(encoded_image) # contains 'image_attention_2d' and 'hidden'
        output["attn_hidden"] = self.relu(attn_output["hidden"])
        output["attn_hidden"] = self.dropout(output["attn_hidden"])

        # combine global and local
        output["hidden"] = torch.cat([output["hidden"], output["attn_hidden"]], axis=-1)

        # predict
        output['hidden'] = self.lin1(output["hidden"])
        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output["logit"] = self.fc(output["hidden"])

        return output


@register_object("simple_sybilx_r50_attn_local", "model")
class SimpleSybilXR50AttnLocal(nn.Module):
    def __init__(self, args):
        super(SimpleSybilXR50AttnLocal, self).__init__(args)
        self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM, args.hidden_size)

    def forward(self, x, batch = None):
        output = {}

        # encode
        encoded_image = self.image_encoder(x)
        output["activ_2d"] = encoded_image
        
        # attn
        attn_output = self.attn(encoded_image) # contains 'image_attention_2d' and 'hidden'
        output["hidden"] = self.relu(attn_output["hidden"])
        output["hidden"] = self.dropout(output["attn_hidden"])

        # predict
        output['hidden'] = self.lin1(output["hidden"])
        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output["logit"] = self.fc(output["hidden"])

        return output