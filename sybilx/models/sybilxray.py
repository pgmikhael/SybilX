import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet50, resnet152#, vit_l_32, convnext_large

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
        attention_scores = self.attention_fc(x.transpose(1,2)) # (B, WH, C) -> (B, WH, 1)
                                                               # (B, 25, 1536) - > (B, 25 , 1)
    
        output['image_attention'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, -1)
        # (B, WH, 1) -> (B, 1, WH) -> (B, WH)

        attention_scores = self.softmax(attention_scores.transpose(1,2)) #B, 1, WH

        x = x * attention_scores #(B, C, WH) * (B, 1 WH) -> B, C, WH
        x = torch.sum(x, dim=-1) # (B, C)
        output['hidden'] = x.view(B, C)
        return output

@register_object("sybilxray_inception", "model")
class SybilXrayInception(nn.Module):
    def __init__(self, args):
        super(SybilXrayInception, self).__init__()

        self.image_encoder = self.get_image_encoder()
        self.args = args

        if args.with_attention:
            self.pool = AttentionPool2D(num_chan=self.HIDDEN_DIM)
            self.lin1 = nn.Linear(self.HIDDEN_DIM*2, self.HIDDEN_DIM)
        else:
            self.lin1 = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        # if using survival setup then finish with cumulative prob layer, otherwise fc layer
        if "survival" not in args.loss_fns:
            self.fc = nn.Linear(self.HIDDEN_DIM, args.num_classes)
        else:
            self.prob_of_failure_layer = Cumulative_Probability_Layer(
            self.HIDDEN_DIM, args, max_followup=args.max_followup
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_image_encoder(self):
        encoder = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        return nn.Sequential(*list(encoder.children())[:-2])
    
    @property
    def HIDDEN_DIM(self):
        """Size of input to cumulative prob. layer. 

        Must match no. of channels in image encoder hidden output.
        """
        return 1536

    def forward(self, x, batch = None):
        output = {}
        x = self.image_encoder(x)
        output["activ"] = x
        pool_output = self.aggregate_and_classify(x)
        output.update(pool_output)

        return output

    def aggregate_and_classify(self, x):
        # get attention
        if self.args.with_attention:
            pool_output = self.pool(x)
            pool_output["attn_hidden"] = self.relu(pool_output["hidden"])
            pool_output["attn_hidden"] = self.dropout(pool_output["attn_hidden"])
        else:
            pool_output = {}
        
        # pass forward average encoded image 
        pool_output["hidden"] = self.avg_pool(x).squeeze(2).squeeze(2)
        # if using attention concat
        if self.args.with_attention:
            pool_output["hidden"] = torch.cat([pool_output["hidden"], pool_output["attn_hidden"]], axis=-1)

        pool_output['hidden'] = self.lin1(pool_output["hidden"])
        pool_output['hidden'] = self.dropout(self.relu(pool_output['hidden']))

        if "survival" not in self.args.loss_fns:
            pool_output["logit"] = self.fc(pool_output["hidden"])
        else:
            pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"])

        return pool_output

@register_object("sybilxray_r50", "model")
class SybilXrayR50(SybilXrayInception):
    def get_image_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def HIDDEN_DIM(self):
        return 2048

@register_object("sybilxray_r152", "model")
class SybilXrayR152(SybilXrayInception):
    def get_image_encoder(self):
        encoder = resnet152(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def HIDDEN_DIM(self):
        return 2048

#@register_object("sybilxray_vit", "model")
#class SybilXrayViT(SybilXrayInception):
#    def get_image_encoder(self):
#        encoder = vit_l_32(pretrained=True)
#        return nn.Sequential(*list(encoder.children())[:-1])
#
#    @property
#    def HIDDEN_DIM(self):
#        return 3
#
#@register_object("sybilxray_convnext", "model")
#class SybilXrayConvNext(SybilXrayInception):
#    def get_image_encoder(self):
#        encoder = convnext_large(pretrained=True)
#        return nn.Sequential(*list(encoder.children())[:-2])
#
#    @property
#    def HIDDEN_DIM(self):
#        return 1536
