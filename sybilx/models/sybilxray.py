import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet50, resnet152 #, vit_l_16, vit_b_16#, convnext_large

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
    
        output['image_attention_2d'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, -1)
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
            self.pool = AttentionPool2D(num_chan=self.ENCODER_OUTPUT_DIM)
            self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM*2, args.hidden_size)
        else:
            self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM, args.hidden_size)

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        # if using survival setup then finish with cumulative prob layer, otherwise fc layer
        if "survival" in args.loss_fns:
            self.prob_of_failure_layer = Cumulative_Probability_Layer(
                args.hidden_size, args, max_followup=args.max_followup
            )
        elif "corn" in self.args.loss_fns:
            self.prob_of_failure_layer = nn.Linear(args.hidden_size, args.max_followup)
        else:
            self.fc = nn.Linear(args.hidden_size, args.num_classes)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_image_encoder(self):
        encoder = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        return nn.Sequential(*list(encoder.children())[:-2])
    
    @property
    def ENCODER_OUTPUT_DIM(self):
        """Size of input to cumulative prob. layer. 

        Must match no. of channels in image encoder hidden output.
        """
        return 1536

    def forward(self, x, batch = None):
        output = {}
        #if self.args.freeze_encoder_weights:
        #    with torch.no_grad():
        #        x = self.image_encoder(x)
        #else:
        x = self.image_encoder(x)

        output["activ_2d"] = x
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

        if "survival" in self.args.loss_fns or "corn" in self.args.loss_fns:
            pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"])
        else:
            pool_output["logit"] = self.fc(pool_output["hidden"])

        return pool_output

@register_object("sybilxray_r50", "model")
class SybilXrayR50(SybilXrayInception):
    def get_image_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 2048

@register_object("sybilxray_r152", "model")
class SybilXrayR152(SybilXrayInception):
    def get_image_encoder(self):
        encoder = resnet152(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 2048

# VISION TRANSFORMERS
class ViTEncoder(nn.Module):
    """ A pytorch ViT wrapper that returns the image encodings in the same shape as a conv model"""
    def __init__(self, model, patch_size=(16,16)):
        super(ViTEncoder, self).__init__()
        self.model = model
        self.encoder = nn.Sequential(*list(model.children())[:-1])[1]
        self.patch_size = patch_size
        self.output_size = (patch_size[0]-2, patch_size[1]-2)

    def forward(self, img):
        # Use conv layer to rescale to patch size
        x = self.model._process_input(img)

        # pass through ViT encoder
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        # remove class token and reshape
        x = x[:, 1:] 
        x = x.permute(0,2,1) # (B, W*H, C) -> (B, C, W*H)
        x = x.view(-1, self.model.hidden_dim, self.output_size[0], self.output_size[1]) # (B, C, W*H) -> (B, C, W, H)

        return x

#@register_object("sybilxray_vit_h_14", "model")
#class SybilXrayViT(SybilXrayInception):
#    def get_image_encoder(self):
#        encoder = vit_h_14(pretrained=True)
#        return nn.Sequential(*list(encoder.children())[:-1])
#
#    @property
#    def ENCODER_OUTPUT_DIM(self):
#        return 3

#@register_object("sybilxray_vit_l_32", "model")
#class SybilXrayViT(SybilXrayInception):
#    def get_image_encoder(self):
#        encoder = vit_l_32(pretrained=True)
#        return nn.Sequential(*list(encoder.children())[:-1])
#
#    @property
#    def ENCODER_OUTPUT_DIM(self):
#        return 3

@register_object("sybilxray_vit_l_16", "model")
class SybilXrayViT(SybilXrayInception):
    def get_image_encoder(self):
        vit = vit_l_16(pretrained=True)
        return ViTEncoder(vit, patch_size=(16,16))

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 1024

@register_object("sybilxray_vit_b_16", "model")
class SybilXrayViT(SybilXrayInception):
    def get_image_encoder(self):
        vit = vit_b_16(pretrained=True)
        return ViTEncoder(vit, patch_size=(16,16))

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 768

#@register_object("sybilxray_convnext", "model")
#class SybilXrayConvNext(SybilXrayInception):
#    def get_image_encoder(self):
#        encoder = convnext_large(pretrained=True)
#        return nn.Sequential(*list(encoder.children())[:-2])
#
#    @property
#    def ENCODER_OUTPUT_DIM(self):
#        return 1536
