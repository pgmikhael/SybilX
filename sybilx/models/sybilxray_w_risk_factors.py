import copy
import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet50

from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.utils.loading import get_lightning_model
from sybilx.utils.registry import register_object, get_object
from sybilx.datasets.risk_factors import PLCORiskFactorVectorizer


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


@register_object("simple_sybilx_r50_w_risk", "model")
class SimpleSybilXR50WRisk(nn.Module):
    def __init__(self, args):
        super(SimpleSybilXR50Risk, self).__init__()
        self.args = args
        self.vectorizer = get_object(args.vectorizer, "vectorizer")(args)
        self.length_risk_factor_vector = self.vectorizer.vector_length

        # risk factor encoder
        self.risk_encoder = nn.Linear(self.length_risk_factor_vector, self.length_risk_factor_vector)

        # encode 
        self.image_encoder = self.get_image_encoder()
        # pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # predict
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM + self.length_risk_factor_vector, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, args.num_classes)
        
    def get_image_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-2])

    @property
    def ENCODER_OUTPUT_DIM(self):
        return 2048

    def forward(self, x, batch = None):
        output = {}

        # encode image
        encoded_image = self.image_encoder(x)
        output["activ_2d"] = encoded_image
        # pool
        output["hidden"] = self.avg_pool(encoded_image).squeeze(2).squeeze(2)
        # encode risk factors
        output["encoded_risk_factors"] = self.risk_encoder(batch['risk_factors'])
        # concatenate
        concatenated = torch.cat([output["hidden"], output["encoded_risk_factors"]], dim=-1)
        # predict
        output['hidden'] = self.lin1(concatenated)
        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output["logit"] = self.fc(output["hidden"])

        return output


@register_object("simple_sybilx_r50_attn_global_w_risk", "model")
class SimpleSybilXR50AttnGlobalWRisk(nn.Module):
    def __init__(self, args):
        super(SimpleSybilXR50AttnGlobalWRisk, self).__init__()
        self.args = args
        self.vectorizer = get_object(args.vectorizer, "vectorizer")(args)
        self.length_risk_factor_vector = self.vectorizer.vector_length

        # risk factor encoder
        self.risk_encoder = nn.Linear(self.length_risk_factor_vector, self.length_risk_factor_vector)
        # encode 
        self.image_encoder = self.get_image_encoder()
        # attention
        self.attn = AttentionPool2D(num_chan=self.ENCODER_OUTPUT_DIM)
        # pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # predict
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM*2 + self.length_risk_factor_vector, args.hidden_size)
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
        # encode risk factors
        output["encoded_risk_factors"] = self.risk_encoder(batch['risk_factors'])
        # combine global and local and risk factors
        output["hidden"] = torch.cat([output["hidden"], output["attn_hidden"], output["encoded_risk_factors"]], axis=-1)
        # predict
        output['hidden'] = self.lin1(output["hidden"])
        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output["logit"] = self.fc(output["hidden"])

        return output


@register_object("simple_sybilx_r50_attn_local_w_risk", "model")
class SimpleSybilXR50AttnLocalWRisk(SimpleSybilXR50AttnGlobal):
    def __init__(self, args):
        super(SimpleSybilXR50AttnLocalWRisk, self).__init__(args)
        self.vectorizer = get_object(args.vectorizer, "vectorizer")(args)
        self.length_risk_factor_vector = self.vectorizer.vector_length

        # risk factor encoder
        self.risk_encoder = nn.Linear(self.length_risk_factor_vector, self.length_risk_factor_vector)
        self.lin1 = nn.Linear(self.ENCODER_OUTPUT_DIM + self.length_risk_factor_vector, args.hidden_size)

    def forward(self, x, batch = None):
        output = {}

        # encode
        encoded_image = self.image_encoder(x)
        output["activ_2d"] = encoded_image
        
        # attn
        attn_output = self.attn(encoded_image) # contains 'image_attention_2d' and 'hidden'
        output["hidden"] = self.relu(attn_output["hidden"])
        output["hidden"] = self.dropout(output["hidden"])

        # encode risk factors
        output["encoded_risk_factors"] = self.risk_encoder(batch['risk_factors'])
        # concatenate
        concatenated = torch.cat([output["hidden"], output["encoded_risk_factors"]], dim=-1)
        # predict
        output['hidden'] = self.lin1(concatenated)

        output['hidden'] = self.dropout(self.relu(output['hidden']))
        output["logit"] = self.fc(output["hidden"])

        return output


@register_object("simple_sybilx_r50_multi_task", "model")
class SimpleSybilXMultiTaskModel(nn.Module):
    def __init__(self, args):
        super(SimpleSybilXMultiTaskModel, self).__init__(args)

        # proj_encoder_model is actually just image encoder model
        self.image_encoder = get_object(args.proj_encoder_model, 'model')(args)

        self.length_risk_factor_vector = get_object(args.vectorizer, "vectorizer")(args).vector_length
        for key in args.risk_factor_keys:
            num_key_features = args.risk_factor_key_to_num_class[key]
            key_fc = nn.Linear(self.ENCODER_OUTPUT_DIM, num_key_features)
            self.add_module("{}_fc".format(key), key_fc)

    def forward(self, x, batch):
        output = {}
        model_output = self.image_encoder(x)

        for indx, key in enumerate(self.args.risk_factor_keys):
            output["{}_logit".format(key)] = self._modules["{}_fc".format(key)](model_output['hidden'])

        return output