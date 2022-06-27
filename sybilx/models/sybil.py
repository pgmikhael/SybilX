import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.models.pooling_layer import MultiAttentionPool
from sybilx.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer
from sybilx.utils.registry import register_object


@register_object("sybil", "model")
class SybilNet(nn.Module):
    def __init__(self, args):
        super(SybilNet, self).__init__()
        self.args = args
        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])

        self.pool = MultiAttentionPool()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            self.hidden_dim, args, max_followup=args.max_followup
        )

    def forward(self, x, batch = None):
        output = {}
        if self.args.from_hiddens:
            return { 'logit': self.prob_of_failure_layer(x) }
        else:
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


class RiskFactorPredictor(SybilNet):
    def __init__(self, args):
        super(RiskFactorPredictor, self).__init__(args)

        self.length_risk_factor_vector = NLSTRiskFactorVectorizer(args).vector_length
        for key in args.risk_factor_keys:
            num_key_features = args.risk_factor_key_to_num_class[key]
            key_fc = nn.Linear(args.hidden_dim, num_key_features)
            self.add_module("{}_fc".format(key), key_fc)

    def forward(self, x, batch):
        output = {}
        x = self.image_encoder(x)
        output = self.pool(x, batch)

        hidden = output["hidden"]
        for indx, key in enumerate(self.args.risk_factor_keys):
            output["{}_logit".format(key)] = self._modules["{}_fc".format(key)](hidden)

        return output

    def get_loss_functions(self):
        return ["risk_factor_loss"]

@register_object("full_sybil", "model")
class SybilFull(nn.Module):
    def __init__(self, args):
        super(SybilFull, self).__init__()
        self.args = args
        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.pool = MultiAttentionPool()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        self.use_pred_risk_factors = args.use_pred_risk_factors
        self.use_available_risk_factors = args.use_available_risk_factors

        num_risk_factor_features = 0
        for key in args.risk_factor_keys:
            num_key_features = args.risk_factor_key_to_num_class[key]
            num_risk_factor_features+=num_key_features
            key_fc = nn.Linear(512, num_key_features)
            self.add_module("{}_fc".format(key), key_fc)
        self.hidden_dim = 512 + num_risk_factor_features
        self.risk_layer = Cumulative_Probability_Layer(self.hidden_dim, args, max_followup=args.max_followup)
    
    def forward(self, x, batch):
        output = {}
        x = self.image_encoder(x)
        output = self.pool(x)
        output["activ"] = x

        img_hidden = output["hidden"]
        output["img_hidden"] = self.relu(self.dropout(img_hidden))
        risk_factor_hidden = [img_hidden]
        for indx, key in enumerate(self.args.risk_factor_keys):
            logit = self._modules["{}_fc".format(key)](img_hidden)
            rf_preds = F.softmax(logit, dim = -1)
            output["{}_logit".format(key)] = logit
            if self.use_available_risk_factors:
                # use if known
                is_rf_known = (torch.sum(batch["risk_factors"][indx], dim=-1) > 0).unsqueeze(-1).float()
                rf = is_rf_known * batch["risk_factors"][indx] + (1-is_rf_known) * rf_preds
                risk_factor_hidden.append(rf)
            elif self.use_pred_risk_factors:
                risk_factor_hidden.append(rf_preds)
            else:
                risk_factor_hidden.append(batch["risk_factors"][indx])

        hidden = torch.concat(risk_factor_hidden, dim=-1)
        output["hidden"] = hidden
        output["logit"] = self.risk_layer(hidden)

        return output

@register_object("full_sybilv2", "model")
class SybilFullv2(nn.Module):
    def __init__(self, args):
        super(SybilFullv2, self).__init__()
        self.args = args
        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.pool = MultiAttentionPool()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        self.use_pred_risk_factors = args.use_pred_risk_factors
        self.use_available_risk_factors = args.use_available_risk_factors

        num_risk_factor_features = 0
        for key in args.risk_factor_keys:
            num_key_features = args.risk_factor_key_to_num_class[key]
            num_risk_factor_features+=num_key_features
            key_fc = nn.Linear(512, num_key_features)
            self.add_module("{}_fc".format(key), key_fc)

        self.img_hidden_dim = 512
        self.risk_factor_hidden_dim = num_risk_factor_features
        
        self.clinical_risk_layer = Cumulative_Probability_Layer(self.risk_factor_hidden_dim, args, max_followup=args.max_followup)
        self.final_risk_layer = Cumulative_Probability_Layer(args.max_followup + self.img_hidden_dim, args, max_followup=args.max_followup)
    
    def forward(self, x, batch):
        output = {}
        x = self.image_encoder(x)
        output = self.pool(x)
        output["activ"] = x

        img_hidden = output["hidden"]
        output["img_hidden"] = self.relu(self.dropout(img_hidden))
        risk_factor_hidden = []
        for indx, key in enumerate(self.args.risk_factor_keys):
            logit = self._modules["{}_fc".format(key)](img_hidden)
            rf_preds = F.softmax(logit, dim = -1)
            output["{}_logit".format(key)] = logit
            if self.use_available_risk_factors:
                # use if known
                is_rf_known = (torch.sum(batch["risk_factors"][indx], dim=-1) > 0).unsqueeze(-1).float()
                rf = is_rf_known * batch["risk_factors"][indx] + (1-is_rf_known) * rf_preds
                risk_factor_hidden.append(rf)
            elif self.use_pred_risk_factors:
                risk_factor_hidden.append(rf_preds)
            else:
                risk_factor_hidden.append(batch["risk_factors"][indx])

        risk_factor_hidden = torch.concat(risk_factor_hidden, dim=-1)

        # img_logit= self.risk_layer1(img_hidden)
        rf_logit = self.clinical_risk_layer(risk_factor_hidden)
        output['risk_factor_logit']  = rf_logit
        prelogit = torch.concat([output["img_hidden"],rf_logit], dim = -1)
        output["logit"] = self.final_risk_layer(prelogit)

        return output
