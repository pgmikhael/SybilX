import torch
import torch.nn as nn


class AlignmentMLP(nn.Module):
    """
    Simple MLP discriminator to be used as adversary for alignment of hiddens.
    """

    def __init__(self, args):
        super(AlignmentMLP, self).__init__()
        self.args = args

        # calculate input size based on chosn layers (default: just 'hidden')
        discrim_input_size = 512 + args.num_classes if args.adv_conditional else 512

        # init discriminator
        self.model = nn.Sequential(
            nn.Linear(discrim_input_size, 512),
            nn.BatchNorm1d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(256, args.adv_num_classes),
        )

    def forward(self, model_output, batch=None):
        # concatenate hiddens of chosen layers
        label_logit = model_output["logit"].detach()
        hiddens = (
            torch.cat([model_output["hidden"], label_logit], dim=-1)
            if self.args.adv_conditional
            else model_output["hidden"]
        )
        # pass hiddens through mlp
        output = {"logit": self.model(hiddens)}

        return output


class MultiAlignmentMLP(nn.Module):
    def __init__(self, args):
        super(MultiAlignmentMLP, self).__init__()
        self.args = args

        # calculate input size based on chosn layers (default: just 'hidden')
        discrim_input_size = 512 + args.num_classes if args.adv_conditional else 512

        # init discriminator
        self.device_model = nn.Sequential(
            nn.Linear(discrim_input_size, 512),
            nn.BatchNorm1d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7),
        )

        self.thickness_model = nn.Sequential(
            nn.Linear(discrim_input_size, 512),
            nn.BatchNorm1d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

    def forward(self, model_output, batch=None):
        # concatenate hiddens of chosen layers
        label_logit = model_output["logit"].detach()
        hiddens = (
            torch.cat([model_output["hidden"], label_logit], dim=-1)
            if self.args.adv_conditional
            else model_output["hidden"]
        )
        # pass hiddens through mlp
        output = {
            "device_logit": self.device_model(hiddens),
            "thickness_logit": self.thickness_model(hiddens),
        }

        return output
