import copy
import torch
import torch.nn as nn
from torchvision.models import resnet18
from sybilx.utils.registry import register_object, get_object

CKPT_PATH = "/Mounts/rbg-storage1/users/itamarc/SybilX/r8w-00001.pth.tar"
CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"
                  ]

@register_object("pretrained_moco", "model")
class PretrainedMoCo(nn.Module):
    def __init__(self, args):
        super(PretrainedMoCo, self).__init__()
        self.args = args
        ckpt_dict = torch.load(CKPT_PATH)
        
        encoder = resnet18() # TODO: check difference with resnet18(CHEXPERT_TASKS, args)

        # load weights
        state_dict = ckpt_dict['state_dict']
        for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            elif 'encoder_k' in k or 'module.queue' in k:
                del state_dict[k]
            elif k.startswith('module.encoder_q.fc'):
                del state_dict[k]
        # for k in list(state_dict.keys()):        
        #     state_dict[k.replace('module.model.', '')] = state_dict[k]
        #     del state_dict[k]

        encoder.load_state_dict(state_dict, strict=False)
        # remove FC layer (includes average pooling)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(512, args.num_classes)

    def forward(self, x, batch = None):
        output = {}

        output['hidden'] = self.image_encoder(x).squeeze(-1).squeeze(-1)
        output["hidden"] = self.relu(output["hidden"])
        output["hidden"] = self.dropout(output["hidden"])
        output['logit'] = self.fc(output["hidden"])
        
        return output
