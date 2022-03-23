import torch
import torch.nn as nn
from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.utils.registry import register_object, get_object
import copy

@register_object("sybilx_teacher_student", "model")
class SybilXTeacherStudent(nn.Module):
    def __init__(self, args):
        super(SybilXTeacherStudent, self).__init__()

        self.args = args
        # SybilXrayInception
        self.projection_encoder = get_object(args.proj_encoder_model, 'model')(args)

        # Sybil
        self.ct_encoder = get_object("sybil" , 'model')(args)

        targs = copy.deepcopy(args)
        targs.mlp_input_dim = args.mlp_input_dim + args.hidden_size
        self.mlp = get_object('mlp', 'model')(targs)

        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            targs.mlp_layer_configuration[-1], args, max_followup=args.max_followup
        )

    def forward(self, x, batch = None):
        output = {}
        # must use 'hidden' here to not use cumulative probablity layer, note includes dropout and activation
        # expects batch['ct'] and batch['projection']
        output['ct_hidden'] = self.ct_encoder(batch['ct'])['hidden']
        output['proj_hidden'] = self.projection_encoder(batch['projection'])['hidden']

        combined_encoding = torch.cat([output['ct_hidden'], output['proj_hidden']], dim=-1)

        output['hidden'] = self.mlp(combined_encoding)['logit']
        output["logit"] = self.prob_of_failure_layer(output["hidden"])

        return output