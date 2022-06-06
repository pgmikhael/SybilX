import torch
import torch.nn as nn
from sybilx.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybilx.utils.registry import register_object, get_object
import copy


# Note, this was never used, it is the CellImager style projection + CT Teacher
@register_object("sybilx_teacher", "model")
class SybilXTeacher(nn.Module):
    def __init__(self, args):
        super(SybilXTeacher, self).__init__()

        self.args = args
        # SybilXrayInception
        self.projection_encoder = get_object(args.proj_encoder_model, 'model')(args)

        # Sybil
        self.ct_encoder = get_object("sybil" , 'model')(args)

        targs = copy.deepcopy(args)
        
        # sybil hiddens are hard-coded 512
        targs.mlp_input_dim = 512 + args.hidden_size
        self.mlp = get_object('mlp', 'model')(targs)

        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            targs.mlp_layer_configuration[-1], args, max_followup=args.max_followup
        )

    def forward(self, x, batch = None):
        output = {}
        # must use 'hidden' here to not use cumulative probablity layer, note includes dropout and activation
        # expects batch['ct'] and batch['projection']
        ct_encoded = self.ct_encoder(x)
        output['ct_hidden'] = ct_encoded['hidden']
        output.update(ct_encoded)

        projection_encoded = self.projection_encoder(batch['projection'])
        output['proj_hidden'] = projection_encoded['hidden']
        output.update(projection_encoded)

        combined_encoding = torch.cat([output['ct_hidden'], output['proj_hidden']], dim=-1)
        output['hidden'] = self.mlp(combined_encoding)['logit']
        output["logit"] = self.prob_of_failure_layer(output["hidden"])

        return output


@register_object("sybilx_student", "model")
class SybilXStudent(nn.Module):
    def __init__(self, args):
        super(SybilXStudent, self).__init__()

        self.args = args
        # SybilXrayInception
        self.projection_encoder = get_object(args.proj_encoder_model, 'model')(args)

        # Sybil
        self.encoder_args = copy.deepcopy(args)
        self.encoder_args.base_model = "sybil"
        self.ct_encoder = get_object(self.encoder_args.lightning_name, "lightning")(self.encoder_args)
        modelpath = "/Mounts/rbg-storage1/snapshots/lung_ct/28a7cd44f5bcd3e6cc760b65c7e0d54d/28a7cd44f5bcd3e6cc760b65c7e0d54depoch=10.ckpt"

        self.ct_encoder = self.ct_encoder.load_from_checkpoint(
            checkpoint_path=modelpath, strict=not self.encoder_args.relax_checkpoint_matching
        )
        self.ct_encoder.args = self.encoder_args

    def forward(self, x, batch = None):
        output = {}
        with torch.no_grad():
            ct_encoded = self.ct_encoder(x)
            output['teacher_hidden'] = ct_encoded['hidden']

        projection_encoded = self.projection_encoder(batch['projection'])
        output.update(projection_encoded)

        return output