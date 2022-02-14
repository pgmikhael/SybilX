import torch
import torch.nn as nn
import copy
from modules.utils.shared import register_object, get_object
from modules.models.layers.cumulative_probability_layer import Cumulative_Probability_Layer


@register_object('fc_classifier', 'model')
class FC(nn.Module):
    def __init__(self, args):
        super(FC, self).__init__()

        self.args = args
        model_layers = []
        cur_dim = args.fc_classifier_input_dim
        model_layers.append(nn.Linear(args.fc_classifier_input_dim, args.num_classes))
        
        self.predictor = nn.Sequential(*model_layers)
    
    def forward(self, x, batch=None):
        output = {}
        output['logit'] = self.predictor(x.float())
        return output

@register_object('mlp', 'model')
class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.args = args

        model_layers = []
        cur_dim = args.mlp_input_dim

        if args.mlp_use_layer_norm:
            model_layers.append(nn.LayerNorm(cur_dim))

        if args.mlp_use_batch_norm:
            model_layers.append(nn.BatchNorm1d(cur_dim))

        for layer_size in args.mlp_layer_configuration[:-1]:
            model_layers.extend( self.append_layer(cur_dim, layer_size, args) )
            cur_dim = layer_size
        
        layer_size = args.mlp_layer_configuration[-1]
        model_layers.extend( self.append_layer(cur_dim, layer_size, args, with_dropout = False) )

        self.predictor = nn.Sequential(*model_layers)
    
    def append_layer(self, cur_dim, layer_size, args, with_dropout = True):
        linear_layer = nn.Linear(cur_dim, layer_size)
        bn = nn.BatchNorm1d(layer_size)
        ln = nn.LayerNorm(layer_size)
        if args.mlp_use_batch_norm:
            seq = [linear_layer, bn, nn.ReLU()]
        elif args.mlp_use_layer_norm:
            seq = [linear_layer, ln, nn.ReLU()]
        else:
            seq = [linear_layer, nn.ReLU()]
        if with_dropout:
            seq.append(nn.Dropout(p=args.dropout))
        return seq
    
    def forward(self, x, batch=None):
        output = {}
        z = self.predictor(x.float())
        output['logit'] = z
        return output

@register_object('mlp_classifier', 'model')
class MLPClassifier(nn.Module):
    def __init__(self, args):
        super(MLPClassifier, self).__init__()

        self.args = args
        
        self.mlp = MLP(args)
        self.dropout = nn.Dropout(p=args.dropout)
        args.fc_classifier_input_dim = args.mlp_layer_configuration[-1] 
        self.predictor = FC(args)
    
    def forward(self, x, batch=None):
        output = {}
        z = self.dropout( self.mlp(x)['logit'] )
        output['logit'] = self.predictor(z)['logit']
        output['hidden'] = z
        return output

@register_object('multitask_mlp', 'model')
class MultiTaskMLP(nn.Module):
    def __init__(self, args):
        super(MultiTaskMLP, self).__init__()
        assert hasattr(args, 'multitask_keys') and hasattr(args, 'multitask_num_classes'), 'MULTI-TASK CLASSIFIER MISSING ARGS [multiclass_keys AND/OR multiclass_num_classes]'
        self.args = args
        self.multitask_keys = args.multitask_keys
        multitask_args = copy.deepcopy(args)
        multitask_args.mlp_input_dim = args.multitask_mlp_input_dim

        for key, num_classes in zip(args.multitask_keys, args.multitask_num_classes):
            multitask_args.mlp_layer_configuration = copy.copy(args.multitask_mlp_configuration)
            multitask_args.mlp_layer_configuration.append( num_classes )
            mlp = MLP(multitask_args)
            self.add_module('{}_mlp'.format(key), mlp )
    
    def forward(self, x, batch=None):
        output = {}
        for key in self.multitask_keys:
            z = self._modules['{}_mlp'.format(key)](x.float(), batch)
            output['{}_logit'.format(key)] = z['logit']
        return output

@register_object('multitask_classifier', 'model')
class MultiTaskFC(nn.Module):
    def __init__(self, args):
        super(MultiTaskFC, self).__init__()
        assert hasattr(args, 'multitask_keys') and hasattr(args, 'multitask_num_classes'), 'MULTI-TASK CLASSIFIER MISSING ARGS [multiclass_keys AND/OR multiclass_num_classes]'
        self.args = args
        self.multitask_keys = args.multitask_keys
        multitask_args = copy.copy(args)
        multitask_args.mlp_input_dim = args.multitask_mlp_input_dim
        multitask_args.mlp_layer_configuration = args.multitask_mlp_configuration

        for key, num_classes in zip(args.multitask_keys, args.multitask_num_classes):
            mlp = MLP(multitask_args)
            self.add_module('{}_mlp'.format(key), mlp )
            fc = nn.Linear(args.multitask_mlp_configuration[-1], num_classes)
            self.add_module('{}_fc'.format(key), fc )
    
    def forward(self, x, batch=None):
        output = {}
        for key in self.multitask_keys:
            z = self._modules['{}_mlp'.format(key)](x.float(), batch)
            output['{}_hidden'.format(key)] = z['logit']
            output['{}_logit'.format(key)] = self._modules['{}_fc'.format(key)](z['logit'])
        return output

