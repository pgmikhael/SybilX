from sybilx.utils.registry import register_object
from sybilx.learning.losses.basic import get_cross_entropy_loss, get_survival_loss, get_corn_survival_loss
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import pdb

EPSILON = 1e-6

@register_object("vanilla_knowledge_distillation_loss", 'loss')
def get_knowledge_distillation_loss(model_output, batch, model, args):
    """
    Computes distillation loss from original "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531v1
    
    NOTE: Includes CE or Survival loss, does not need to be run in conjunction with label loss fn
    as it computes a weighted average of distilled loss and regular loss (ce or survival)

    Returns:
        loss: 
        l_dict (dict): dictionary of loss values for each loss (detached from computation graph)
        p_dict (dict): dictionary of predictions (subclass_probs, probs, preds) and the ground truth (golds)
    """
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    
    y_pred_student = model_output['logit']
    y_pred_teacher = batch['teacher_logit']

    soft_teacher_out = torch.sigmoid(y_pred_teacher / args.distill_temperature)
    # soft_student_out = F.softmax(y_pred_student / args.distill_temperature, dim=1)
    soft_student_out = y_pred_student / args.distill_temperature


    if "survival" in args.loss_fns:
        ce_loss, _, prob_dict = get_survival_loss(model_output, batch, model, args)
        p_dict['censors'] = prob_dict['censors']
    elif "corn" in args.loss_fns:
        ce_loss, _, prob_dict = get_corn_survival_loss(model_output, batch, model, args)
        p_dict['censors'] = prob_dict['censors']
    else:
        # if not using survival setup then need to binarize teacher output
        # select last value which corresponds to 6 yr risk
        soft_teacher_out = soft_teacher_out[:, -1].unsqueeze(-1) 
        complement = 1 - soft_teacher_out
        # index 0 is prob of no cancer, index 1 is prob of 6 year cancer
        soft_teacher_out = torch.cat([complement, soft_teacher_out], dim=1) # B, 2

        ce_loss, _, prob_dict = get_cross_entropy_loss(model_output, batch, model, args)
        l_dict['cross_entropy'] = ce_loss.detach()

    distill_loss = (args.distill_temperature ** 2) * F.cross_entropy(soft_student_out, soft_teacher_out)
    l_dict['distill_loss'] = distill_loss.detach()

    p_dict['probs'] = prob_dict['probs']
    p_dict['golds'] = prob_dict['golds']
    p_dict['preds'] = prob_dict['preds']
    # weighted average: lambda * distill_loss + (1-lambda) * ce_loss
    loss = args.distill_student_loss_lambda *  distill_loss + (1 - args.distill_student_loss_lambda) * ce_loss
    l_dict['student_loss'] = loss.detach()

    return loss, l_dict, p_dict

@register_object("mse_knowledge_distillation_loss", 'loss')
def get_mse_knowledge_distillation_loss(model_output, batch, model, args):
    """
    differs from vanilla kd loss above because it uses mse loss of hiddens rather than cross entropy loss of soft targets
    """
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    
    y_pred_student = model_output['hidden']
    # below should be batch not model_ouput if loading saved hiddens
    y_pred_teacher = batch['teacher_hidden']

    if "survival" in args.loss_fns:
        ce_loss, _, prob_dict = get_survival_loss(model_output, batch, model, args)
        p_dict['censors'] = prob_dict['censors']
    elif "corn" in args.loss_fns:
        ce_loss, _, prob_dict = get_corn_survival_loss(model_output, batch, model, args)
        p_dict['censors'] = prob_dict['censors']
    else:
        ce_loss, _, prob_dict = get_cross_entropy_loss(model_output, batch, model, args)
        l_dict['cross_entropy'] = ce_loss.detach()
    
    assert y_pred_student.shape == y_pred_teacher.shape, "hiddens are not the same shape, so cant compute MSE"
    distill_loss = F.mse_loss(y_pred_student, y_pred_teacher)
    l_dict['distill_loss'] = distill_loss.detach()
    
    p_dict['probs'] = prob_dict['probs']
    p_dict['golds'] = prob_dict['golds']
    p_dict['preds'] = prob_dict['preds']
    # weighted average: lambda * distill_loss + (1-lambda) * ce_loss
    loss = args.distill_student_loss_lambda *  distill_loss + (1 - args.distill_student_loss_lambda) * ce_loss
    l_dict['student_loss'] = loss.detach()

    return loss, l_dict, p_dict

@register_object("subclass_distillation_teacher_loss", 'loss')
def get_subclass_distill_loss_teacher(model_output, batch, model, args):
    '''
    Computes subclass distillation loss for teacher model
    adapted from: https://arxiv.org/abs/2002.03936 

    Returns:
        loss: linear combination of cross entropy loss and auxiliary loss
        l_dict (dict): dictionary of loss values for each loss (detached from computation graph)
        p_dict (dict): dictionary of predictions (subclass_probs, probs, preds) and the ground truth (golds)
    '''
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    
    # Cross Entropy Loss
    logit = model_output['logit'] # B, num_class * num_subclasses
    B, C = logit.shape
    batch['y'] = batch['y'].long()
    
    subclass_probs = torch.softmax(logit / args.distill_temperature, dim = -1)

    p_dict['subclass_probs'] = subclass_probs.detach()

    # split each class into distill_num_subclasses
    # sum then transpose to get correct shape
    class_probs = torch.stack([l.sum(-1) for l in torch.split(subclass_probs, args.distill_num_subclasses, dim = -1)]).T # B, num_classes
    
    class_logit = torch.log(class_probs)
    if not args.predict:
        ce_loss = F.nll_loss(class_logit, batch['y'])
        l_dict['cross_entropy_loss'] = ce_loss.detach()

    probs, preds = torch.topk(class_probs, k = 1)
    p_dict['probs'], p_dict['preds'] = class_probs[:, -1].view(B).detach().view(-1), preds.view(B).detach().view(-1)

    p_dict['golds'] = batch['y']
    if 'has_y' in batch:
        p_dict['has_y'] = batch['has_y']


    # Auxiliary Loss
    if not args.predict:
        logit = F.normalize(logit, dim = -1) # normalize to unit 1 length
        cov = torch.matmul(logit, logit.T) # equivalent to torch.einsum('ib, jb -> ij', logit, logit) pairwise dot-product
        cov = torch.exp(cov/args.distill_aux_temperature) # exponentiate
        cov = torch.log( torch.sum(cov, dim = -1) ) # sum across columns and log
        cov = torch.mean(cov) # mean across rows
        aux_loss = cov - 1/args.distill_aux_temperature - np.log(B)
        l_dict['auxiliary_loss'] = aux_loss.detach()
        loss = args.distill_aux_loss_lambda * aux_loss + ce_loss
        l_dict['teacher_distill_loss'] = loss.detach()
    
    return loss, l_dict, p_dict

@register_object("subclass_distillation_student_loss", 'loss')
def get_subclass_distill_loss_student(model_output, batch, model, args):
    '''
    Computes subclass distillation loss for student model
    adapted from: https://arxiv.org/abs/2002.03936 

    Returns:
        loss: linear combination of cross entropy loss with true labels and teacher-generated (soft) labels
        l_dict (dict): dictionary of loss values for each loss (detached from computation graph)
        p_dict (dict): dictionary of predictions (probs, preds) and the ground truth (golds)
    '''
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    
    logit = model_output['logit'] # B, num_class * num_subclasses
    B, C = logit.shape
    
    # Cross Entropy Loss for actual classes
    batch['y'] = batch['y'].long()
    subclass_probs = torch.softmax(logit / args.distill_temperature, dim = -1)
    
    # split each class into distill_num_subclasses
    # sum then transpose to get correct shape
    class_probs = torch.stack([l.sum(-1) for l in torch.split(subclass_probs, args.distill_num_subclasses, dim = -1)]).T # B, num_classes
    
    class_logit = torch.log(class_probs)
    if not args.predict:
        ce_loss = F.nll_loss(class_logit, batch['y'])
        l_dict['cross_entropy_loss'] = ce_loss.detach()

    probs, preds = torch.topk(class_probs, k = 1)
    p_dict['probs'], p_dict['preds'] = class_probs[:, -1].view(B).detach().view(-1), preds.view(B).detach().view(-1)

    p_dict['golds'] = batch['y']

    # Cross Entropy Loss for subclasses
    if not args.predict:
        cross_entropy = F.log_softmax(logit, dim = -1) * batch['subclass_y']
        distill_ce_loss = (args.distill_temperature ** 2) * torch.mean( -torch.sum(cross_entropy, dim = -1) )
        l_dict['distill_ce_loss'] = distill_ce_loss.detach()

        loss = distill_ce_loss * args.distill_student_loss_lambda + (1 - args.distill_student_loss_lambda) * ce_loss
        l_dict['student_distill_loss'] = loss.detach()
    
    return loss, l_dict, p_dict
    
    
@register_object("mmd_loss", 'loss')
def get_mmd_loss(model_output, batch, model, args):
    """
    Source: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
    
    Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    x = batch['y']
    y = model_output['logit']

    # similarity matrices 
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy 
    dxy = rx.t() + ry - 2. * zz 
    
    XX, YY, XY = (torch.zeros(xx.shape, device = xx.device ),
                  torch.zeros(xx.shape, device = xx.device),
                  torch.zeros(xx.shape, device = xx.device))
    
    # TODO: verify this 
    if args.mmd_kernel == "linear":
        XX += dxx
        YY += dyy
        XY += dxy

    if args.mmd_kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if args.mmd_kernel == "guassian_rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)



@register_object("nst_distillation_loss", 'loss')
def get_nst_loss(model_output, batch, model, args):
    """
    Based on 'Like What You Like: Knowledge Distill via Neuron Selectivity Transfer'
    source: https://arxiv.org/pdf/1707.01219.pdf

    """
    l_dict, p_dict = OrderedDict(), OrderedDict()
    loss = 0
    logit = model_output['logit'] # B, num_class * num_subclasses
    B, C = logit.shape
    
    if not args.predict:
        mmd_loss = get_mmd_loss(model_output, batch, model, args)
        l_dict['distill_mmd_loss'] = mmd_loss.detach()
        loss += (args.nst_lambda / 2) * mmd_loss
        l_dict['student_distill_loss'] = loss.detach()
    
    return loss, l_dict, p_dict

@register_object("itrd_distillation_loss", 'loss')
def get_itrd_loss(model_output, batch, model, args):
    """
    Based on 'Information Theoretic Representation Distillation (ITRD)'
    source: https://arxiv.org/pdf/2112.00459.pdf
    """
    l_dict, p_dict = OrderedDict(), OrderedDict()
    loss = 0
    logit = model_output['logit'] # B, num_class * num_subclasses
    B, C = logit.shape
    
    if not args.predict:
        z_s = logit
        z_t = batch['y']

        # Compute correlation loss
        corr_loss = correlation_loss(z_s, z_t, args.itrd_alpha)
        loss += corr_loss.mul( args.itrd_beta_corr )
        l_dict['corr_loss'] = corr_loss.detach()

        # Compute the mutual information loss 
        mi_loss = mutual_information_loss(z_s, z_t)
        loss += mi_loss.sum().mul( args.itrd_beta_mi )
        l_dict['mutual_info_loss'] = mi_loss.detach()

    return loss, l_dict, p_dict

    
def correlation_loss(z1, z2, alpha, maximize = True):
    """
    z1: batch size x hidden dim
    z2: batch size x hidden dim
    """
    B, d = z1.shape
    # normalize
    z1_norm = (z1 - torch.mean(z1, dim = 0))/( torch.std(z1, dim = 0) + EPSILON)
    z2_norm = (z2 - torch.mean(z2.float(), dim = 0))/( torch.std(z2.float(), dim = 0) + EPSILON)
    # cross-correlation
    correlation = torch.einsum('bx, bx -> x', z1_norm, z2_norm ) / B
    # loss
    if maximize:
        loss = torch.log2( correlation.add(-1).pow(2).pow(alpha).sum() )
    else:
        loss = torch.log2( correlation.pow(2).pow(alpha).sum() )
    return loss

def mutual_information_loss(z1, z2, maximize = True):
    """
    optimize mutual information: https://arxiv.org/abs/2112.00459 
    z1: batch size x hidden dim
    z2: batch size x hidden dim
    """
    z1_norm = F.normalize(z1, p = 2)
    z2_norm = F.normalize(z2, p = 2)
    g_p = torch.einsum('bx, dx -> bd' , z1_norm, z1_norm)
    g_n = torch.einsum('bx, dx -> bd' , z2_norm, z2_norm)
    g_pn = g_p * g_n
    g_p = g_p / torch.trace(g_p)
    # TODO: ask peter about this, original paper includes but new paper excludes
    # g_n = g_n / torch.trace(g_n)
    g_pn = g_pn / torch.trace(g_pn)
    # mi loss
    mi = g_pn.pow(2) - g_p.pow(2) # - g_n.pow(2)
    if maximize:
        loss = mi.sum()
    else:
        loss = (-mi).sum()
    return loss
