#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
""" Utility functions for training DNNs with data parameters"""
import os
import json
import shutil
import random
import wandb
import torch
import numpy as np

from . import sparse_sgd

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_topk_accuracy(prediction, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        prediction (torch.Tensor): N*C tensor, contains logits for N samples over C classes.
        target (torch.Tensor):  labels for each row in prediction.
        topk (tuple of int): different values of k for which top-k accuracy should be computed.

    Returns:
        result (tuple of float): accuracy at different top-k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def save_artifacts(args, epoch, model, class_parameters, inst_parameters):
    """Saves model and data parameters.

    Args:
        args (argparse.Namespace):
        epoch (int): current epoch
        model (torch.nn.Module): DNN model.
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters.
    """
    artifacts = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'class_parameters': class_parameters.cpu().detach().numpy(),
            'inst_parameters': inst_parameters.cpu().detach().numpy()
             }

    file_path = args.save_dir + '/epoch_{}.pth.tar'.format(epoch)
    torch.save(obj=artifacts, f=file_path)


def save_config(save_dir, cfg):
    """Save config file to disk at save_dir.

    Args:
        save_dir (str): path to directory.
        cfg (dict): config file.
    """
    save_path = save_dir + '/config.json'
    if os.path.isfile(save_path):
        raise Exception("Expected an empty folder but found an existing config file.., aborting")
    with open(save_path,  'w') as outfile:
        json.dump(cfg, outfile)


def generate_save_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nModel artifacts (checkpoints and config) are going to be saved in: {}'.format(args.save_dir))
    if os.path.exists(args.save_dir):
        if args.restart:
            print('Deleting old model artifacts found in: {}'.format(args.save_dir))
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir)
        else:
            error='Old artifacts found; pass --restart flag to erase'.format(args.save_dir)
            raise Exception(error)
    else:
        os.makedirs(args.save_dir)


def set_seed(args):
    """Set seed to ensure deterministic runs.

    Note: Setting torch to be deterministic can lead to slow down in training.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_inst_data_params_n_optimizer(config,
    init_class_param, learn_class_parameters, lr_class_param,
    init_inst_param, learn_inst_parameters, lr_inst_param,
    nr_classes, nr_instances,
    device):
    """Returns class and instance level data parameters and their corresponding optimizers.

    Args:
        args (argparse.Namespace):
        nr_classes (int):  number of classes in dataset.
        nr_instances (int): number of instances in dataset.
        device (str): device on which data parameters should be placed.

    Returns:
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters
        optimizer_class_param (SparseSGD): Sparse SGD optimizer for class parameters
        optimizer_inst_param (SparseSGD): Sparse SGD optimizer for instance parameters
    """
    # class-parameter
    class_parameters = torch.tensor(np.ones(nr_classes) * np.log(init_class_param),
                                    dtype=torch.float32,
                                    requires_grad=learn_class_parameters,
                                    device=device)
    optimizer_class_param = sparse_sgd.SparseSGD([class_parameters],
                                      lr=lr_class_param,
                                      momentum=config.data_param_optim_momentum,
                                      skip_update_zero_grad=True)
    if learn_class_parameters:
        print('Initialized class_parameters with: {}'.format(init_class_param))
        print('optimizer_class_param:')
        print(optimizer_class_param)

    # instance-parameter
    inst_parameters = torch.tensor(np.ones(nr_instances) * np.log(init_inst_param),
                                   dtype=torch.float32,
                                   requires_grad=learn_inst_parameters,
                                   device=device)
    optimizer_inst_param = sparse_sgd.SparseSGD([inst_parameters],
                                     lr=lr_inst_param,
                                     momentum=config.data_param_optim_momentum,
                                     skip_update_zero_grad=True)
    if learn_inst_parameters:
        print('Initialized inst_parameters with: {}'.format(init_inst_param))
        print('optimizer_inst_param:')
        print(optimizer_inst_param)

    return class_parameters, inst_parameters, optimizer_class_param, optimizer_inst_param


def get_data_param_for_minibatch(learn_class_parameters, learn_inst_parameters,
                                 class_param_minibatch, inst_param_minibatch):
    """Returns the effective data parameter for instances in a minibatch as per the specified curriculum.

    Args:
        args (argparse.Namespace):
        class_param_minibatch (torch.Tensor): class level parameters for samples in minibatch.
        inst_param_minibatch (torch.Tensor): instance level parameters for samples in minibatch.

    Returns:
        effective_data_param_minibatch (torch.Tensor): data parameter for samples in the minibatch.
    """
    sigma_class_minibatch = torch.exp(class_param_minibatch).view(-1, 1)
    sigma_inst_minibatch = torch.exp(inst_param_minibatch).view(-1, 1)

    if learn_class_parameters and learn_inst_parameters:
        # Joint curriculum
        effective_data_param_minibatch = sigma_class_minibatch + sigma_inst_minibatch
    elif learn_class_parameters:
        # Class level curriculum
        effective_data_param_minibatch = sigma_class_minibatch
    elif learn_inst_parameters:
        # Instance level curriculum
        effective_data_param_minibatch = sigma_inst_minibatch
    else:
        # This corresponds to the baseline case without data parameters
        effective_data_param_minibatch = 1.0

    return effective_data_param_minibatch


def apply_weight_decay_data_parameters(
    learn_inst_parameters, wd_inst_param,
    learn_class_parameters, wd_class_param,
    loss, class_parameter_minibatch, inst_parameter_minibatch):
    """Applies weight decay on class and instance level data parameters.

    We apply weight decay on only those data parameters which participate in a mini-batch.
    To apply weight-decay on a subset of data parameters, we explicitly include l2 penalty on these data
    parameters in the computational graph. Note, l2 penalty is applied in log domain. This encourages
    data parameters to stay close to value 1, and prevents data parameters from obtaining very high or
    low values.

    Args:
        args (argparse.Namespace):
        loss (torch.Tensor): loss of DNN model during forward.
        class_parameter_minibatch (torch.Tensor): class level parameters for samples in minibatch.
        inst_parameter_minibatch (torch.Tensor): instance level parameters for samples in minibatch.

    Returns:
        loss (torch.Tensor): loss augmented with l2 penalty on data parameters.
    """

    # Loss due to weight decay on instance-parameters
    if learn_inst_parameters and wd_inst_param > 0.0:
        loss = loss + 0.5 * wd_inst_param * (inst_parameter_minibatch ** 2).sum()

    # Loss due to weight decay on class-parameters
    if learn_class_parameters and wd_class_param > 0.0:
        # (We apply weight-decay to only those classes which are present in the mini-batch)
        loss = loss + 0.5 * wd_class_param * (class_parameter_minibatch ** 2).sum()

    return loss


def clamp_data_parameters(
    skip_clamp_data_param, learn_inst_parameters, learn_class_parameters,
    class_parameters, inst_parameters,
    clamp_inst_sigma_config, clamp_cls_sigma_config):
    """Clamps class and instance level parameters within specified range.

    Args:
        args (argparse.Namespace):
        class_parameters (torch.Tensor): class level parameters.
        inst_parameters (torch.Tensor): instance level parameters.
        config (dict): config file for the experiment.
    """
    if skip_clamp_data_param is False:
        if learn_inst_parameters:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=clamp_inst_sigma_config['min'],
                max=clamp_inst_sigma_config['max'])
        if learn_class_parameters:
            # Project the sigma's to be within certain range
            class_parameters.data = class_parameters.data.clamp_(
                min=clamp_cls_sigma_config['min'],
                max=clamp_cls_sigma_config['max'])


def log_stats(data, name, step):
    """Logs statistics on tensorboard for data tensor.

    Args:
        data (torch.Tensor): torch tensor.
        name (str): name under which stats for the tensor should be logged.
        step (int): step used for logging
    """
    wandb.log({'{}/highest'.format(name): torch.max(data).item()}, step=step)
    wandb.log({'{}/lowest'.format(name): torch.min(data).item()},  step=step)
    wandb.log({'{}/mean'.format(name): torch.mean(data).item()},   step=step)
    wandb.log({'{}/std'.format(name): torch.std(data).item()},     step=step)
    # wandb.log({'{}'.format(name): wandb.histogram(data.data.cpu().numpy())}, step=step)


def log_intermediate_iteration_stats(log_path_prefix, log_path_postfix, epx, learn_class_parameters, learn_inst_parameters,
                                     class_parameters, inst_parameters, top1=None, top5=None):
    """Log stats for data parameters and loss on tensorboard."""
    if top5 is not None:
        wandb.log({'scores/accuracy_top5' + log_path_postfix: top5.avg}, step=epx)
    if top1 is not None:
        wandb.log({'scores/accuracy_top1' + log_path_postfix: top1.avg}, step=epx)

    # Log temperature stats
    if learn_class_parameters:
        log_stats(data=torch.exp(class_parameters),
                  name=log_path_prefix + 'class_parameters' + log_path_postfix,
                  step=epx)
    if learn_inst_parameters:
        log_stats(data=torch.exp(inst_parameters),
                  name=log_path_prefix + 'inst_parameters' + log_path_postfix,
                  step=epx)
