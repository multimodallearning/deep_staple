from enum import Enum, auto
import torch

import sparse_sgd

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DataParamMode(Enum):
    ONLY_INSTANCE_PARAMS = auto()
    ONLY_CLASS_PARAMS = auto()
    COMBINED_INSTANCE_CLASS_PARAMS = auto()
    SEPARATE_INSTANCE_CLASS_PARAMS = auto()
    DISABLED = auto()

class DataParamOptim(Enum):
    ADAM = auto()
    SGD = auto()
    SPARSE_SGD = auto()

class DataParameterManager():

    def __init__(self, instance_keys, class_keys, config=None, device='cpu'):

        self.data_param_mode = config.data_param_mode
        self.disabled = False or self.data_param_mode == DataParamMode.DISABLED

        self.instance_keys = instance_keys
        self.class_keys = class_keys

        self.nr_instances = len(instance_keys)
        self.nr_classes = len(class_keys)

        self.init_inst_param = config.init_inst_param
        self.lr_inst_param = config.lr_inst_param

        self.init_class_param = config.init_class_param
        self.lr_class_param = config.lr_class_param

        self.device = device

        # Configure weight decay
        self.wd_inst_param = config.wd_inst_param
        self.wd_class_param = config.wd_class_param

        # Configure data parameter clamping
        self.skip_clamp_data_param = config.skip_clamp_data_param

        self.clamp_sigma_min = config.clamp_sigma_min
        self.clamp_sigma_max = config.clamp_sigma_max

        if config.optim_algorithm == DataParamOptim.SGD:
            assert 'momentum' in config.optim_options, \
                "Data parameter optimization with SGD needs momentum > 0 to be specified "\
                "otherwise optimization will fail."

        self.optim_algorithm = config.optim_algorithm
        self.optim_options = config.optim_options

        # Prepare the data parameters and optimizer
        (self.data_parameters_dict,
         self.dp_optimizer) = self.get_data_params_n_optimizer(device)



    def get_data_params_n_optimizer(self, device):
        """Returns class and instance level data parameters and their corresponding optimizers.

        Args:

        Returns:

        """

        nr_instances = self.nr_instances
        nr_classes = self.nr_classes

        data_parameters_dict = dict()

        if self.data_param_mode == DataParamMode.DISABLED:
            return (None, None)

        elif self.data_param_mode == DataParamMode.ONLY_INSTANCE_PARAMS:
            # Create nr_instances data parameters
            for pinst_idx, inst_key in enumerate(self.instance_keys):
                param = torch.ones(1) * self.init_inst_param
                param = torch.nn.parameter.Parameter(param, requires_grad=True).to(device=device)
                data_parameters_dict[inst_key] = param

            print(f"Initialized instance data parameters with: {self.init_inst_param}")

        elif self.data_param_mode == DataParamMode.ONLY_CLASS_PARAMS:
            # Create nr_classes data parameters
            for pcls_idx, class_key in enumerate(self.class_keys):
                param = torch.ones(1) * self.init_class_param
                param = torch.nn.parameter.Parameter(param, requires_grad=True).to(device=device)
                data_parameters_dict[class_key] = param
                param.stepped_on = 0
            print(f"Initialized class data parameters with: {self.init_class_param}")

        elif self.data_param_mode == DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS:
            # Create nr_instances * nr_classes data parameters
            for pinst_idx, inst_key in enumerate(self.instance_keys):
                cls_dict = {}
                for pcls_idx, class_key in enumerate(self.class_keys):
                    param = torch.ones(1) * (self.init_inst_param + self.init_class_param)
                    param = torch.nn.parameter.Parameter(param, requires_grad=True).to(device=device)
                    cls_dict[class_key] = param

                data_parameters_dict[inst_key] = cls_dict.copy()

            print(f"Initialized combined data parameters with: {self.init_inst_param + self.init_class_param}")

        elif self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
            # Create nr_instances + nr_classes data parameters
            for p_idx, dp_key \
                in enumerate(list(self.instance_keys)+list(self.class_keys)):

                key_prefix='dp_inst:' if p_idx < nr_instances else 'dp_class:'
                init_val = self.init_inst_param \
                    if p_idx < nr_instances else self.init_class_param

                param = torch.ones(1) * init_val
                param = torch.nn.parameter.Parameter(param, requires_grad=True).to(device=device)
                data_parameters_dict[key_prefix+str(dp_key)] = param

            print(f"Initialized instance data parameters with: {self.init_inst_param}")
            print(f"Initialized class data parameters with: {self.init_class_param}")

        else:
            raise ValueError

        # Setup torch.nn.parameter.Parameters
        self.data_parameters_dict = data_parameters_dict

        # Build parameter groups for optimizer
        if self.data_param_mode == DataParamMode.ONLY_INSTANCE_PARAMS:
            param_groups = \
                [{'params': self.get_parameter_list(inst_keys='all'), 'lr': self.lr_inst_param}]

        elif self.data_param_mode == DataParamMode.ONLY_CLASS_PARAMS:
            param_groups = \
                [{'params': self.get_parameter_list(class_keys='all'), 'lr': self.lr_class_param}]

        elif self.data_param_mode == DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS:
            param_groups = \
                [{'params': self.get_parameter_list(inst_keys='all', class_keys='all'), \
                    'lr': max(self.lr_inst_param, self.lr_class_param)}]

        elif self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
            param_groups = \
                [{'params': self.get_parameter_list(inst_keys='all'), 'lr': self.lr_inst_param}] \
                + [{'params': self.get_parameter_list(class_keys='all'), 'lr': self.lr_class_param}]

        # Select optimizer
        if self.optim_algorithm == DataParamOptim.ADAM:
            dp_optimizer = torch.optim.Adam(param_groups, **self.optim_options)

        elif self.optim_algorithm == DataParamOptim.SGD:
            dp_optimizer = torch.optim.SGD(param_groups, **self.optim_options)

        else:
            raise(ValueError)

        return data_parameters_dict, dp_optimizer



    def parametrify_logits(self, bare_logits, inst_keys=(), reduced_onehot_targets=()):

        B, *SPATIAL_DIMS, CLS = bare_logits.shape
        num_dims = bare_logits.dim()

        if self.class_keys:
            assert CLS == self.nr_classes, \
                f"Logits shape should be BxSPATIAL_DIMSxCLS but got {bare_logits.shape} "\
                f"with CLS={CLS} != len(self.class_keys)={self.nr_classes}."

        if self.data_param_mode == DataParamMode.ONLY_INSTANCE_PARAMS:
            assert inst_keys != None, "Please specify inst_keys."
            d_params = self.get_parameter_tensor(inst_keys=inst_keys).exp()

            l_shape = torch.Size((B,) + (1,)*(num_dims-1))
            # Logits have shape BxSPATIAL_DIMSxCLS
            # Divide along batch dim
            parametrified_logits = bare_logits / d_params.view(l_shape)

        elif self.data_param_mode == DataParamMode.ONLY_CLASS_PARAMS:

            # Now get only class parameters for class labeled in target of instance in batch
            # Because all instances share their class parameters loading all class parameters
            # also for unlabeled classes will affect untargeted classes
            d_params = self.get_sparse_class_params(inst_keys, reduced_onehot_targets).exp()

            l_shape = torch.Size((B,) + (1,)*(num_dims-2) + (CLS,))
            # Logits have shape BxSPATIAL_DIMSxCLS
            # Divide along class onehot dim
            parametrified_logits = bare_logits / d_params.view(l_shape)

        elif self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
            inst_params = self.get_parameter_tensor(inst_keys=inst_keys).exp()

            # Class params have shape BxCLS
            class_params = self.get_sparse_class_params(inst_keys, reduced_onehot_targets).exp()

            l_shape_inst = torch.Size((B,) + (1,)*(num_dims-1))
            l_shape_class = torch.Size((B,)+ (1,)*(num_dims-2) + (CLS,))
            # Logits have shape BxSPATIAL_DIMSxCLS
            d_params = (
                inst_params.view(l_shape_inst)
                + class_params.view(l_shape_class)
            )

            parametrified_logits = bare_logits / d_params

        elif self.data_param_mode == DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS:
            # OPTION 1: Everytime load all class parameters for every instance. They are not shared.
            # d_params = self.get_parameter_tensor(inst_keys=inst_keys, class_keys='all').exp()

            # # OPTION 2: Only load the specific class parameters which are labeled in the instance target
            d_params = self.get_sparse_class_params(inst_keys, reduced_onehot_targets).exp()

            l_shape = torch.Size((B,) + (1,)*(num_dims-2) + (CLS,))
            # Logits have shape BxSPATIAL_DIMSxCLS
            parametrified_logits = bare_logits / d_params.view(l_shape)

        elif self.data_param_mode == DataParamMode.DISABLED:
            pass

        else:
            raise ValueError

        return parametrified_logits



    def get_sparse_class_params(self, inst_keys, reduced_onehot_targets):
        # Return class params for batch. Only return class params which are
        # referenced in target atlas. Returns torch.Size(BxCLS)
        d_params = []
        for i_key, i_targets in zip(inst_keys, reduced_onehot_targets):
            # Convert class indices to corresponding class keys
            inst_c_keys = [key for o_h, key in zip(i_targets, self.class_keys) \
                if o_h > 0]

            if self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
                # As parameters are split it is not possible to get a class param of an instance here
                d_inst_c_params = self.get_parameter_tensor(
                    class_keys=inst_c_keys, expand_to_full_classes=True
                )
            else:
                d_inst_c_params = self.get_parameter_tensor(
                    inst_keys=[i_key], class_keys=inst_c_keys, expand_to_full_classes=True
                )

            d_params.append(d_inst_c_params)

        return torch.stack(d_params, dim=0)


    def apply_weight_decay(self, loss, inst_keys):
        """Applies weight decay on class and instance level data parameters.

        We apply weight decay on only those data parameters which participate in a mini-batch.
        To apply weight-decay on a subset of data parameters, we explicitly include l2 penalty on these data
        parameters in the computational graph. Note, l2 penalty is applied in log domain. This encourages
        data parameters to stay close to value 1, and prevents data parameters from obtaining very high or
        low values.

        Returns:
            loss (torch.Tensor): loss augmented with l2 penalty on data parameters.
        """

        if self.data_param_mode == None:
            pass

        elif self.data_param_mode == DataParamMode.ONLY_INSTANCE_PARAMS:
            if self.wd_inst_param > .0:
                loss += 0.5 * self.wd_inst_param * (self.get_parameter_tensor(inst_keys=inst_keys) ** 2).sum()

        elif self.data_param_mode == DataParamMode.ONLY_CLASS_PARAMS:
            if self.wd_class_param > .0:
                loss += 0.5 * self.wd_class_param * (self.get_parameter_tensor(class_keys='all').exp() ** 2).sum()

        elif self.data_param_mode == DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS:
            if self.wd_class_param > .0:
                loss += 0.5 * self.wd_class_param * ( self.get_parameter_tensor(inst_keys, 'all').exp() ** 2).sum()

        elif self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
            if self.wd_inst_param > .0:
                loss += 0.5 * self.wd_inst_param * (self.get_parameter_tensor(inst_keys=inst_keys) ** 2).sum()

            if self.wd_class_param > .0:
                loss += 0.5 * self.wd_class_param * (self.get_parameter_tensor(class_keys='all') ** 2).sum()

        else:
            raise ValueError

        return loss



    def clamp(self):
        """Clamps class and instance level parameters within specified range.
        """
        if (self.data_param_mode != None) and (not self.skip_clamp_data_param):

            for param in self.get_flat_parameter_list():
                param.data.clamp_(self.clamp_sigma_min, self.clamp_sigma_max)


    def do_basic_train_step(self, loss_fn, logits, target, optimizer, inst_keys=(),
                            scaler=None):

        assert target.dtype == torch.long, "target must be one-hot-encoded long."

        optimizer.zero_grad()

        if self.disabled:
            loss = loss_fn(logits, target.float())

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            return loss.item()

        else:
            if self.optim_algorithm == DataParamOptim.ADAM:
                self.dp_optimizer.zero_grad(set_to_none=False)

            elif self.optim_algorithm == DataParamOptim.SGD:
                self.dp_optimizer.zero_grad(set_to_none=True)

            # Do only sum over spatial dimensions (not batch and one-hot-dimension)
            reduction_dims = tuple(range(1,target.dim()-1))
            # Get a list of all available class indices in target (inverse one-hot)
            if reduction_dims != ():
                reduced_onehot_targets = target.sum(reduction_dims).clip(0,1)
            else:
                reduced_onehot_targets = target.clip(0,1)

            dp_logits = self.parametrify_logits(logits, inst_keys, reduced_onehot_targets)

            loss = loss_fn(dp_logits, target.float())
            loss = self.apply_weight_decay(loss, inst_keys)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.step(self.dp_optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()
                self.dp_optimizer.step()

            self.clamp()

        return loss.item()



    def get_data_parameters_dict(self) -> dict:
        return self.data_parameters_dict



    def get_flat_parameter_list(self):
        all_params = []
        for item in self.data_parameters_dict.values():
            if isinstance(item, dict):
                # Instance param dict can be nested with class param dicts
                for param in item.values():
                    all_params.append(param)
            else:
                all_params.append(item)

        return all_params



    def get_parameter_tensor(self, inst_keys=(), class_keys=(), expand_to_full_classes=False) -> torch.Tensor:

        i_len = self.nr_instances if inst_keys == 'all' else len(inst_keys)
        c_len = self.nr_classes if (class_keys == 'all' or expand_to_full_classes) else len(class_keys)

        if expand_to_full_classes:
            select_cls_idxs = torch.tensor([c_idx for c_idx, key in enumerate(self.class_keys) if key in class_keys]).long()
        else:
            select_cls_idxs = torch.arange(c_len)

        if self.data_param_mode == DataParamMode.ONLY_INSTANCE_PARAMS:
            assert inst_keys != (), \
            "Please specifiy instance keys for 'DataParamMode.ONLY_INSTANCE_PARAMS'."

            params = self.get_parameter_list(inst_keys=inst_keys, class_keys=class_keys)
            return torch.cat(params)

        elif self.data_param_mode == DataParamMode.ONLY_CLASS_PARAMS:
            assert class_keys != (), \
            "Please specifiy class keys for 'DataParamMode.ONLY_CLASS_PARAMS'."

            # Initialize for sparse class tensor here
            tens = torch.ones((c_len)) * self.init_class_param
            params = self.get_parameter_list(inst_keys=inst_keys, class_keys=class_keys)
            tens[select_cls_idxs] = torch.cat(params)
            return tens

        elif self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
            assert (inst_keys != ()) != (class_keys != ()), \
            "Please specify either instance or class keys for "\
            "'DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS'."

            if inst_keys != ():
                params = self.get_parameter_list(inst_keys=inst_keys)
                tens = torch.cat(params)
            else:
                # Initialize for sparse class tensor here
                tens = torch.ones((c_len)) * self.init_class_param
                params = self.get_parameter_list(class_keys=class_keys)
                tens[select_cls_idxs] = torch.cat(params)
                return tens

            return tens

        elif self.data_param_mode == DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS:
            assert (inst_keys != ()) and (class_keys != ()), \
            "Please specify instance and class keys for "\
            "'DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS'."

            # Initialize for sparse class tensor here
            params = self.get_parameter_list(inst_keys=inst_keys, class_keys=class_keys)
            if expand_to_full_classes:
                params = torch.cat(params).view(i_len, -1)
                tens = torch.ones((i_len, self.nr_classes)) * (self.init_inst_param + self.init_class_param)
            else:
                params = torch.cat(params).view(i_len, c_len)
                tens = torch.ones((i_len, c_len)) * (self.init_inst_param + self.init_class_param)

            tens[:,select_cls_idxs] = params
            return tens

        raise ValueError



    def get_parameter_list(self, inst_keys=(), class_keys=()) -> list:

        if inst_keys == 'all':
            inst_keys = self.instance_keys
        if class_keys == 'all':
            class_keys = self.class_keys

        if self.data_param_mode == DataParamMode.ONLY_INSTANCE_PARAMS:
            assert inst_keys != (), \
            "Please specifiy instance keys for 'DataParamMode.ONLY_INSTANCE_PARAMS'."
            return [self.data_parameters_dict[key] for key in inst_keys]

        elif self.data_param_mode == DataParamMode.ONLY_CLASS_PARAMS:
            assert class_keys != (), \
            "Please specifiy class keys for 'DataParamMode.ONLY_CLASS_PARAMS'."
            return[self.data_parameters_dict[key] for key in class_keys]

        elif self.data_param_mode == DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS:
            assert (inst_keys != ()) and (class_keys != ()), \
            "Please specify instance and class keys for "\
            "'DataParamMode.COMBINED_INSTANCE_CLASS_PARAMS'."

            params = []
            for ikey in inst_keys:
                for ckey in class_keys:
                    params.append(self.data_parameters_dict[ikey][ckey])

            return params

        elif self.data_param_mode == DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS:
            assert (inst_keys != ()) != (class_keys != ()), \
            "Please specify either instance or class keys for "\
            "'DataParamMode.SEPARATE_INSTANCE_CLASS_PARAMS'."

            if inst_keys != ():
                key_prefix = 'dp_inst:'
                dp_keys = inst_keys
            else:
                key_prefix = 'dp_class:'
                dp_keys = class_keys

            return [self.data_parameters_dict[key_prefix+str(key)] for key in dp_keys]

        raise ValueError



    def set_enabled(self, enabled=True):
        self.disabled = not enabled



def get_basic_config_adam():
    config = DotDict({
    'data_param_mode': DataParamMode.ONLY_INSTANCE_PARAMS,
    'init_class_param': 0.01,
    'lr_class_param': 0.1,
    'init_inst_param': 1.0,
    'lr_inst_param': 0.1,
    'wd_inst_param': 0.0,
    'wd_class_param': 0.0,

    'skip_clamp_data_param': False,
    'clamp_sigma_min': np.log(1/20),
    'clamp_sigma_max': np.log(20),
    'optim_algorithm': DataParamOptim.ADAM,
    'optim_options': dict(
        # momentum=.9
        betas=(0.9, 0.999)
    )
    return config
})



def get_basic_config_sgd():
    config = DotDict({
    'data_param_mode': DataParamMode.ONLY_INSTANCE_PARAMS,
    'init_class_param': 0.01,
    'lr_class_param': 0.1,
    'init_inst_param': 1.0,
    'lr_inst_param': 0.1,
    'wd_inst_param': 0.0,
    'wd_class_param': 0.0,

    'skip_clamp_data_param': False,
    'clamp_sigma_min': np.log(1/20),
    'clamp_sigma_max': np.log(20),
    'optim_algorithm': DataParamOptim.SGD,
    'optim_options': dict(
        momentum=.9
    )
    return config
})