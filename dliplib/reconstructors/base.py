# -*- coding: utf-8 -*-
import os
import copy
import odl
import torch
import numpy as np

from math import ceil
from tqdm import tqdm
from warnings import warn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

from dival import LearnedReconstructor
from dival.measure import PSNR

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BaseLearnedReconstructor(LearnedReconstructor):
    HYPER_PARAMS = {
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'batch_size': {
            'default': 64,
            'retrain': True
        },
        'lr': {
            'default': 0.01,
            'retrain': True
        },
        'normalize_by_opnorm': {
            'default': False,
            'retrain': True
        }
    }

    def __init__(self, ray_trafo, epochs=None, batch_size=None, lr=None,
                 normalize_by_opnorm=None,
                 num_data_loader_workers=8, use_cuda=True, show_pbar=True,
                 fbp_impl='astra_cuda', hyper_params=None, log_dir=None,
                 log_num_validation_samples=0,
                 save_best_learned_params_path=None,
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform from which the FBP operator is constructed.
        scales : int, optional
            Number of scales in the U-Net (a hyper parameter).
        epochs : int, optional
            Number of epochs to train (a hyper parameter).
        batch_size : int, optional
            Batch size (a hyper parameter).
        lr : float, optional
            Base learning rate (a hyper parameter).
        normalize_by_opnorm : bool, optional
            Whether to normalize :attr:`ray_trafo` by its operator norm.
        num_data_loader_workers : int, optional
            Number of parallel workers to use for loading data.
        use_cuda : bool, optional
            Whether to use cuda for the U-Net.
        show_pbar : bool, optional
            Whether to show tqdm progress bars during the epochs.
        fbp_impl : str, optional
            The backend implementation passed to
            :class:`odl.tomo.RayTransform` in case no `ray_trafo` is specified.
            Then ``dataset.get_ray_trafo(impl=fbp_impl)`` is used to get the
            ray transform and FBP operator.
        log_dir : str, optional
            Tensorboard log directory (name of sub-directory in utils/logs).
            If `None`, no logs are written.
        log_num_valiation_samples : int, optional
            Number of validation images to store in tensorboard logs.
            This option only takes effect if ``log_dir is not None``.
        save_best_learned_params_path : str, optional
            Save best model weights during training under the specified path by
            calling :meth:`save_learned_params`.
        """
        super().__init__(reco_space=ray_trafo.domain,
                         observation_space=ray_trafo.range,
                         hyper_params=hyper_params, **kwargs)
        self.ray_trafo = ray_trafo
        self.num_data_loader_workers = num_data_loader_workers
        self.use_cuda = use_cuda
        self.show_pbar = show_pbar
        self.fbp_impl = fbp_impl
        self.log_dir = log_dir
        self.log_num_validation_samples = log_num_validation_samples
        self.save_best_learned_params_path = save_best_learned_params_path
        self.model = None

        if epochs is not None:
            self.epochs = epochs
            if kwargs.get('hyper_params', {}).get('epochs') is not None:
                warn("hyper parameter 'epochs' overridden by constructor "
                     "argument")

        if batch_size is not None:
            self.batch_size = batch_size
            if kwargs.get('hyper_params', {}).get('batch_size') is not None:
                warn("hyper parameter 'batch_size' overridden by constructor "
                     "argument")

        if lr is not None:
            self.lr = lr
            if kwargs.get('hyper_params', {}).get('lr') is not None:
                warn("hyper parameter 'lr' overridden by constructor argument")

        if normalize_by_opnorm is not None:
            self.normalize_by_opnorm = normalize_by_opnorm
            if (kwargs.get('hyper_params', {}).get('normalize_by_opnorm')
                    is not None):
                warn("hyper parameter 'normalize_by_opnorm' overridden by "
                     "constructor argument")

        if self.normalize_by_opnorm:
            self.opnorm = odl.power_method_opnorm(self.ray_trafo)
            self.ray_trafo = (1./self.opnorm) * self.ray_trafo

        self.device = (torch.device('cuda:0')
                       if self.use_cuda and torch.cuda.is_available() else
                       torch.device('cpu'))

    def eval(self, test_data):
        self.model.eval()

        running_psnr = 0.0
        with tqdm(test_data, desc='test ',
                  disable=not self.show_pbar) as pbar:
            for obs, gt in pbar:
                rec = self.reconstruct(obs)
                running_psnr += PSNR(rec, gt)

        return running_psnr / len(test_data)

    def train(self, dataset):
        torch.random.manual_seed(1)
        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            part='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape))

        dataset_validation = dataset.create_torch_dataset(
            part='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

        # reset model before training
        self.init_model()

        criterion = torch.nn.MSELoss()
        self.init_optimizer(dataset_train=dataset_train)

        # create PyTorch dataloaders
        data_loaders = {'train': DataLoader(
            dataset_train, batch_size=self.batch_size,
            num_workers=self.num_data_loader_workers, shuffle=True,
            pin_memory=True),
            'validation': DataLoader(
                dataset_validation, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers,
                shuffle=True, pin_memory=True)}

        dataset_sizes = {'train': len(dataset_train),
                         'validation': len(dataset_validation)}

        self.init_scheduler(dataset_train=dataset_train)
        if self.scheduler is not None:
            schedule_every_batch = isinstance(
                self.scheduler, (CyclicLR, OneCycleLR))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_psnr = 0

        if self.log_dir is not None:
            writer = SummaryWriter(
                log_dir=os.path.join(BASE_DIR, 'utils/logs', self.log_dir),
                max_queue=0)
            validation_samples = dataset.get_data_pairs(
                'validation', self.log_num_validation_samples)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1),
                          disable=not self.show_pbar) as pbar:
                    for inputs, labels in pbar:
                        if self.normalize_by_opnorm:
                            inputs = (1./self.opnorm) * inputs
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self.optimizer.step()
                                if (self.scheduler is not None and
                                        schedule_every_batch):
                                    self.scheduler.step()
                                # lrs.append(self.scheduler.get_lr())
                                # losses.append(loss.item())

                        for i in range(outputs.shape[0]):
                            labels_ = labels[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, labels_)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})
                        if self.log_dir is not None and phase == 'train':
                            step = (epoch * ceil(dataset_sizes['train']
                                                 / self.batch_size)
                                    + ceil(running_size / self.batch_size))
                            writer.add_scalar('loss/{}'.format(phase),
                                              torch.tensor(running_loss/running_size), step)
                            writer.add_scalar('psnr/{}'.format(phase),
                                              torch.tensor(running_psnr/running_size), step)

                    if self.scheduler is not None and not schedule_every_batch:
                        self.scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if self.log_dir is not None and phase == 'validation':
                        step = (epoch+1) * ceil(dataset_sizes['train']
                                                / self.batch_size)
                        writer.add_scalar('loss/{}'.format(phase),
                                          epoch_loss, step)
                        writer.add_scalar('psnr/{}'.format(phase),
                                          epoch_psnr, step)

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        if self.save_best_learned_params_path is not None:
                            self.save_learned_params(
                                self.save_best_learned_params_path)

            # import matplotlib.pyplot as plt
            # plt.plot(lrs)
            # plt.show()
            # plt.plot(lrs, losses)
            # plt.show()

                if (phase == 'validation' and self.log_dir is not None and
                        self.log_num_validation_samples > 0):
                    with torch.no_grad():
                        val_images = []
                        for (y, x) in validation_samples:
                            y = torch.from_numpy(
                                np.asarray(y))[None, None].to(self.device)
                            x = torch.from_numpy(
                                np.asarray(x))[None, None].to(self.device)
                            reco = self.model(y)
                            reco -= torch.min(reco)
                            reco /= torch.max(reco)
                            val_images += [reco, x]
                        writer.add_images(
                            'validation_samples', torch.cat(val_images),
                            (epoch + 1) * (ceil(dataset_sizes['train'] /
                                                self.batch_size)),
                            dataformats='NCWH')

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)

    def init_model(self):
        """
        Initialize :attr:`model`.
        Called in :meth:`train` at the beginning.
        """
        raise NotImplementedError

    def init_optimizer(self, dataset_train):
        """
        Initialize the optimizer.
        Called in :meth:`train`, after calling :meth:`init_model` and before
        calling :meth:`init_scheduler`.

        Parameters
        ----------
        dataset_train : :class:`torch.utils.data.Dataset`
            The training (torch) dataset constructed in :meth:`train'.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def init_scheduler(self, dataset_train):
        """
        Initialize the learning rate scheduler.
        Called in :meth:`train`, after calling :meth:`init_optimizer`.

        Parameters
        ----------
        dataset_train : :class:`torch.utils.data.Dataset`
            The training (torch) dataset constructed in :meth:`train'.
        """
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr,
            steps_per_epoch=ceil(len(dataset_train) / self.batch_size),
            epochs=self.epochs)

    def _reconstruct(self, observation):
        self.model.eval()
        with torch.set_grad_enabled(False):
            obs_tensor = torch.from_numpy(
                np.asarray(observation)[None, None])
            if self.normalize_by_opnorm:
                obs_tensor = obs_tensor / self.opnorm
            obs_tensor = obs_tensor.to(self.device)
            reco_tensor = self.model(obs_tensor)
            reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)

    def save_learned_params(self, path):
        path = path if path.endswith('.pt') else path + '.pt'
        torch.save(self.model.state_dict(), path)

    def load_learned_params(self, path, force_parallel=False):
        path = path if path.endswith('.pt') else path + '.pt'
        self.init_model()
        map_location = ('cuda:0' if self.use_cuda and torch.cuda.is_available()
                        else 'cpu')
        state_dict = torch.load(path, map_location=map_location)

        # backwards-compatibility with non-data_parallel weights
        data_parallel = list(state_dict.keys())[0].startswith('module.')
        if force_parallel and not data_parallel:
            state_dict = {('module.' + k): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
