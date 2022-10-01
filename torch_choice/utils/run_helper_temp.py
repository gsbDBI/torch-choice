"""
This is a template script for researchers to train the PyTorch-based model with minimal effort.
The researcher only needs to initialize the dataset and the model, this training template comes with default
hyper-parameters including batch size and learning rate. The researcher should experiment with different levels
of hyper-parameter if the default setting doesn't converge well.
"""
import time
from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from bemb.model import BEMBFlex
from sklearn import metrics
from torch_choice.data import ChoiceDataset
from torch_choice.data import utils as data_utils
from torch_choice.data.utils import create_data_loader
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.utils.std import parameter_std


class LightningModelWrapper(pl.LightningModule):
    def __init__(self,
                 model: Union [ConditionalLogitModel, NestedLogitModel],
                 learning_rate: float=0.3):
        """The lightning model wrapper for conditional and nested logit model."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return str(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def _get_performance_dict(self, batch):
        item_index = batch['item'].item_index if isinstance(self.model, NestedLogitModel) else batch.item_index
        ll -= self.model.negative_log_likelihood(batch, item_index).detach().item()
        return {'log_likelihood': ll}

    def training_step(self, batch, batch_idx):
        item_index = batch['item'].item_index if isinstance(self.model, NestedLogitModel) else batch.item_index
        loss = self.model.loss(batch, item_index)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))
        for key, val in self._get_performance_dict(batch).items():
            self.log('test_' + key, val, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        for key, val in self._get_performance_dict(batch).items():
            self.log('val_' + key, val, prog_bar=True, batch_size=len(batch))

    def test_step(self, batch, batch_idx):
        for key, val in self._get_performance_dict(batch).items():
            self.log('test_' + key, val, prog_bar=True, batch_size=len(batch))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def fit_model(self, dataset_list: List[ChoiceDataset], batch_size: int=-1, num_epochs: int=10, num_workers: int=8, **kwargs) -> "LitBEMBFlex":
    #     """A standard pipeline of model training and evaluation.

    #     Args:
    #         dataset_list (List[ChoiceDataset]): train_dataset, validation_test, and test_dataset in a list of length 3.
    #         batch_size (int, optional): batch_size for training and evaluation. Defaults to -1, which indicates full-batch training.
    #         num_epochs (int, optional): number of epochs for training. Defaults to 10.
    #         **kwargs: additional keyword argument for the pytorch-lightning Trainer.

    #     Returns:
    #         LitBEMBFlex: the trained bemb model.
    #     """

    #     def section_print(input_text):
    #         """Helper function for printing"""
    #         print('=' * 20, input_text, '=' * 20)
    #     # present a summary of the model received.
    #     section_print('model received')
    #     print(self)

    #     # present a summary of datasets received.
    #     section_print('data set received')
    #     print('[Training dataset]', dataset_list[0])
    #     print('[Validation dataset]', dataset_list[1])
    #     print('[Testing dataset]', dataset_list[2])

    #     # create pytorch dataloader objects.
    #     train = create_data_loader(dataset_list[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     validation = create_data_loader(dataset_list[1], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #     # WARNING: the test step takes extensive memory cost since it computes likelihood for all items.
    #     # we run the test step with a much smaller batch_size.
    #     test = create_data_loader(dataset_list[2], batch_size=batch_size // 10, shuffle=False, num_workers=num_workers)

    #     section_print('train the model')
    #     trainer = pl.Trainer(gpus=1 if ('cuda' in str(self)) else 0,  # use GPU if the model is currently on the GPU.
    #                         max_epochs=num_epochs,
    #                         check_val_every_n_epoch=1,
    #                         log_every_n_steps=1,
    #                         **kwargs)
    #     start_time = time.time()
    #     trainer.fit(self, train_dataloaders=train, val_dataloaders=validation)
    #     print(f'time taken: {time.time() - start_time}')

    #     section_print('test performance')
    #     trainer.test(self, dataloaders=test)
    #     return self



def run_original(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.01, num_epochs=5000, report_frequency=None):
    """All in one script for the model training and result presentation."""
    if report_frequency is None:
        report_frequency = (num_epochs // 10)

    assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
        f'A model of type {type(model)} is not supported by this runner.'
    model = deepcopy(model)  # do not modify the model outside.
    trained_model = deepcopy(model)  # create another copy for returning.
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('=' * 20, 'received model', '=' * 20)
    print(model)
    print('=' * 20, 'received dataset', '=' * 20)
    print(dataset)
    print('=' * 20, 'training the model', '=' * 20)
    # fit the model.
    for e in range(1, num_epochs + 1):
        # track the log-likelihood to minimize.
        ll, count = 0.0, 0.0
        for batch in data_loader:
            item_index = batch['item'].item_index if isinstance(model, NestedLogitModel) else batch.item_index
            # the model.loss returns negative log-likelihood + regularization term.
            loss = model.loss(batch, item_index)

            if (e % report_frequency) == 0:
                # record log-likelihood.
                ll -= model.negative_log_likelihood(batch, item_index).detach().item() # * len(batch)
                count += len(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ll /= count
        if (e % report_frequency) == 0:
            print(f'Epoch {e}: Log-likelihood={ll}')

    if dataset_test is not None:
        test_ll = - model.negative_log_likelihood(dataset_test, dataset_test.item_index).detach().item()
        print('Test set log-likelihood: ', test_ll)

    # current methods of computing standard deviation will corrupt the model, load weights into another model for returning.
    state_dict = deepcopy(model.state_dict())
    trained_model.load_state_dict(state_dict)

    # get mean of estimation.
    mean_dict = dict()
    for k, v in model.named_parameters():
        mean_dict[k] = v.clone()

    # estimate the standard error of the model.
    if isinstance(model, ConditionalLogitModel):
        def nll_loss(model):
            y_pred = model(dataset)
            return F.cross_entropy(y_pred, dataset.item_index, reduction='sum')
    elif isinstance(model, NestedLogitModel):
        def nll_loss(model):
            d = dataset[torch.arange(len(dataset))]
            return model.negative_log_likelihood(d, d['item'].item_index)

    std_dict = parameter_std(model, nll_loss)

    print('=' * 20, 'model results', '=' * 20)
    report = list()
    for coef_name, std in std_dict.items():
        std = std.cpu().detach().numpy()
        mean = mean_dict[coef_name].cpu().detach().numpy()
        coef_name = coef_name.replace('coef_dict.', '').replace('.coef', '')
        for i in range(mean.size):
            report.append({'Coefficient': coef_name + f'_{i}',
                           'Estimation': float(mean[i]),
                           'Std. Err.': float(std[i])})
    report = pd.DataFrame(report).set_index('Coefficient')
    print(f'Training Epochs: {num_epochs}\n')
    print(f'Learning Rate: {learning_rate}\n')
    print(f'Batch Size: {batch_size if batch_size != -1 else len(dataset)} out of {len(dataset)} observations in total\n')
    print(f'Final Log-likelihood: {ll}\n')
    print('Coefficients:\n')
    print(report.to_markdown())
    return trained_model


def section_print(input_text):
    """Helper function for printing"""
    print('=' * 20, input_text, '=' * 20)


def run_on_lightning(lightning_model: LightningModelWrapper,
                     dataset_list: List[ChoiceDataset],
                     batch_size: int=-1,
                     num_epochs: int=10,
                     num_workers: int=8,
                     **kwargs) -> LightningModelWrapper:
    """A standard pipeline of model training and evaluation.

    Args:
        model (LitBEMBFlex): the initialized pytorch-lightning wrapper of bemb.
        dataset_list (List[ChoiceDataset]): train_dataset, validation_test, and test_dataset in a list of length 3.
        batch_size (int, optional): batch_size for training and evaluation. Defaults to -1, which indicates full-batch training.
        num_epochs (int, optional): number of epochs for training. Defaults to 10.
        **kwargs: additional keyword argument for the pytorch-lightning Trainer.

    Returns:
        LitBEMBFlex: the trained bemb model.
    """
    # present a summary of the model received.
    model = lightning_model.model
    model_clone = deepcopy(model)  # create another copy for returning.
    section_print('model received')
    print(model)

    # present a summary of datasets received.
    section_print('data set received')
    print('[Training dataset]', dataset_list[0])
    print('[Validation dataset]', dataset_list[1])
    print('[Testing dataset]', dataset_list[2])

    # create pytorch dataloader objects.
    train = create_data_loader(dataset_list[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if dataset_list[1] is not None:
        validation = create_data_loader(dataset_list[1], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        validation = None
    # WARNING: the test step takes extensive memory cost since it computes likelihood for all items.
    # we run the test step with a much smaller batch_size.
    if dataset_list[2] is not None:
        test = create_data_loader(dataset_list[2], batch_size=batch_size // 10, shuffle=False, num_workers=num_workers)
    else:
        test = None

    section_print('train the model')
    trainer = pl.Trainer(gpus=1 if ('cuda' in str(model.device)) else 0,  # use GPU if the model is currently on the GPU.
                         max_epochs=num_epochs,
                         check_val_every_n_epoch=1,
                         log_every_n_steps=1,
                         **kwargs)
    start_time = time.time()
    trainer.fit(model, train_dataloaders=train, val_dataloaders=validation)
    print(f'time taken: {time.time() - start_time}')

    # current methods of computing standard deviation will corrupt the model, load weights into another model for returning.
    state_dict = deepcopy(model.state_dict())
    model_clone.load_state_dict(state_dict)

    # get mean of estimation.
    mean_dict = dict()
    for k, v in model.named_parameters():
        mean_dict[k] = v.clone()

    # estimate the standard error of the model.
    dataset_for_std = dataset_list[0]
    if isinstance(model, ConditionalLogitModel):
        def nll_loss(model):
            y_pred = model(dataset_for_std)
            return F.cross_entropy(y_pred, dataset_for_std.item_index, reduction='sum')
    elif isinstance(model, NestedLogitModel):
        def nll_loss(model):
            d = dataset_for_std[torch.arange(len(dataset_for_std))]
            return model.negative_log_likelihood(d, d['item'].item_index)

    std_dict = parameter_std(model, nll_loss)

    # TODO: continue working here.
    print('=' * 20, 'model results', '=' * 20)
    report = list()
    for coef_name, std in std_dict.items():
        std = std.cpu().detach().numpy()
        mean = mean_dict[coef_name].cpu().detach().numpy()
        coef_name = coef_name.replace('coef_dict.', '').replace('.coef', '')
        for i in range(mean.size):
            report.append({'Coefficient': coef_name + f'_{i}',
                           'Estimation': float(mean[i]),
                           'Std. Err.': float(std[i])})
    report = pd.DataFrame(report).set_index('Coefficient')
    print(f'Training Epochs: {num_epochs}\n')
    print(f'Learning Rate: {learning_rate}\n')
    print(f'Batch Size: {batch_size if batch_size != -1 else len(dataset)} out of {len(dataset)} observations in total\n')
    print(f'Final Log-likelihood: {ll}\n')
    print('Coefficients:\n')
    print(report.to_markdown())

    if test is not None:
        section_print('test performance')
        trainer.test(model, dataloaders=test)
    else:
        print('No test dataset provided.')
    return model
