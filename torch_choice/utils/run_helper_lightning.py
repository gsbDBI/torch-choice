"""
This is a template script for researchers to train the PyTorch-based model with minimal effort.
The researcher only needs to initialize the dataset and the model, this training template comes with default
hyper-parameters including batch size and learning rate. The researcher should experiment with different levels
of hyper-parameter if the default setting doesn't converge well.

This is a modified version of the original run_helper.py script, which is modified to work with PyTorch Lightning.
"""
import time
from copy import deepcopy
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch_choice.data import ChoiceDataset
from torch_choice.data.utils import create_data_loader
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.utils.std import parameter_std


class LightningModelWrapper(pl.LightningModule):
    def __init__(self,
                 model: Union [ConditionalLogitModel, NestedLogitModel],
                 learning_rate: float,
                 optimizer: str):
        """
        The pytorch-lightning model wrapper for conditional and nested logit model.
        Ideally, end users don't need to interact with this class. This wrapper will be called by the run() function.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_class_string = optimizer

    def __str__(self) -> str:
        return str(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def _get_performance_dict(self, batch):
        item_index = batch['item'].item_index if isinstance(self.model, NestedLogitModel) else batch.item_index
        ll = - self.model.negative_log_likelihood(batch, item_index).detach().item()
        return {'log_likelihood': ll}

    def training_step(self, batch, batch_idx):
        item_index = batch['item'].item_index if isinstance(self.model, NestedLogitModel) else batch.item_index
        loss = self.model.loss(batch, item_index)
        self.log('train_loss', loss, prog_bar=False, batch_size=len(batch))
        # skip computing log-likelihood for training steps to speed up training.
        # for key, val in self._get_performance_dict(batch).items():
            # self.log('test_' + key, val, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        for key, val in self._get_performance_dict(batch).items():
            self.log('val_' + key, val, prog_bar=False, batch_size=len(batch))

    def test_step(self, batch, batch_idx):
        for key, val in self._get_performance_dict(batch).items():
            self.log('test_' + key, val, prog_bar=False, batch_size=len(batch))

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_class_string)(self.parameters(), lr=self.learning_rate)

# def run_original(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.01, num_epochs=5000, report_frequency=None):
#     """All in one script for the model training and result presentation."""
#     if report_frequency is None:
#         report_frequency = (num_epochs // 10)

#     assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
#         f'A model of type {type(model)} is not supported by this runner.'
#     model = deepcopy(model)  # do not modify the model outside.
#     trained_model = deepcopy(model)  # create another copy for returning.
#     print('=' * 20, 'received model', '=' * 20)
#     print(model)
#     print('=' * 20, 'received dataset', '=' * 20)
#     print(dataset)
#     print('=' * 20, 'training the model', '=' * 20)


def section_print(input_text):
    """Helper function for printing"""
    print('=' * 20, input_text, '=' * 20)


def run(model: Union [ConditionalLogitModel, NestedLogitModel],
        dataset_train: ChoiceDataset,
        dataset_val: Optional[ChoiceDataset]=None,
        dataset_test: Optional[ChoiceDataset]=None,
        optimizer: str='adam',
        batch_size: int=-1,
        learning_rate: float=0.01,
        num_epochs: int=10,
        num_workers: int=0,
        device: Optional[str]=None,
        **kwargs) -> Union[ConditionalLogitModel, NestedLogitModel]:
    """_summary_

    Args:
        model (Union[ConditionalLogitModel, NestedLogitModel]): the constructed model.
        dataset_train (ChoiceDataset): the dataset for training.
        dataset_val (ChoiceDataset): an optional dataset for validation.
        dataset_test (ChoiceDataset): an optional dataset for testing.
        batch_size (int, optional): batch size for model training. Defaults to -1.
        learning_rate (float, optional): learning rate for model training. Defaults to 0.01.
        num_epochs (int, optional): number of epochs for the training. Defaults to 10.
        num_workers (int, optional): number of parallel workers for data loading. Defaults to 0.
        device (Optional[str], optional): the device that trains the model, if None is specified, the function will
            use the current device of the provided model. Defaults to None.
        **kwargs: other keyword arguments for the pytorch lightning trainer, this is for users with experience in
            pytorch lightning and wish to customize the training process.

    Returns:
        Union[ConditionalLogitModel, NestedLogitModel]: the trained model.
    """
    # ==================================================================================================================
    # Setup the lightning wrapper.
    # ==================================================================================================================
    lightning_model = LightningModelWrapper(model, learning_rate=learning_rate, optimizer=optimizer)
    if device is None:
        # infer from the model device.
        device = model.device
    # the cloned model will be used for standard error calculation later.
    model_clone = deepcopy(model)
    section_print('model received')
    print(model)

    # ==================================================================================================================
    # Prepare the data.
    # ==================================================================================================================
    # present a summary of datasets received.
    section_print('data set received')
    print('[Train dataset]', dataset_train)
    print('[Validation dataset]', dataset_val)
    print('[Test dataset]', dataset_test)

    # create pytorch dataloader objects.
    train_dataloader = create_data_loader(dataset_train.to(device), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if dataset_val is not None:
        val_dataloader = create_data_loader(dataset_val.to(device), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_dataloader = None

    if dataset_test is not None:
        test_dataloader = create_data_loader(dataset_test.to(device), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_dataloader = None

    # ==================================================================================================================
    # Training the model.
    # ==================================================================================================================
    # if the validation dataset is provided, do early stopping.
    callbacks = [EarlyStopping(monitor="val_ll", mode="max", patience=10, min_delta=0.001)] if val_dataloader is not None else []

    trainer = pl.Trainer(gpus=1 if ('cuda' in str(model.device)) else 0,  # use GPU if the model is currently on the GPU.
                         max_epochs=num_epochs,
                         check_val_every_n_epoch=num_epochs // 100,
                         log_every_n_steps=num_epochs // 100,
                         callbacks=callbacks,
                         **kwargs)
    start_time = time.time()
    trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f'Time taken for training: {time.time() - start_time}')
    if test_dataloader is not None:
        trainer.test(lightning_model, test_dataloaders=test_dataloader)
    else:
        print('Skip testing, no test dataset is provided.')

    # ====== get the standard error of the model ====== #
    # current methods of computing standard deviation will corrupt the model, load weights into another model for returning.
    state_dict = deepcopy(lightning_model.model.state_dict())
    model_clone.load_state_dict(state_dict)

    # get mean of estimation.
    mean_dict = dict()
    for k, v in lightning_model.model.named_parameters():
        mean_dict[k] = v.clone()

    # estimate the standard error of the model.
    dataset_for_std = dataset_train.clone()

    if isinstance(model, ConditionalLogitModel):
        def nll_loss(model):
            y_pred = model(dataset_for_std)
            return F.cross_entropy(y_pred, dataset_for_std.item_index, reduction='sum')
    elif isinstance(model, NestedLogitModel):
        def nll_loss(model):
            d = dataset_for_std[torch.arange(len(dataset_for_std))]
            return model.negative_log_likelihood(d, d['item'].item_index)
    std_dict = parameter_std(model_clone, nll_loss)

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
    # print(f'Training Epochs: {num_epochs}\n')
    # print(f'Learning Rate: {learning_rate}\n')
    # print(f'Batch Size: {batch_size if batch_size != -1 else len(dataset_list[0])} out of {len(dataset_list[0])} observations in total in test set\n')

    lightning_model.model.to(device)
    train_ll = - lightning_model.model.negative_log_likelihood(dataset_train, dataset_train.item_index).detach().item()

    if dataset_val is not None:
        val_ll = - lightning_model.model.negative_log_likelihood(dataset_val, dataset_val.item_index).detach().item()
    else:
        val_ll = 'N/A'

    if dataset_test is not None:
        test_ll = - lightning_model.model.negative_log_likelihood(dataset_test, dataset_test.item_index).detach().item()
    else:
        test_ll = 'N/A'
    print(f'Final Log-likelihood: [Training] {train_ll}, [Validation] {val_ll}, [Test] {test_ll}\n')
    print('Coefficients:\n')
    print(report.to_markdown())
    return model
