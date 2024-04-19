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
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import norm

from torch_choice.data import ChoiceDataset
from torch_choice.data.utils import create_data_loader
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.utils.std import parameter_std


class LightningModelWrapper(pl.LightningModule):
    def __init__(self,
                 model: Union [ConditionalLogitModel, NestedLogitModel],
                 learning_rate: float,
                 model_optimizer: str):
        """
        The pytorch-lightning model wrapper for conditional and nested logit model.
        Ideally, end users don't need to interact with this class. This wrapper will be called by the run() function.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_class_string = model_optimizer

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

def section_print(input_text):
    """Helper function for printing"""
    print('=' * 20, input_text, '=' * 20)


def run(model: Union [ConditionalLogitModel, NestedLogitModel],
        dataset_train: ChoiceDataset,
        dataset_val: Optional[ChoiceDataset]=None,
        dataset_test: Optional[ChoiceDataset]=None,
        model_optimizer: str='Adam',
        batch_size: int=-1,
        learning_rate: float=0.01,
        num_epochs: int=10,
        num_workers: int=0,
        device: Optional[str]=None,
        report_std: bool=True,
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
        report_std (bool, optional): whether to report standard error for the estimated parameters. Defaults to True.
        **kwargs: other keyword arguments for the pytorch lightning trainer, this is for users with experience in
            pytorch lightning and wish to customize the training process.

    Returns:
        Union[ConditionalLogitModel, NestedLogitModel]: the trained model.
    """
    # ==================================================================================================================
    # Setup the lightning wrapper.
    # ==================================================================================================================
    lightning_model = LightningModelWrapper(model,
                                            learning_rate=learning_rate,
                                            model_optimizer=model_optimizer)
    if device is None:
        # infer from the model device.
        device = str(model.device)
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

    trainer = pl.Trainer(accelerator="cuda" if "cuda" in device else device,  # note: "cuda:0" is not a accelerator name.
                         devices="auto",
                         max_epochs=num_epochs,
                         check_val_every_n_epoch=max(num_epochs // 100, 1),
                         log_every_n_steps=max(num_epochs // 100, 1),
                         callbacks=callbacks,
                         **kwargs)
    start_time = time.time()
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
    print(f'Time taken for training: {time.time() - start_time}')
    if test_dataloader is not None:
        trainer.test(lightning_model, test_dataloader)
    else:
        print('Skip testing, no test dataset is provided.')

    if not report_std:
        return model

    # ==================================================================================================================
    # Construct standard error, z-value, and p-value of coefficients.
    # ==================================================================================================================
    # current methods of computing standard deviation will corrupt the model, load weights into another model for returning.
    state_dict = deepcopy(lightning_model.model.state_dict())
    if "lambdas" in state_dict:
        # if the model is using a specific lambda for each nest, it will create an additional `lambdas` tensor in the forward process.
        # the forward method on `model_clone` has never been called, so it does not have the `lambdas` tensor.
        # we need to drop the `lambdas` tensor from the state_dict to avoid the error while loading the state dict.
        # The lambdas tensor is simply a copy of the lambda_weight in this case.
        assert torch.all(state_dict["lambdas"] == state_dict["lambda_weight"]), \
            f"lambdas and lambda_weight should be the same, maximum difference: {torch.max(torch.abs(state_dict['lambdas'] - state_dict['lambda_weight']))}: {state_dict['lambdas']=:}, {state_dict['lambda_weight']=:}"
        state_dict.pop("lambdas")
    model_clone.load_state_dict(state_dict, strict=True)

    # get mean of estimation.
    mean_dict = dict()
    for k, v in lightning_model.model.named_parameters():
        mean_dict[k] = v.clone()

    # estimate the standard error of the model.
    dataset_for_std = dataset_train.clone()

    if isinstance(model, ConditionalLogitModel):
        def nll_loss(model):
            y_pred = model(dataset_for_std)
            item_index = dataset_for_std.item_index.clone()
            if model.model_outside_option:
                assert y_pred.shape == (len(dataset_for_std), model.num_items+1)
                # y_pred has shape (len(dataset_for_std.choice_set), model.num_items+1) since the last column is the probability of the outside option.
                # F.cross_entropy is not smart enough to handle the -1 outside option in y.
                # Even though y_pred[:, -1] nad y_pred[:, model.num_items] are the same, F.cross_entropy does not know.
                # We need to fix it manually.
                # manually modify the index for the outside option.
                item_index[item_index == -1] = model.num_items

            return F.cross_entropy(y_pred, item_index, reduction='sum')
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
                           'Estimation': float(mean.reshape(-1,)[i]),
                           'Std. Err.': float(std.reshape(-1,)[i])
                           })
    report = pd.DataFrame(report).set_index('Coefficient')

    # Compute z-value
    report['z-value'] = report['Estimation'] / report['Std. Err.']

    # Compute p-value (two tails).
    report['Pr(>|z|)'] = (1 - norm.cdf(abs(report['z-value']))) * 2

    # Compute significance stars
    report['Significance'] = ''
    report.loc[report['Pr(>|z|)'] < 0.001, 'Significance'] = '***'
    report.loc[(report['Pr(>|z|)'] >= 0.001) & (report['Pr(>|z|)'] < 0.01), 'Significance'] = '**'
    report.loc[(report['Pr(>|z|)'] >= 0.01) & (report['Pr(>|z|)'] < 0.05), 'Significance'] = '*'

    # Compute log-likelihood on the final model on all splits of datasets.
    lightning_model.model.to(device)
    is_nested = isinstance(lightning_model.model, NestedLogitModel)

    train_ll = - lightning_model.model.negative_log_likelihood(dataset_train.datasets if is_nested else dataset_train,
                                                               dataset_train.item_index).detach().item()

    if dataset_val is not None:
        val_ll = - lightning_model.model.negative_log_likelihood(dataset_val.datasets if is_nested else dataset_val,
                                                                 dataset_val.item_index).detach().item()
    else:
        val_ll = 'N/A'

    if dataset_test is not None:
        test_ll = - lightning_model.model.negative_log_likelihood(dataset_test.datasets if is_nested else dataset_test,
                                                                  dataset_test.item_index).detach().item()
    else:
        test_ll = 'N/A'

    print(f'Log-likelihood: [Training] {train_ll}, [Validation] {val_ll}, [Test] {test_ll}\n')
    print(report.to_markdown())
    print("Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    return model