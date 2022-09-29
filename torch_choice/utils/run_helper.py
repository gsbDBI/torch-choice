"""
This is a template script for researchers to train the PyTorch-based model with minimal effort.
The researcher only needs to initialize the dataset and the model, this training template comes with default
hyper-parameters including batch size and learning rate. The researcher should experiment with different levels
of hyper-parameter if the default setting doesn't converge well.
"""
import pandas as pd
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch_choice.data import utils as data_utils
from torch_choice.utils.std import parameter_std
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel


def run(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.01, num_epochs=5000):
    """All in one script for the model training and result presentation."""
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

            if e % (num_epochs // 10) == 0:
                # record log-likelihood.
                ll -= model.negative_log_likelihood(batch, item_index).detach().item() # * len(batch)
                count += len(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ll /= count
        if e % (num_epochs // 10) == 0:
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
