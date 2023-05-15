"""
This is a template script for researchers to train the PyTorch-based model with minimal effort.
The researcher only needs to initialize the dataset and the model, this training template comes with default
hyper-parameters including batch size and learning rate. The researcher should experiment with different levels
of hyper-parameter if the default setting doesn't converge well.
"""
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim
from torch_choice.data import utils as data_utils
from torch_choice.utils.std import parameter_std
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel


def run(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.01, num_epochs=5000, report_frequency=None, compute_std=True, return_final_training_log_likelihood=False, model_optimizer='Adam'):
    """All in one script for the model training and result presentation."""
    if report_frequency is None:
        report_frequency = (num_epochs // 10)

    assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
        f'A model of type {type(model)} is not supported by this runner.'
    model = deepcopy(model)  # do not modify the model outside.
    trained_model = deepcopy(model)  # create another copy for returning.
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = {'SGD': torch.optim.SGD,
                 'Adagrad': torch.optim.Adagrad,
                 'Adadelta': torch.optim.Adadelta,
                 'Adam': torch.optim.Adam}[model_optimizer](model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    print('=' * 20, 'received model', '=' * 20)
    print(model)
    print('=' * 20, 'received dataset', '=' * 20)
    print(dataset)
    print('=' * 20, 'training the model', '=' * 20)

    total_loss_history = list()
    tol = 0.001  # stop if the loss failed to improve tol proportion of average performance in the last k iterations.
    k = 5
    # fit the model.
    for e in range(1, num_epochs + 1):
        # track the log-likelihood to minimize.
        ll, count, total_loss = 0.0, 0.0, 0.0
        for batch in data_loader:
            item_index = batch['item'].item_index if isinstance(model, NestedLogitModel) else batch.item_index
            # the model.loss returns negative log-likelihood + regularization term.
            loss = model.loss(batch, item_index)
            total_loss -= loss

            with torch.no_grad():
                if (e % report_frequency) == 0:
                    # record log-likelihood.
                    ll -= model.negative_log_likelihood(batch, item_index).detach().item() # * len(batch)
                    count += len(batch)

                    pred = model.forward(batch).argmax(dim=1)
                    acc = (pred == item_index).float().mean().item()
                    print('Accuracy: ', acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        current_loss = float(total_loss.detach().item())
        # if e > k:
        if False:
            past_avg = np.mean(total_loss_history[-k:])
            improvement = (past_avg - current_loss) / past_avg
            if improvement < tol:
                print(f'Early stopped at {e} epochs.')
                break
        total_loss_history.append(current_loss)
        # ll /= count
        if (e % report_frequency) == 0:
            print(f'Epoch {e}: Log-likelihood={ll}')

    if dataset_test is not None:
        test_ll = - model.negative_log_likelihood(dataset_test, dataset_test.item_index).detach().item()
        print('Test set log-likelihood: ', test_ll)

    # final training log-likelihood.
    ll = - model.negative_log_likelihood(dataset if isinstance(model, ConditionalLogitModel) else dataset.datasets, dataset.item_index).detach().item() # * len(batch)

    if not compute_std:
        if return_final_training_log_likelihood:
            return model, ll
        else:
            return model
    else:
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
        print(f'Training Epochs: stopped at {e}, maximum allowed: {num_epochs}\n')
        print(f'Learning Rate: {learning_rate}\n')
        print(f'Batch Size: {batch_size if batch_size != -1 else len(dataset)} out of {len(dataset)} observations in total\n')
        print(f'Final Log-likelihood: {ll}\n')
        print('Coefficients:\n')
        print(report.to_markdown())
        if return_final_training_log_likelihood:
            return trained_model, ll
        else:
            return trained_model
