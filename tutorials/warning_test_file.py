import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice import run
print(torch.__version__)

if __name__ == "__main__":
    # ==================================================================================================================
    # ==================================================================================================================
    if False:
        dataset = ChoiceDataset(
            num_items=7,
            item_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]),
            num_users=7,
            user_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]),
            num_sessions=7,
            session_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]),
        )
        print(dataset.num_items)
        print(dataset.num_users)
        print(dataset.num_sessions)

    # ==================================================================================================================
    # ==================================================================================================================
    if False:
        dataset = ChoiceDataset(
            item_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]),
            user_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]),
            session_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]),
        )
        print(dataset.num_items)
        print(dataset.num_users)
        print(dataset.num_sessions)

    # ==================================================================================================================
    # ==================================================================================================================
    if True:
        dataset = ChoiceDataset(
            item_index = torch.LongTensor([0, 1, 2, 3, -1, 5, 6]),
            user_index = torch.LongTensor([0, 1, 2, 3, 5, 6]),
            session_index = torch.LongTensor([0, 1, 2, 4, 5, 6]),
        )
        print(dataset.num_items)
        print(dataset.num_users)
        print(dataset.num_sessions)
