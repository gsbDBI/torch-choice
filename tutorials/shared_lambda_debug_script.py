# ignore warnings for nicer outputs.
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice import run
print(torch.__version__)


if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    DEVICE = 'cuda'
else:
    print('Running tutorial on CPU')
    DEVICE = 'cpu'

if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/HC.csv', index_col=0)
    df = df.reset_index(drop=True)
    df.head()

    # what was actually chosen.
    item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
    item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
    num_items = df['idx.id2'].nunique()
    # cardinal encoder.
    encoder = dict(zip(item_names, range(num_items)))
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)

    # nest feature: no nest feature, all features are item-level.
    nest_dataset = ChoiceDataset(item_index=item_index.clone()).to(DEVICE)

    # item feature.
    item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
    price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)
    price_obs.shape
    item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs).to(DEVICE)
    dataset = JointDataset(nest=nest_dataset, item=item_dataset)

    nest_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                    1: ['gc', 'ec', 'er']}

    # encode items to integers.
    for k, v in nest_to_item.items():
        v = [encoder[item] for item in v]
        nest_to_item[k] = sorted(v)

    model = NestedLogitModel(nest_to_item=nest_to_item,
                            nest_formula='',
                            item_formula='(price_obs|constant)',
                            dataset=dataset,
                            shared_lambda=False)

    model = model.to(DEVICE)
    run(model, dataset, num_epochs=1000, model_optimizer="LBFGS")
