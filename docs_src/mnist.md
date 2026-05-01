# Using `torch-choice` as Benchmark Model in Machine Learning Setting: MNIST Dataset

This tutorial demonstrate the usage of `torch-choice`'s logit model as a benchmark multinominal model in machine learning setting. We will use the MNIST dataset as an example.


```python
from time import time
import os
import torch, torchvision
from torch_choice.data import ChoiceDataset
from torch_choice.model import ConditionalLogitModel
from torch_choice import run

```


```python
print("PyTorch Version: ",torch.__version__)
print("GPU Available: ",torch.cuda.is_available())
USE_GPU = bool(int(os.environ.get("MNIST_TUTORIAL_USE_GPU", "0"))) and torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'
print('Using device:', DEVICE)

```

    PyTorch Version:  1.13.0+cu117
    GPU Available:  True
    Using device: cuda



```python
# download MNIST dataset.
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
```


```python
print(f'{mnist_train.data.shape=:}')
print(f'{mnist_train.targets.shape=:}')
print(f'{mnist_test.data.shape=:}')
print(f'{mnist_test.targets.shape=:}')
```

    mnist_train.data.shape=torch.Size([60000, 28, 28])
    mnist_train.targets.shape=torch.Size([60000])
    mnist_test.data.shape=torch.Size([10000, 28, 28])
    mnist_test.targets.shape=torch.Size([10000])



```python
X = torch.cat([mnist_train.data.reshape(60000, -1), mnist_test.data.reshape(10000, -1)], dim=0)
y = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
print(f'{X.shape=:}')
print(f'{y.shape=:}')
N_train = 10000
N_test = 2000
N = N_train + N_test
X = X[:N]
y = y[:N]
print(f'Using subset with {N_train} training and {N_test} test samples')

```

    X.shape=torch.Size([70000, 784])
    y.shape=torch.Size([70000])


We assume each image in the MNIST dataset is corresponding to a session, and we are predicting the "item" chosen in this session. The chosen "item" is the digit in the image.


```python
dataset = ChoiceDataset(session_index=torch.arange(N), item_index=y, session_image=X)
train_index = torch.arange(N_train)
test_index = torch.arange(N_train, N_train + N_test)
# we don't have a validation set.
dataset_train = dataset[train_index].to(DEVICE)
dataset_test = dataset[test_index].to(DEVICE)

```

For each digit $i \in \{0, 1, \dots 9\}$, for each image indexed $n \in \{1, 2, \dots, 70000\}$, let $X^{(n)} \in \mathbb{R}^{768}$ denote image $n$'s feature vector. The potential of image $n$ to represent digit $i$ is captured by:
$$
U_{i}^{(n)} = \alpha_i + (X^{(n)})^T \beta_i
$$

The predicted probability of image $n$ being digit $i$ is given by the soft-max transformation of above potentials:

$$
P_{i}^{(n)} = \frac{\exp(U_{i}^{(n)})}{\sum_{j=0}^9 \exp(U_{j}^{(n)})}
$$


```python
model = ConditionalLogitModel(
    formula='(session_image|item-full) + (1|item-full)',
    dataset=dataset_train,
    num_items=10)
model = model.to(DEVICE)
```


```python
start_time = time()
run(model, dataset_train=dataset_train, dataset_test=dataset_test, num_epochs=50, learning_rate=0.003, model_optimizer="LBFGS", batch_size=-1, device=DEVICE, report_std=False)
print('Time taken:', time() - start_time)

```

    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /home/tianyudu/anaconda3/envs/dev/lib/python3.9/site-packages/pytorch_lightning/trainer/configuration_validator.py:108: PossibleUserWarning: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
      rank_zero_warn(
    You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name  | Type                  | Params
    ------------------------------------------------
    0 | model | ConditionalLogitModel | 7.9 K 
    ------------------------------------------------
    7.9 K     Trainable params
    0         Non-trainable params
    7.9 K     Total params
    0.031     Total estimated model params size (MB)


    ==================== model received ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (session_image[item-full]): Coefficient(variation=item-full, num_items=10, num_users=None, num_params=784, 7840 trainable parameters in total, device=cuda:0).
        (intercept[item-full]): Coefficient(variation=item-full, num_items=10, num_users=None, num_params=1, 10 trainable parameters in total, device=cuda:0).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[session_image[item-full]] with 784 parameters, with item-full level variation.
    X[intercept[item-full]] with 1 parameters, with item-full level variation.
    device=cuda:0
    ==================== data set received ====================
    [Train dataset] ChoiceDataset(label=[], item_index=[60000], user_index=[], session_index=[60000], item_availability=[], session_image=[70000, 784], device=cuda:0)
    [Validation dataset] None
    [Test dataset] ChoiceDataset(label=[], item_index=[10000], user_index=[], session_index=[10000], item_availability=[], session_image=[70000, 784], device=cuda:0)


    /home/tianyudu/anaconda3/envs/dev/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 16 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    /home/tianyudu/anaconda3/envs/dev/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py:1609: PossibleUserWarning: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=3). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(



    Training: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=300` reached.
    You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]


    Time taken for training: 114.026784658432


    /home/tianyudu/anaconda3/envs/dev/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, test_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 16 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(



    Testing: 0it [00:00, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">    test_log_likelihood    </span>│<span style="color: #800080; text-decoration-color: #800080">    -3652.419677734375     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    Time taken: 114.07550883293152



```python
model = model.to(DEVICE)
```


```python
train_acc = torch.mean((model.forward(dataset_train).argmax(dim=1) == dataset_train.item_index).float())
test_acc = torch.mean((model.forward(dataset_test).argmax(dim=1) == dataset_test.item_index).float())
print(f"Training Accuracy: {train_acc*100:.2f}%.")
print(f"Test Accuracy: {test_acc*100:.2f}%.")
```

    Training Accuracy: 94.35%.
    Test Accuracy: 91.91%.

