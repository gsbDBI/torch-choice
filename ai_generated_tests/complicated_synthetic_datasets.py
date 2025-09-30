import torch

from torch_choice.data.choice_dataset import ChoiceDataset

try:
    from torch_choice.data.joint_dataset import JointDataset
except ImportError:
    JointDataset = None


def generate_basic_dataset(num_cases=100, num_alternatives=5, num_features=10):
    """Generate a basic synthetic ChoiceDataset.

    Args:
        num_cases (int): number of cases (observations).
        num_alternatives (int): number of alternatives per case.
        num_features (int): number of features for each alternative.

    Returns:
        ChoiceDataset: A synthetic dataset.
    """
    # item_index: For each case, randomly choose one alternative
    item_index = torch.randint(0, num_alternatives, (num_cases,))

    # itemsession_cost_freq_ovt: shape (num_cases, num_alternatives, num_features)
    cost_freq_ovt = torch.rand(num_cases, num_alternatives, num_features)

    # session_income: shape (num_cases, 1), simulate incomes between 30000 and 100000
    session_income = torch.randint(30000, 100000, (num_cases, 1)).float()

    # itemsession_ivt: shape (num_cases, num_alternatives), some random values
    itemsession_ivt = torch.rand(num_cases, num_alternatives)

    dataset = ChoiceDataset(item_index=item_index,
                             itemsession_cost_freq_ovt=cost_freq_ovt,
                             session_income=session_income,
                             itemsession_ivt=itemsession_ivt)
    return dataset


def generate_large_dataset(num_cases=1000, num_alternatives=10, num_features=15):
    """Generate a large synthetic ChoiceDataset with more data points and features.

    Args:
        num_cases (int): number of cases.
        num_alternatives (int): number of alternatives.
        num_features (int): number of features per alternative.

    Returns:
        ChoiceDataset: A synthetic dataset.
    """
    item_index = torch.randint(0, num_alternatives, (num_cases,))
    cost_freq_ovt = torch.rand(num_cases, num_alternatives, num_features)
    session_income = torch.randint(30000, 150000, (num_cases, 1)).float()
    itemsession_ivt = torch.rand(num_cases, num_alternatives)

    dataset = ChoiceDataset(item_index=item_index,
                             itemsession_cost_freq_ovt=cost_freq_ovt,
                             session_income=session_income,
                             itemsession_ivt=itemsession_ivt)
    return dataset


def generate_joint_dataset(num_cases=200, num_alternatives=5, num_features=8):
    """Generate a JointDataset by combining two synthetic ChoiceDatasets.

    Args:
        num_cases (int): number of cases.
        num_alternatives (int): number of alternatives per case.
        num_features (int): number of features per alternative.

    Returns:
        JointDataset or None: A JointDataset if JointDataset is available, otherwise None.
    """
    ds1 = generate_basic_dataset(num_cases, num_alternatives, num_features)
    ds2 = generate_basic_dataset(num_cases, num_alternatives, num_features)

    if JointDataset is not None:
        joint_ds = JointDataset(nest=ds1, item=ds2)
        return joint_ds
    else:
        return None


def generate_complex_dataset(num_cases=500, num_alternatives=7, num_features=20, missing_rate=0.1):
    """Generate a complex synthetic ChoiceDataset with additional structure, outliers, and missing data.

    Args:
        num_cases (int): number of cases.
        num_alternatives (int): number of alternatives per case.
        num_features (int): number of features per alternative.
        missing_rate (float): fraction of missing values to introduce.

    Returns:
        ChoiceDataset: A synthetic dataset with complex characteristics.
    """
    # item_index: For each case, randomly choose one alternative
    item_index = torch.randint(0, num_alternatives, (num_cases,))

    # Generate base features from normal distribution, then add an alternative-dependent offset
    base = torch.randn(num_cases, num_alternatives, num_features)
    offsets = torch.linspace(0, 5, steps=num_alternatives).view(1, num_alternatives, 1).expand(num_cases, num_alternatives, num_features)
    cost_freq_ovt = base + offsets

    # Introduce missing values in cost_freq_ovt
    mask = torch.rand(num_cases, num_alternatives, num_features) < missing_rate
    cost_freq_ovt[mask] = float('nan')

    # session_income: generate incomes from a normal distribution centered at 60000 with std 15000, and add outliers
    incomes = torch.normal(mean=60000, std=15000, size=(num_cases, 1))
    outlier_mask = torch.rand(num_cases, 1) < 0.05  # 5% outliers
    incomes[outlier_mask] = incomes[outlier_mask] * 3
    session_income = incomes

    # itemsession_ivt: generate using sigmoid of normal distribution, simulating probabilities
    itemsession_ivt = torch.sigmoid(torch.randn(num_cases, num_alternatives))
    # Introduce missing values in itemsession_ivt
    mask_ivt = torch.rand(num_cases, num_alternatives) < missing_rate
    itemsession_ivt[mask_ivt] = float('nan')

    dataset = ChoiceDataset(item_index=item_index,
                             itemsession_cost_freq_ovt=cost_freq_ovt,
                             session_income=session_income,
                             itemsession_ivt=itemsession_ivt)
    return dataset


if __name__ == '__main__':
    # Generate and print details for the basic dataset
    ds_basic = generate_basic_dataset(num_cases=100, num_alternatives=5, num_features=10)
    print('Basic Synthetic Dataset generated:')
    print('  item_index shape:', ds_basic.item_index.shape)
    print('  itemsession_cost_freq_ovt shape:', ds_basic.itemsession_cost_freq_ovt.shape)
    print('  session_income shape:', ds_basic.session_income.shape)
    print('  itemsession_ivt shape:', ds_basic.itemsession_ivt.shape)

    # Generate and print details for the large dataset
    ds_large = generate_large_dataset()
    print('\nLarge Synthetic Dataset generated:')
    print('  item_index shape:', ds_large.item_index.shape)
    print('  itemsession_cost_freq_ovt shape:', ds_large.itemsession_cost_freq_ovt.shape)
    print('  session_income shape:', ds_large.session_income.shape)
    print('  itemsession_ivt shape:', ds_large.itemsession_ivt.shape)

    # Generate and print details for the joint dataset
    ds_joint = generate_joint_dataset()
    if ds_joint is not None:
        print('\nJoint Synthetic Dataset generated:')
        print('  nest.item_index shape:', ds_joint.nest.item_index.shape)
        print('  item.item_index shape:', ds_joint.item.item_index.shape)
    else:
        print('\nJointDataset is not available in the current installation.')

    # Generate and print details for the complex dataset
    ds_complex = generate_complex_dataset()
    print('\nComplex Synthetic Dataset generated:')
    print('  item_index shape:', ds_complex.item_index.shape)
    print('  itemsession_cost_freq_ovt shape:', ds_complex.itemsession_cost_freq_ovt.shape)
    print('  session_income shape:', ds_complex.session_income.shape)
    print('  itemsession_ivt shape:', ds_complex.itemsession_ivt.shape)