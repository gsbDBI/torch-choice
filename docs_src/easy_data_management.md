# Easy Data Wrapper Tutorial
The data construction covered in the Data Management tutorial might be too complicated for users without prior experience in PyTorch.
This tutorial offers a helper class to wrap the dataset, all the user needs to know is

(1) loading data-frames to Python, Pandas provides one-line solution to loading various types of data files including CSV, TSV, Stata, and Excel.

(2) basic usage of pandas. 

We aim to make this tutorial as self-contained as possible, so you don't need to be worried if you haven't went through the *Data Management tutorial*. But we invite you to go through that tutorial to obtain a more in-depth understanding of data management in this project.

Author: Tianyu Du

Date: May. 20, 2022

Update: Jul. 9, 2022


```python
__author__ = 'Tianyu Du'
```

Let's import a few necessary packages.


```python
import pandas as pd
import torch
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper
```

## References and Background for Stata Users
This tutorial aim to show how to manage choice datasets using the `torch-choice` package, we will follow the Stata documentation [here](https://www.stata.com/manuals/cm.pdf) to offer a seamless experience for the user to transfer prior knowledge in other packages to our package.

*From Stata Documentation*: Choice models (CM) are models for data with outcomes that are choices. The choices are selected by a decision maker, such as a person or a business (i.e., the **user**), from a set of possible alternatives (i.e., the **items**). For instance, we could model choices made by consumers who select a breakfast cereal from several different brands. Or we could model choices made by businesses who chose whether to buy TV, radio, Internet, or newspaper advertising.

Models for choice data come in two varieties—models for discrete choices and models for rank-ordered alternatives. When each individual selects a single alternative, say, he or she purchases one box of cereal, the data are discrete choice data. When each individual ranks the choices, say, he or she orders cereals from most favorite to least favorite, the data are rank-ordered data. Stata has commands for fitting both discrete choice models and rank-ordered models.

Our `torch-choice` package handles the **discrete choice** models in the Stata document above.

# Motivations
In the following parts, we demonstrate how to convert a long-format data (e.g., the one used in Stata) to the `ChoiceDataset` data format expected by our package.

But first, *Why do we want another `ChoiceDataset` object instead of just one long-format data-frame?*
In earlier versions of Stata, we can only have one single data-frame loaded in memory, this would introduce memory error especially when teh dataset is large. For example, you have a dataset of a million decisions recorded, each consists of four items, and each item has a persistent *built quality* that stay the same in all observations. The Stata format would make a million copy of these variables, which is very inefficient.

We would need to collect a couple of data-frames as the essential pieces to build our `ChoiceDataset`. Don't worry, as soon as you have the data-frames ready, the `EasyDataWrapper` helper class would take care of the rest.

We call a single statistical observation a **"purchase record"** and use this terminology throughout the tutorial. 


```python
df = pd.read_stata('https://www.stata-press.com/data/r17/carchoice.dta')
```

We load the artificial dataset from the Stata website. Here we borrow the description of dataset reported from the `describe` command in Stata. 

```
Contains data from https://www.stata-press.com/data/r17/carchoice.dta
 Observations:         3,160                  Car choice data
    Variables:             6                  30 Jul 2020 14:58
---------------------------------------------------------------------------------------------------------------------------------------------------
Variable      Storage   Display    Value
    name         type    format    label      Variable label
---------------------------------------------------------------------------------------------------------------------------------------------------
consumerid      int     %8.0g                 ID of individual consumer
car             byte    %9.0g      nation     Nationality of car
purchase        byte    %10.0g                Indicator of car purchased
gender          byte    %9.0g      gender     Gender: 0 = Female, 1 = Male
income          float   %9.0g                 Income (in $1,000)
dealers         byte    %9.0g                 No. of dealerships in community
---------------------------------------------------------------------------------------------------------------------------------------------------
Sorted by: consumerid  car

```

In this dataset, the first four rows with `consumerid == 1` corresponds to the first **purchasing record**, it means the consumer with ID 1 was making the decision among four types of cars (i.e., **items**) and chose `American` car (since the `purchase == 1` in that row of `American` car).

Even though there were four types of cars, not all of them were available all the time. For example, for the **purchase record** by consumer with ID 4, only American, Japanese, and European cars were available (note that there is no row in the dataset with `consumerid == 4` and `car == 'Korean'`, this indicates unavailability of a certain item.)


```python
df.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>consumerid</th>
      <th>car</th>
      <th>purchase</th>
      <th>gender</th>
      <th>income</th>
      <th>dealers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>American</td>
      <td>1</td>
      <td>Male</td>
      <td>46.699997</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Male</td>
      <td>46.699997</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>European</td>
      <td>0</td>
      <td>Male</td>
      <td>46.699997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Korean</td>
      <td>0</td>
      <td>Male</td>
      <td>46.699997</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>American</td>
      <td>1</td>
      <td>Male</td>
      <td>26.100000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Male</td>
      <td>26.100000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>European</td>
      <td>0</td>
      <td>Male</td>
      <td>26.100000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>Korean</td>
      <td>0</td>
      <td>Male</td>
      <td>26.100000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>American</td>
      <td>0</td>
      <td>Male</td>
      <td>32.700001</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>Japanese</td>
      <td>1</td>
      <td>Male</td>
      <td>32.700001</td>
      <td>6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>European</td>
      <td>0</td>
      <td>Male</td>
      <td>32.700001</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>American</td>
      <td>1</td>
      <td>Female</td>
      <td>49.199997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Female</td>
      <td>49.199997</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
      <td>European</td>
      <td>0</td>
      <td>Female</td>
      <td>49.199997</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>American</td>
      <td>0</td>
      <td>Male</td>
      <td>24.299999</td>
      <td>8</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Male</td>
      <td>24.299999</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>European</td>
      <td>1</td>
      <td>Male</td>
      <td>24.299999</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>American</td>
      <td>1</td>
      <td>Female</td>
      <td>39.000000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Female</td>
      <td>39.000000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>European</td>
      <td>0</td>
      <td>Female</td>
      <td>39.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>American</td>
      <td>0</td>
      <td>Male</td>
      <td>33.000000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Male</td>
      <td>33.000000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>European</td>
      <td>1</td>
      <td>Male</td>
      <td>33.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>Korean</td>
      <td>0</td>
      <td>Male</td>
      <td>33.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8</td>
      <td>American</td>
      <td>1</td>
      <td>Male</td>
      <td>20.299999</td>
      <td>6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8</td>
      <td>Japanese</td>
      <td>0</td>
      <td>Male</td>
      <td>20.299999</td>
      <td>5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8</td>
      <td>European</td>
      <td>0</td>
      <td>Male</td>
      <td>20.299999</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>American</td>
      <td>0</td>
      <td>Male</td>
      <td>38.000000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9</td>
      <td>Japanese</td>
      <td>1</td>
      <td>Male</td>
      <td>38.000000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>29</th>
      <td>9</td>
      <td>European</td>
      <td>0</td>
      <td>Male</td>
      <td>38.000000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# Components of the Consumer Choice Modelling Problem
We begin with essential component of the consumer choice modelling problem. Walking through these components should help you understand what kind of data our models are working on.

### Purchasing Record
Each row (record) of the dataset is called a **purchasing record**, which includes *who* bought *what* at *when* and *where*.
Let $B$ denote the number of **purchasing records** in the dataset (i.e., number of rows of the dataset). Each row $b \in \{1,2,\dots, B\}$ corresponds to a purchase record (i.e., *who* bought *what* at *where and when*).

### Items and Categories
To begin with, there are $I$ **items** indexed by $i \in \{1,2,\dots,I\}$ under our consideration.

Further, the researcher can optionally partition the set items into $C$ **categories** indexed by $c \in \{1,2,\dots,C\}$. Let $I_c$ denote the collection of items in category $c$, it is easy to verify that

$$
\bigcup_{c \in \{1, 2, \dots, C\}} I_c = \{1, 2, \dots I\}
$$

If the researcher does not wish to model different categories differently, the researcher can simply put all items in one single category: $I_1 = \{1, 2, \dots I\}$, so that all items belong to the same category.

**Note**: since we will be using PyTorch to train our model, we represent their identities with integer values instead of the raw human-readable names of items (e.g., Dell 24 inch LCD monitor).
Raw item names can be encoded easily with [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

### Users
Each purchaing reocrd is naturally associated with an **user** indexed by $u \in \{1,2,\dots,U\}$ (*who*) as well.

### Sessions
Our data structure encompasses *where and when* using a notion called **session** indexed by $s \in \{1,2,\dots, S\}$.
For example, when the data came from a single store over the period of a year. In this case, the notion of *where* does not matter that much, and session $s$ is simply the date of purchase.

Another example is that we have the purchase record from different stores, the session $s$ can be defined as a pair of *(date, store)* instead.

If the researcher does not wish to handle records from different sessions differently, the researcher can assign the same session ID to all rows of the dataset.

To summarize, each purchasing record $b$ in the dataset is characterized by a user-session-item tuple $(u, s, i)$.

When there are multiple items bought by the same user in the same session, there will be multiple rows in the dataset with the same $(u, s)$ corresponding to the same receipt.

# Format the Dataset a Little Bit
The wrapper we built requires several data frames, providing the correct information is all we need to do in this tutorial, the data wrapper will handle the construction of `ChoiceDataset` for you.

**Note**: The dataset in this tutorial is a bit over-simplified, we only have one purchase record for each user in each session, so the `consumerid` column identifies all of the user, the session, and the purchase record (because we have different dealers for the same type of car, we define each purchase record of it's session instead of assigning all purchase records to the same session).
That is, we have a single user makes a single choice in each single session.

The **main dataset** should contain the following columns:

1. `purchase_record_column`: a column identifies **purchase record** (also called **case** in Stata syntax). this tutorial, the `consumerid` column is the identifier. For example, the first 4 rows of the dataset (see above) has `consumerid == 1`, this means we should look at the first 4 rows together and they constitute the first purchase record.
2. `item_name_column`: a column identifies **names of items**, which is `car` in the dataset above. This column provides information above the availability as well. As mentioned above, there is no column with `car == Korean` in the fourth purchasing record (`consumerid == 4`), so we know that Korean car was not available that time.
3. `choice_column`: a column identifies the **choice** made by the consumer in each purchase record, which is the `purchase` column in our example. Exactly one row per purchase record (i.e., rows with the same values in `purchase_record_column`) should have 1, while the values are zeros for all other rows.
4. `user_index_column`: a *optional* column identifies the **user** making the choice, which is also `consumerid` in our case.
5. `session_index_column`: a *optional* column identifies the **session** of the choice, which is also `consumerid` in our case.

As you might have noticed, the `consumerid` column in the data-frame identifies multiple pieces of information: `purchase_record`, `user_index`, and `session_index`. This is not a mistake, you can use the same column in `df` to supply multiple pieces of information. 


```python
df.gender.value_counts(dropna=False)
```




    Male      2283
    Female     854
    NaN         23
    Name: gender, dtype: int64



The only modification required is to convert `gender` (with values of `Male`, `Female` or `NaN`) to integers because PyTorch does **not** handle strings. For simplicity, we will assume all `NaN` gender to be `Female` (you should **not** do this in a real application!) and re-define the gender variable as $\mathbb{I}\{\texttt{gender} == \texttt{Male}\}$.


```python
# we change gender to binary 0/1 because pytorch doesn't handle strings.
df['gender'] = (df['gender'] == 'Male').astype(int)
```

Now the `gender` column contains only binary integers.


```python
df.gender.value_counts(dropna=False)
```




    1    2283
    0     877
    Name: gender, dtype: int64



The data-frame looks like the following right now:


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>consumerid</th>
      <th>car</th>
      <th>purchase</th>
      <th>gender</th>
      <th>income</th>
      <th>dealers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>American</td>
      <td>1</td>
      <td>1</td>
      <td>46.699997</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Japanese</td>
      <td>0</td>
      <td>1</td>
      <td>46.699997</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>European</td>
      <td>0</td>
      <td>1</td>
      <td>46.699997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Korean</td>
      <td>0</td>
      <td>1</td>
      <td>46.699997</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>American</td>
      <td>1</td>
      <td>1</td>
      <td>26.100000</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



# Adding the Observables

The next step is to identify observables going into the model.

Specifically, we would want to add:
1. `gender` and `income` as user-specific observables
2. and `dealers` as (session, item)-specific observable. Such observables are called **price observables** in our setting, why? because price is the most typical (session, item)-specific observable.

## Method 1: Adding Observables by Extracting Columns of the Dataset
As you can see, `gender`, `income` and `dealers` are already encompassed in `df`, the first way to add observables is simply mentioning these columns while initializing the `EasyDatasetWrapper` object.

You can supply a list of names of columns to each of `{user, item, session, price}_observable_columns` keyword argument. For example, we use `user_observable_columns=['gender', 'income']` to inform the `EasyDatasetWrapper` that we wish to derive user-specific observables from the `gender` and `income` columns of `df`.

Also, we inform the `EasyDatasetWrapper` that we want to derive (session, item)-specific (i.e., price observable) by specifying `price_observable_columns=['dealers']`.

Since our package leverages GPU-acceleration, it is necessary to supply the **device** on which the dataset should reside.
The `EasyDatasetWrapper` also takes a `device` keyword, which can be either `'cpu'` or an appropriate CUDA device.


```python
if torch.cuda.is_available():
    device = 'cuda'  # use GPU if available
else:
    device = 'cpu'  # use CPU otherwise
```


```python
data_1 = EasyDatasetWrapper(main_data=df,
                            # TODO: better naming convention? Need to discuss.
                            # after discussion, we add it to the default value
                            # in the data wrapper class.
                            # these are just names.
                            purchase_record_column='consumerid',
                            choice_column='purchase',
                            item_name_column='car',
                            user_index_column='consumerid',
                            session_index_column='consumerid',
                            # it can be derived from columns of the dataframe or supplied as
                            user_observable_columns=['gender', 'income'],
                            price_observable_columns=['dealers'],
                            device=device)

```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.


The dataset has a `summary()` method, which can be used to print out the summary of the dataset.


```python
data_1.summary()
```

    * purchase record index range: [1 2 3] ... [883 884 885]
    * Space of 4 items:
                       0         1         2       3
    item name  American  European  Japanese  Korean
    * Number of purchase records/cases: 885.
    * Preview of main data frame:
          consumerid       car  purchase  gender     income  dealers
    0              1  American         1       1  46.699997        9
    1              1  Japanese         0       1  46.699997       11
    2              1  European         0       1  46.699997        5
    3              1    Korean         0       1  46.699997        1
    4              2  American         1       1  26.100000       10
    ...          ...       ...       ...     ...        ...      ...
    3155         884  Japanese         1       1  20.900000       10
    3156         884  European         0       1  20.900000        4
    3157         885  American         1       1  30.600000       10
    3158         885  Japanese         0       1  30.600000        5
    3159         885  European         0       1  30.600000        4
    
    [3160 rows x 6 columns]
    * Preview of ChoiceDataset:
    ChoiceDataset(label=[], item_index=[885], user_index=[885], session_index=[885], item_availability=[885, 4], user_gender=[885, 1], user_income=[885, 1], price_dealers=[885, 4, 1], device=cuda:0)


You can access the `ChoiceDataset` object constructed by calling the `data.choice_dataset` object. 


```python
data_1.choice_dataset
```




    ChoiceDataset(label=[], item_index=[885], user_index=[885], session_index=[885], item_availability=[885, 4], user_gender=[885, 1], user_income=[885, 1], price_dealers=[885, 4, 1], device=cuda:0)



## Method 2: Adding Observables as Data Frames
We can also construct data frames and use data frames to supply different observables. This is useful when you have a large dataset, for example, if there are many purchase records for the same user (to be concrete, say $U$ users and $N$ purchase records for each user, resulting $U \times N$ total purchase records). Using a single data-frame requires a lot of memory: you need to store $U \times N$ entires of user genders in total. However, user genders should be persistent across all purchasing records, if we use a separate data-frame mapping user index to gender of the user, we only need to store $U$ entries (i.e., one for each user) of gender information.

Similarly, the long-format data requires storing each piece of item-specific information for number of purchase records times, which leads to inefficient usage of disk/memory space.

### How Do Observable Data-frame Look Like?

Our package natively support the following four types of observables:

1. **User Observables**: user-specific observables (e.g., gender and income) should (1) have length equal to the number of unique users in the dataset (885 here); (2) contains a column named as `user_index_column` (`user_index_column` is a variable, the actual column name should be the **value** of variable `user_index_column`! E.g., here the user observable data-frame should have a column named `'consumerid'`); (3) the user observable can have any number of other columns beside the `user_index_column` column, each of them corresponding to a user-specific observable. For example, a data-frame containing $X$ user-specific observables has shape `(num_users, X + 1)`.

2. **Item Observables** item-specific observables (not shown in this tutorial) should be (1) have length equal to the number of unique items in the dataset (4 here); (2) contain a column named as `item_index_column` (`item_index_column` is a variable, the actual column name should be the **value** of variable `item_index_column`! E.g., here the item observable data-frame should have a column named `'car'`); (3) the item observable can have any number of other columns beside the `item_index_column` column, each of them corresponding to a item-specific observable.

3. **Session Observable** session-specific observables (not shown in this tutorial) should be (1) have length equal to the number of unique sessions in the dataset; (2) contain a column named as `session_index_column` (`session_index_column` is a variable, the actual column name should be the **value** of variable `session_index_column`! E.g., here the session observable data-frame should have a column named `'consumerid'`); (3) the session observable can have any number of other columns beside the `session_index_column` column, each of them corresponding to a session-specific observable.

4. **Price Observables** (session, item)-specific observables (e.g., dealers) should be (1) contains a column named as `session_index_column` (e.g., `consumerid` in our example) **and** a column named as `item_name_column` (e.g., `car` in our example), (2) the price observable can have any number of other columns beside the `session_index_column` and `item_name_column` columns, each of them corresponding to a (session, item)-specific observable. For example, a data-frame containing $X$ (session, item)-specific observables has shape `(num_sessions, num_items, X + 2)`.

We encourage the reader to review the *Data Management Tutorial* for more details on types of observables.

### Suggested Procedure of Storing and Loading Data
1. Suppose `SESSION_INDEX` column in `df_main` is the index of the session, `ALTERNATIVES` column is the index of the car.
2. For user-specific observables, you should have a CSV on disk with columns {`consumerid`, `var_1`, `var_2`, ...}.
3. You load the user-specific dataset as `user_obs = pd.read_csv(..., index='consumerid')`.

Let's first construct the data frame for user genders first.


```python
gender = df.groupby('consumerid')['gender'].first().reset_index()
```

The user-observable data-frame contains a column of user IDs (the `consumerid` column), this column should have exactly the same name as the column containing user indices. Otherwise, the wrapper won't know which column corresponds to user IDs and which column corresponds to variables.


```python
gender.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>consumerid</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Then, let's build the data-frame for user-specific income variables.


```python
income = df.groupby('consumerid')['income'].first().reset_index()
```


```python
income.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>consumerid</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>46.699997</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>26.100000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>32.700001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>49.199997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>24.299999</td>
    </tr>
  </tbody>
</table>
</div>



Please note that we can have multiple observables contained in the same data-frame as well.


```python
gender_and_income = df.groupby('consumerid')[['gender', 'income']].first().reset_index()
gender_and_income
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>consumerid</th>
      <th>gender</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>46.699997</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>26.100000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>32.700001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>49.199997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>24.299999</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>880</th>
      <td>881</td>
      <td>1</td>
      <td>45.700001</td>
    </tr>
    <tr>
      <th>881</th>
      <td>882</td>
      <td>1</td>
      <td>69.800003</td>
    </tr>
    <tr>
      <th>882</th>
      <td>883</td>
      <td>0</td>
      <td>45.599998</td>
    </tr>
    <tr>
      <th>883</th>
      <td>884</td>
      <td>1</td>
      <td>20.900000</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>1</td>
      <td>30.600000</td>
    </tr>
  </tbody>
</table>
<p>885 rows × 3 columns</p>
</div>



The price observable data-frame contains two columns identifying session (i.e., the `consumerid` column) and item (i.e., the `car` column). The session index column should have exactly the same name as the session index column in `df` and the column indexing columns should have exactly the same name as the item-name-column in `df`.


```python
dealers = df[['consumerid', 'car', 'dealers']]
```


```python
dealers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>consumerid</th>
      <th>car</th>
      <th>dealers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>American</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Japanese</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>European</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Korean</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>American</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



## Build Datasets using `EasyDatasetWrapper` with Observables as Data-Frames
We can observables as data-frames using `{user, item, session, price}_observable_data` keyword arguments. 


```python
data_2 = EasyDatasetWrapper(main_data=df,
                            purchase_record_column='consumerid',
                            choice_column='purchase',
                            item_name_column='car',
                            user_index_column='consumerid',
                            session_index_column='consumerid',
                            # above are the same as before, but we update the following.
                            user_observable_data={'gender': gender, 'income': income},
                            price_observable_data={'dealers': dealers},
                            device=device)
```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.



```python
# Use summary to see what's inside the data wrapper.
data_2.summary()
```

    * purchase record index range: [1 2 3] ... [883 884 885]
    * Space of 4 items:
                       0         1         2       3
    item name  American  European  Japanese  Korean
    * Number of purchase records/cases: 885.
    * Preview of main data frame:
          consumerid       car  purchase  gender     income  dealers
    0              1  American         1       1  46.699997        9
    1              1  Japanese         0       1  46.699997       11
    2              1  European         0       1  46.699997        5
    3              1    Korean         0       1  46.699997        1
    4              2  American         1       1  26.100000       10
    ...          ...       ...       ...     ...        ...      ...
    3155         884  Japanese         1       1  20.900000       10
    3156         884  European         0       1  20.900000        4
    3157         885  American         1       1  30.600000       10
    3158         885  Japanese         0       1  30.600000        5
    3159         885  European         0       1  30.600000        4
    
    [3160 rows x 6 columns]
    * Preview of ChoiceDataset:
    ChoiceDataset(label=[], item_index=[885], user_index=[885], session_index=[885], item_availability=[885, 4], user_gender=[885, 1], user_income=[885, 1], price_dealers=[885, 4, 1], device=cuda:0)


Alternatively, we can supply user income and gender as a single dataframe, instead of `user_gender` and `user_income` tensors, now the constructed `ChoiceDataset` contains a single `user_gender_and_income` tensor with shape (885, 2) encompassing both income and gender of users.


```python
data_3 = EasyDatasetWrapper(main_data=df,
                            purchase_record_column='consumerid',
                            choice_column='purchase',
                            item_name_column='car',
                            user_index_column='consumerid',
                            session_index_column='consumerid',
                            # above are the same as before, but we update the following.
                            user_observable_data={'gender_and_income': gender_and_income},
                            price_observable_data={'dealers': dealers},
                            device=device)
```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.



```python
data_3.summary()
```

    * purchase record index range: [1 2 3] ... [883 884 885]
    * Space of 4 items:
                       0         1         2       3
    item name  American  European  Japanese  Korean
    * Number of purchase records/cases: 885.
    * Preview of main data frame:
          consumerid       car  purchase  gender     income  dealers
    0              1  American         1       1  46.699997        9
    1              1  Japanese         0       1  46.699997       11
    2              1  European         0       1  46.699997        5
    3              1    Korean         0       1  46.699997        1
    4              2  American         1       1  26.100000       10
    ...          ...       ...       ...     ...        ...      ...
    3155         884  Japanese         1       1  20.900000       10
    3156         884  European         0       1  20.900000        4
    3157         885  American         1       1  30.600000       10
    3158         885  Japanese         0       1  30.600000        5
    3159         885  European         0       1  30.600000        4
    
    [3160 rows x 6 columns]
    * Preview of ChoiceDataset:
    ChoiceDataset(label=[], item_index=[885], user_index=[885], session_index=[885], item_availability=[885, 4], user_gender_and_income=[885, 2], price_dealers=[885, 4, 1], device=cuda:0)


## Method 3: Mixing Method 1 and Method 2
The `EasyDataWrapper` also support supplying observables as a mixture of above methods. The following example supplies `gender` user observable as a data-frame but `income` and `dealers` as column names. 


```python
data_4 = EasyDatasetWrapper(main_data=df,
                            purchase_record_column='consumerid',
                            choice_column='purchase',
                            item_name_column='car',
                            user_index_column='consumerid',
                            session_index_column='consumerid',
                            # above are the same as before, but we update the following.
                            user_observable_data={'gender': gender},
                            user_observable_columns=['income'],
                            price_observable_columns=['dealers'],
                            device=device)
```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.



```python
data_4.summary()
```

    * purchase record index range: [1 2 3] ... [883 884 885]
    * Space of 4 items:
                       0         1         2       3
    item name  American  European  Japanese  Korean
    * Number of purchase records/cases: 885.
    * Preview of main data frame:
          consumerid       car  purchase  gender     income  dealers
    0              1  American         1       1  46.699997        9
    1              1  Japanese         0       1  46.699997       11
    2              1  European         0       1  46.699997        5
    3              1    Korean         0       1  46.699997        1
    4              2  American         1       1  26.100000       10
    ...          ...       ...       ...     ...        ...      ...
    3155         884  Japanese         1       1  20.900000       10
    3156         884  European         0       1  20.900000        4
    3157         885  American         1       1  30.600000       10
    3158         885  Japanese         0       1  30.600000        5
    3159         885  European         0       1  30.600000        4
    
    [3160 rows x 6 columns]
    * Preview of ChoiceDataset:
    ChoiceDataset(label=[], item_index=[885], user_index=[885], session_index=[885], item_availability=[885, 4], user_gender=[885, 1], user_income=[885, 1], price_dealers=[885, 4, 1], device=cuda:0)


# Sanity Checks
Lastly, let's check choice datasets constructed via different methods are actually the same.

The `==` method of choice datasets will compare the non-NAN entries of all tensors in datasets. 


```python
print(data_1.choice_dataset == data_2.choice_dataset)
print(data_1.choice_dataset == data_4.choice_dataset)
```

    True
    True


For `data_3`, we have `income` and `gender` combined:


```python
data_3.choice_dataset.user_gender_and_income == torch.cat([data_1.choice_dataset.user_gender, data_1.choice_dataset.user_income], dim=1)
```




    tensor([[True, True],
            [True, True],
            [True, True],
            ...,
            [True, True],
            [True, True],
            [True, True]], device='cuda:0')



Now let's compare what's inside the data structure and our raw data.


```python
bought_raw = df[df['purchase'] == 1]['car'].values
bought_data = list()
encoder = {0: 'American', 1: 'European', 2: 'Japanese', 3: 'Korean'}
for b in data_1.choice_dataset.item_index:
    bought_data.append(encoder[float(b)])
```


```python
all(bought_raw == bought_data)
```




    True



Then, let's compare the income and gender variable contained in the dataset. 


```python
X = df.groupby('consumerid')['income'].first().values
Y = data_1.choice_dataset.user_income.cpu().numpy().squeeze()
all(X == Y)
```




    True




```python
X = df.groupby('consumerid')['gender'].first().values
```


```python
Y = data_1.choice_dataset.user_gender.cpu().numpy().squeeze()
all(X == Y)
```




    True



Lastly, let's compare the `price_dealer` variable. Since there are NAN-values in it for unavailable cars, we can't not use `all(X == Y)` to compare them. We will first fill NANs values with `-1` and then compare resulted data-frames.


```python
# rearrange columns to align it with the internal encoding scheme of the data wrapper.
X = df.pivot('consumerid', 'car', 'dealers')[['American', 'European', 'Japanese', 'Korean']]
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>car</th>
      <th>American</th>
      <th>European</th>
      <th>Japanese</th>
      <th>Korean</th>
    </tr>
    <tr>
      <th>consumerid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>881</th>
      <td>8.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>882</th>
      <td>8.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>9.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>884</th>
      <td>12.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>885</th>
      <td>10.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>885 rows × 4 columns</p>
</div>




```python
Y = data_1.choice_dataset.price_dealers.squeeze(dim=-1)
```


```python
Y
```




    tensor([[ 9.,  5., 11.,  1.],
            [10.,  2.,  7.,  1.],
            [ 8.,  2.,  6., nan],
            ...,
            [ 9.,  5.,  8.,  1.],
            [12.,  4., 10., nan],
            [10.,  4.,  5., nan]], device='cuda:0')




```python
print(X.fillna(-1).values == torch.nan_to_num(Y, -1).cpu().numpy())
```

    [[ True  True  True  True]
     [ True  True  True  True]
     [ True  True  True  True]
     ...
     [ True  True  True  True]
     [ True  True  True  True]
     [ True  True  True  True]]


This concludes our tutorial on building the dataset, if you wish more in-depth understanding of the data structure, please refer to the *Data Management Tutorial*.


