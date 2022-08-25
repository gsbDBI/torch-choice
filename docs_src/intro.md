# Introduction
This document briefly introduces the choice model we aim to solve.

In short, all models (the conditional logit model, the nested logit model, and the Bayesian embedding model) in the package aim to predict which item a user will purchase while facing the shelves in a supermarket.

Specifically, for each user $u$ and item $i$, models compute a value $U_{ui}$ predicting the utility user $u$ will get from purchasing item $i$. User $u$ is predicted to purchase the item $i$, generating the maximum utility.

However, the usage of our models is not limited to this supermarket context; researchers can adjust the definition of **user** and **item** to fit any choice modeling context. The [related project](./projects.md) page overviews some extensions of our models to other contexts.

## Notes on Encodings
Since we will be using PyTorch to train our model, we represent their identities with *consecutive* integer values instead of the raw human-readable names of items (e.g., Dell 24-inch LCD monitor). Similarly, you would need to encode user indices and session indices as well.
Raw item names can be encoded easily with [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) (The [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) works as well).

## Components of the Choice Modeling Problem
We aim to predict users' choices while facing multiple items available, e.g., which brand of milk the user will purchase in the supermarket.

We begin with essential components of the choice modeling problem. Walking through these components should help you understand what kind of data our models are working on.

### Purchasing Record
A **purchasing record** is a record describing *who* bought *what* at *when* and *where*.
Let $B$ denote the number of **purchasing records** in the dataset (i.e., number of rows/observation of the dataset). Each row $b \in \{1,2,\dots, B\}$ corresponds to a purchase record.

### *What*: Items and Categories
To begin with, suppose there are $I$ **items** indexed by $i \in \{1,2,\dots,I\}$.

The researcher can optionally partition the set items into $C$ **categories** indexed by $c \in \{1,2,\dots,C\}$. Let $I_c$ denote the collection of items in category $c$. It's easy to see that the union of all $I_c$ is the entire set of items $\{1, 2, \dots I\}$.
Suppose the researcher does not wish to model different categories differently. In that case, the researcher can put all items in one category: $I_1 = \{1, 2, \dots I\}$, so all items belong to the same category.

For each purchasing record $b \in \{1,2,\dots, B\}$, there is a corresponding $i_b \in \{1,2,\dots,I\}$ saying which item was chosen in this record.

#### Important Note on Encoding
Since we will be using PyTorch to train our model, we represent their identities with integer values instead of the raw human-readable names of items (e.g., Dell 24-inch LCD monitor).
Similarly, you would need to encode user indices and session indices as well.

Raw item names can be encoded easily with [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) (The [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) works as well).

Here is an example of encoding generic item names to integers using `sklearn.preprocessing.LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
raw_items = ['Macbook Laptop', 'Dell 24-inch Monitor', 'Orange', 'Apple (Fruit)']

encoded_items = enc.fit_transform(raw_items)
print(encoded_items)
# output: [2 1 3 0]

# for each 0 <= i <= 3, enc.classes_[i] reveals the raw name of item encoded to i.
print(enc.classes_)
# output: ['Apple (Fruit)' 'Dell 24-inch Monitor' 'Macbook Laptop' 'Orange']
# For example, the first entry of enc.classes_ is 'Apple (Fruit)', this means 'Apple (Fruit)' was encoded to 0 in this process.
# The last item in the `raw_item` list was 'Apple (Fruit)', and the last item in the `encoded_item` list was 0 as we expected.
```

### *Who*: Users
The counter-part of items in our setting are **user** indexed by $u \in \{1,2,\dots,U\}$ as well.

For each purchasing record $b \in \{1,2,\dots, B\}$, there is a corresponding $u_b \in \{1,2,\dots,I\}$ describing which user was making the decision.

### *When and Where*: Sessions
Our data structure encompasses *where and when* using a notion called **session** indexed by $s \in \{1,2,\dots, S\}$.

For example, we had the purchase record from five different stores for every day in 2021, then a session $s$ is defined as a pair of *(date, storeID)*, and there are $5 \times 365$ sessions in total.

In another example, suppose the data came from a single store for over a year. In this case, the notion of *where* is immaterial, and session $s$ is simply the date of purchase.

The notion of sessions can be more flexible than just date and location. For example, if we want to distinguish between online ordering and in-store purchasing, we can define the session as (date, storeID, IsOnlineOrdering).
The session variable serves as a tool for the researcher to split the dataset; the usefulness of the session will be more evident after introducing observables (features) later.

If the researcher does not wish to handle records from different sessions differently, the researcher can assign the same session ID to all dataset rows.

### Putting Everything Together
To summarize, each purchasing record $b \in \{1, 2, \dots, B\}$ in the dataset is characterized by a user-session-item tuple $(u_b, s_b, i_b)$. The totality of $B$ purchasing records consists of the dataset we are modeling.

When the same user buys multiple items in the same session, the dataset will have multiple purchasing records with the same $(u, s)$ corresponding to the same receipt.

### Item Availability
It is not necessarily that all items are available in every session; items can get out of stock in particular sessions.

To handle these cases, the researcher can *optionally* provide a boolean tensor $A \in \{\texttt{True}, \texttt{False}\}^{S\times I}$ to indicate which items are available for purchasing in each session. $A_{s, i} = \texttt{True}$ if and only if item $i$ was available in session $s$.

While predicting the purchase probabilities, the model sets the probability for these unavailable items to zero and normalizes probabilities among available items.
If the item availability is not provided, the model assumes all items are available in all sessions.

### Observables
Next, let's talk about observables (yes, it's the same as *feature* in machine learning literature, it's the $X$ variable).
The researcher can incorporate observables of, for example, users and items into the model.

Currently, the package support the following types of observables, where $K_{...}$ denote the number of observables.

1. `user_obs` $\in \mathbb{R}^{U\times K_{user}}$: user observables such as user age.
2. `item_obs` $\in \mathbb{R}^{I\times K_{item}}$: item observables such as item quality.
3. `session_obs` $\in \mathbb{R}^{S \times K_{session}}$: session observable such as whether the purchase was made on weekdays.
4. `price_obs` $\in \mathbb{R}^{S \times I \times K_{price}}$, price observables are values depending on **both** session and item such as the price of item.

Please note that we consider these four types as **definitions** of observable types. For example, whenever a variable is user-specific, then we call it an `user_obs`.
This package defines observables in the above way so that the package can easily track the variation of variables and handle these observable tensors correctly.

#### Note on the `Price` Observable
The `price_obs` term might look confusing at the first glance.
As mentioned above, price-observables are defined to be these observables depending on both session and item. If an observable depends on both session and item, it is called a *price observable* no matter if it is related to the actual price or not.

For example, in the context of online shopping, the shipping cost depends on both the item (i.e., the item purchased) and the session (i.e., when and where you purchase). In this case, the shipping cost observable is a price-observable but it's not an actual price.

Conversely, the actual price of an item might not change across sessions. For example, a 10-dollar Amazon gift card costs 10 dollars regardless of the session; in this case the *price* variable is in fact an *item observable* as it *only* depends on the item.

### A Toy Example
Suppose we have a dataset of purchase history from two stores (Store A and B) on two dates (Sep 16 and 17), both stores sell {apple, banana, orange} (`num_items=3`) and there are three people came to those stores between Sep 16 and 17.

| user_index | session_index       | item_index  |
| ---------- | ------------------- | ------ |
| Amy        | Sep-17-2021-Store-A | banana |
| Ben        | Sep-17-2021-Store-B | apple  |
| Ben        | Sep-16-2021-Store-A | orange |
| Charlie    | Sep-16-2021-Store-B | apple  |
| Charlie    | Sep-16-2021-Store-B | orange |

**NOTE**: For demonstration purposes, the example dataset has `user_index`, `session_index` and `item_index` as strings, they should be consecutive integers in actual production. One can easily convert them to integers using `sklearn.preprocessing.LabelEncoder`.

In the example above,
- `user_index=[0,1,1,2,2]` (with encoding `0=Amy, 1=Ben, 2=Charlie`),
- `session_index=[0,1,2,3,3]` (with encoding `0=Sep-17-2021-Store-A, 1=Sep-17-2021-Store-B, 2=Sep-16-2021-Store-A, 3=Sep-16-2021-Store-B`),
- `item_index=[0,1,2,1,2]` (with encoding `0=banana, 1=apple, 2=orange`).

Suppose we believe people's purchasing decision depends on the nutrition levels of these fruits; suppose apple has the highest nutrition level and banana has the lowest one, we can add

`item_obs=[[1.5], [12.0], [3.3]]` $\in \mathbb{R}^{3\times 1}$. The shape of this tensor is number-of-items by number-of-observable.


**NOTE**: If someone went to one store and bought multiple items (e.g., Charlie bought both apple and orange at Store B on Sep-16), we include them as separate rows in the dataset and model them independently.

## Models
The `torch-choice` library provides two models, the conditional logit model and the nested logit model, for modeling the dataset. Each model takes in $(u_b, s_b)$ altogether with observables and outputs a probability of purchasing each $\tilde{i} \in \{1, 2, \dots, I\}$, denoted as $\hat{p}_{u_b, s_b, \tilde{i}}$. In cases when not all items are available, the model sets the probability of unavailable items to zero and normalizes probabilities among available items. $\hat{p}_{u_b, s_b, \tilde{i}}$ is the predicted probability of purchasing item $\tilde{i}$ in session $s_b$ by user $u_b$ given all information we know. Model parameters are trained using gradient descent algorithm and the loss function is the negative log-likelihood of the model $-\sum_{b=1}^B \log(\hat{p}_{u_b, s_b, i_b})$.

The major difference among models lies in the way they compute predicted probabilities.