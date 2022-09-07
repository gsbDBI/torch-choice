# Introduction
# Introduction

## Logistic Regression and Choice Models

[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) models the probability that user $u$ chooses item $i$ in session $s$ by the logistic function
$$
P_{uis} = \frac{e^{\mu_{uis}}}{\Sigma_{j \in A_{us}}e^{\mu_{ujs}}}
$$
where, 

$$\mu_{uis} = \alpha + \beta X + \gamma W + \dots$$;

here $X$, $W$ are predictors (independent variables) for users and items respectively (these can be constant or can vary across session), and greek letters $\alpha$, $\beta$ and $\gamma$ are learned parameters
$A_{us}$ is the set of items available for user $u$ in session $s$.

When users are choosing over items, we can write utility $U_{uis}$ that user $u$ derives from item $i$ in session $s$, as
$U_{uis} = \mu_{uis} + \epsilon_{uis}$
where $\epsilon$ is an unobserved random error term.

If we assume iid extreme value type 1 errors for $\epsilon_{uis}$, this leads to the above logistic probabilities of user $u$ choosing item $i$ in session $s$, as shown by [McFadden](https://en.wikipedia.org/wiki/Choice_modelling), and as often studied in Econometrics.

## Package
We implement a fully flexible setup, where we allow 
1. coefficients ($\alpha$, $\beta$, $\gamma$, $\dots$) to be constant, user-specific (i.e., $\alpha=\alpha_u$), item-specific (i.e., $\alpha=\alpha_i$), session-specific (i.e., $\alpha=\alpha_t$), or (session, item)-specific (i.e., $\alpha=\alpha_{ti}$). For example, specifying $\alpha$ to be item-specific is equivalent to adding an item-level fixed effect.
2. Observables ($X$, $Y$, $\dots$) to be constant, user-specific, item-specific, session-specific, or (session, item) (such as price) and (session, user) (such as income) specific as well.
3. Specifying availability sets $A_{us}$

This flexibility in coefficients and features allows for more than 20 types of additive terms to $U_{uis}$, which enables modelling rich structures.

As such, this package can be used to learn such models for
1. Parameter Estimation, as in the Transportation Choice example below
2. Prediction, as in the MNIST handwritten digits classification example below

Examples with Utility Form:
1. Transportation Chioce (from the Mode Canada dataset) [(Detailed Tutorial)](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb)

$$
U_{uit} = \beta^0_i + \beta^{1\top} X^{price: (cost, freq, ovt)}_{it} + \beta^2_i X^{session:income}_t + \beta^3_i X_{it}^{price:ivt} + \epsilon_{uit}
$$

This is also described as a conditional logit model in Econometrics.


2. MNIST classification [(Upcoming Detailed Tutorial)]()

$$
U_{it} = \beta_i X^{session:pixelvalues} + \epsilon_{it}
$$

This is a classic problem used for exposition in Computer Science to motivate various Machine Learning models. There is no concept of a user in this setup. Our package allows for models of this nature and is fully usable for Machine Learning problems with added flexibility over [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

We highly recommend users to go through [tutorials](https://github.com/gsbDBI/torch-choice/blob/main/tutorials) we prepared to get a better understanding of what the package is offering. We present multiple examples, and for each case we specify the utility form. The [related project](./projects.md) page overviews some extensions of our models to other contexts.

## Notes on Encodings
Since we will be using PyTorch to train our model, we accept user and item identities with integer values from [0, 1, .. *num_users* - 1] and [0, 1, .. *num_items* - 1] instead of the raw human-readable names of items (e.g., Dell 24-inch LCD monitor) or any other encoding. The user is responsible to encode user indices, item indices and session indices, wherever appliable (some setups do not require session and/or user identifiers)
Raw item/user/session names can be encoded easily with [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) (The [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) works as well).

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

## Components of the Choice Modeling Problem
For the rest of this tutorial, we will consider retail supermarket choice as the concrete setting.

We aim to predict users' choices while choosing between multiple available items, e.g., which brand of milk the user will purchase in the supermarket.

We begin with essential components of the choice modeling problem. Walking through these components helps understand what kind of data our models are working on.

### Purchase Record
A **purchase record** is a record describing *who* bought *what* at *when* and *where*.
Let $B$ denote the number of **purchase records** in the dataset (i.e., number of rows/observation of the dataset). Each row $b \in \{1,2,\dots, B\}$ corresponds to a purchase record.

### *What*: Items and Categories
To begin with, suppose there are $I$ **items** indexed by $i \in \{1,2,\dots,I\}$.

The researcher can optionally partition the set items into $C$ **categories** indexed by $c \in \{1,2,\dots,C\}$. Let $I_c$ denote the collection of items in category $c$. It's easy to see that the union of all $I_c$ is the entire set of items $\{1, 2, \dots I\}$.
Suppose the researcher does not wish to model different categories differently. In that case, the researcher can put all items in one category: $I_1 = \{1, 2, \dots I\}$, so all items belong to the same category.

For each purchase record $b \in \{1,2,\dots, B\}$, there is a corresponding $i_b \in \{1,2,\dots,I\}$ saying which item was chosen in this record.

### *Who*: Users
The agent which makes choices in our setting is a **user** indexed by $u \in \{1,2,\dots,U\}$ as well.

For each purchase record $b \in \{1,2,\dots, B\}$, there is a corresponding $u_b \in \{1,2,\dots,I\}$ describing which user was making the decision.

### *When and Where*: Sessions
Our data structure encompasses *where and when* using a notion called **session** indexed by $s \in \{1,2,\dots, S\}$.

For example, we had the purchase record from five different stores for every day in 2021, then a session $s$ is defined as a pair of *(date, storeID)*, and there are $5 \times 365$ sessions in total.

In another example, suppose the data came from a single store for over a year. In this case, the notion of *where* is immaterial, and session $s$ is simply the date of purchase.

The notion of sessions can be more flexible than just date and location. For example, if we want to distinguish between online ordering and in-store purchasing, we can define the session as (date, storeID, IsOnlineOrdering).
The session variable serves as a tool for the researcher to split the dataset; the usefulness of the session will be more evident after introducing observables (features) later.

If the researcher does not wish to handle records from different sessions differently, the researcher can assign the same session ID to all dataset rows.

### Putting Everything Together
To summarize, each purchase record $b \in \{1, 2, \dots, B\}$ in the dataset is characterized by a user-session-item tuple $(u_b, s_b, i_b)$. The totality of $B$ purchase records consists of the dataset we are modeling.

When the same user buys multiple items in the same session, the dataset will have multiple purchase records with the same $(u, s)$ corresponding to the same receipt. In this case, the modeling assumption is that the user buys at most one item from each category available to choose from.

### Item Availability
It is not necessarily that all items are available in every session; items can get out of stock in particular sessions.

To handle these cases, the researcher can *optionally* provide a boolean tensor $A \in \{\texttt{True}, \texttt{False}\}^{S\times I}$ to indicate which items are available for purchase in each session. $A_{s, i} = \texttt{True}$ if and only if item $i$ was available in session $s$.

While predicting the purchase probabilities, the model sets the probability for these unavailable items to zero and normalizes probabilities among available items.
If the item availability is not provided, the model assumes all items are available in all sessions.

### Observables
Next, let's talk about observables. This is the same as a *feature* in machine learning literature, commonly denoted using $X$.
The researcher can incorporate observables of, for example, users and/or items into the model.

Currently, the package support the following types of observables, where $K_{...}$ denote the number of observables.

1. `user_obs` $\in \mathbb{R}^{U\times K_{user}}$: user observables such as user age.
2. `item_obs` $\in \mathbb{R}^{I\times K_{item}}$: item observables such as item quality.
3. `session_obs` $\in \mathbb{R}^{S \times K_{session}}$: session observable such as whether the purchase was made on weekdays.
4. `itemsession_obs` $\in \mathbb{R}^{S \times I \times K_{itemsession}}$, item-session observables are values depending on **both** session and item such as the price of item. These can also be called `price_obs`
4. `usersession_obs` $\in \mathbb{R}^{S \times U \times K_{usersession}}$, user-session observables are values depending on **both** session and user such as the income of the user.

Please note that we consider these four types as **definitions** of observable types. For example, whenever a variable is user-specific, then we call it an `user_obs`.
This package defines observables in the above way so that the package can easily track the variation of variables and handle these observable tensors correctly.

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

Suppose we believe people's purchase decision depends on the nutrition levels of these fruits; suppose apple has the highest nutrition level and banana has the lowest one, we can add

`item_obs=[[1.5], [12.0], [3.3]]` $\in \mathbb{R}^{3\times 1}$. The shape of this tensor is number-of-items by number-of-observable.


**NOTE**: If someone went to one store and bought multiple items (e.g., Charlie bought both apple and orange at Store B on Sep-16), we include them as separate rows in the dataset and model them independently.

## Models
The `torch-choice` library provides two models, the conditional logit model and the nested logit model, for modeling the dataset. Each model takes in $(u_b, s_b)$ altogether with observables and outputs a probability of purchasing each $\tilde{i} \in \{1, 2, \dots, I\}$, denoted as $\hat{p}_{u_b, s_b, \tilde{i}}$. In cases when not all items are available, the model sets the probability of unavailable items to zero and normalizes probabilities among available items. $\hat{p}_{u_b, s_b, \tilde{i}}$ is the predicted probability of purchasing item $\tilde{i}$ in session $s_b$ by user $u_b$ given all information we know. Model parameters are trained using gradient descent algorithm and the loss function is the negative log-likelihood of the model $-\sum_{b=1}^B \log(\hat{p}_{u_b, s_b, i_b})$.

The major difference among models lies in the way they compute predicted probabilities.
