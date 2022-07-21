# Introduction
This document provides a short introduction to the consumer choice model we aim to solve.

In short, all models in the package aim to predict which item an user will purchase while facing the shelves in a supermarket.
More specifically, for each user $u$ and item $i$, models compute a value $U_{ui}$ predicting the utility user $u$ will get from purchasing item $i$, then user $u$ is predicted to purchase the item $i$ generating the maximum utility.

However, the usage of our models is not limited to this supermarket context, researchers can adjust the definition of **user** and **item** to fit any choice modelling context. The [related project](./projects.md) page overviews some extensions of our models to other context.

## Components of the Consumer Choice Modelling Problem
We begin with essential component of the consumer choice modelling problem. Walking through these components should help you understand what kind of data our models are working on.

### Purchasing Record
Each row (record) of the dataset is called a **purchasing record**, which includes *who* bought *what* at *when* and *where*.
Let $B$ denote the number of **purchasing records** in the dataset (i.e., number of rows of the dataset). Each row $b \in \{1,2,\dots, B\}$ corresponds to a purchase record (i.e., *who* bought *what* at *where and when*).

### Items and Categories
To begin with, there are $I$ **items** indexed by $i \in \{1,2,\dots,I\}$ under our consideration.

Further, the researcher can optionally partition the set items into $C$ **categories** indexed by $c \in \{1,2,\dots,C\}$. Let $I_c$ denote the collection of items in category $c$, it is easy to verify that
$
\bigcup_{c \in \{1, 2, \dots, C\}} I_c = \{1, 2, \dots I\}
$
If the researcher does not wish to model different categories differently, the researcher can simply put all items in one single category: $I_1 = \{1, 2, \dots I\}$, so that all items belong to the same category.

**Note**: since we will be using PyTorch to train our model, we represent their identities with integer values instead of the raw human-readable names of items (e.g., Dell 24 inch LCD monitor).
Raw item names can be encoded easily with [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)

### Users
Each purchaing reocrd is naturally associated with an **user** indexed by $u \in \{1,2,\dots,U\}$ (*who*) as well.

### Sessions
Our data structure encompasses *where and when* using a notion called **session** indexed by $s \in \{1,2,\dots, S\}$.
For example, when the data came from a single store over the period of a year. In this case, the notion of *where* does not matter that much, and session $s$ is simply the date of purchase.

Another example is that we have the purchase record from different stores, the session $s$ can be defined as a pair of *(date, store)* instead.

If the researcher does not wish to handle records from different sessions differently, the researcher can assign the same session ID to all rows of the dataset.

To summarize, each purchasing record $b$ in the dataset is characterized by a user-session-item tuple $(u, s, i)$.

When there are multiple items bought by the same user in the same session, there will be multiple rows in the dataset with the same $(u, s)$ corresponding to the same receipt.

### Item Availability
It is not necessarily that all items are available in every session, items can get out-of-stock in particular sessions.

To handle these cases, the researcher can *optionally* provide a boolean tensor $\in \{\texttt{True}, \texttt{False}\}^{S\times I}$ to indicate which items are available for purchasing in each session.
While predicting the purchase probabilities, the model sets the probability for these unavailable items to zero and normalizes probabilities among available items.
If the item availability is not provided, the model assumes all items are available in all sessions.

## Available Models
