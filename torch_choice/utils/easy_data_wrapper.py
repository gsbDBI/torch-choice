"""
This is a helper class for creating ChoiceDataset class, we only assume very basic python knowledge to use this utility.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List
from sklearn.preprocessing import LabelEncoder
from torch_choice.data import ChoiceDataset

__author__ = 'Tianyu Du'


class EasyDatasetWrapper():
    """An easy-to-use interface for creating ChoiceDataset object, please refer to the doc-string of the `__init__` method
    for more details. You feed it with a couple of pandas data-frames and necessary information, this EasyDatasetWrapper
    would create the ChoiceDataset object for you.
    """

    def __init__(self,
                 main_data: pd.DataFrame,
                 purchase_record_column: str,
                 item_name_column: str,
                 choice_column: str,
                 user_index_column: Optional[str] = None,
                 session_index_column: Optional[str] = None,
                 # Option 1: feed in data-frames of observables.
                 user_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 item_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 useritem_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 session_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 price_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 itemsession_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 useritemsession_observable_data: Optional[Dict[str, pd.DataFrame]] = None,
                 # Option 2: derive observables from columns of main_data.
                 user_observable_columns: Optional[List[str]] = None,
                 item_observable_columns: Optional[List[str]] = None,
                 useritem_observable_columns: Optional[List[str]] = None,
                 session_observable_columns: Optional[List[str]] = None,
                 price_observable_columns: Optional[List[str]] = None,
                 itemsession_observable_columns: Optional[List[str]] = None,
                 useritemsession_observable_columns: Optional[List[str]] = None,
                 num_items: Optional[int] = None,
                 num_users: Optional[int] = None,
                 num_sessions: Optional[int] = None,
                 # Misc.
                 device: str = 'cpu'):
        """The initialization method of EasyDatasetWrapper.

        Args:
            main_data (pd.DataFrame): the main dataset holding all purchase records in a "long-format", each row of the
                dataset is an item in a purchase record.
                The main_data data-frame should contains the following columns:
            purchase_record_column (str): the column in main_data identifies the index of purchasing records.
            item_name_column (str): the column in main_data identifies the name of items.
            choice_column (str): the column in the main_data identifies the bought item, for all rows with the same value
                of `purchase_record_column`, there should be exactly one row with `choice_column` equal to 1, all
                other rows should be 0.
            user_index_column (Optional[str], optional): an optional column indicating the user in each purchasing records,
                values should be constant across all rows with the same `purchase_record_column`. Defaults to None.
            session_index_column (Optional[str], optional): an optional column indicating the session in each purchasing records,
                values should be constant across all rows with the same `purchase_record_column`. Defaults to None.

            The keys of all of *_observable_data are the name of the observable data.

            user_observable_data (Optional[Dict[str, pd.DataFrame]], optional): a dictionary with keys as the name of each
                observable. The values should be a pandas data-frame contains a column named by the value of `user_index_column`
                and consisting of values from `main_data[user_index_column]`. Defaults to dict().

            item_observable_data (Optional[Dict[str, pd.DataFrame]], optional): a dictionary with keys as the name of each
                observable. The values should be a pandas data-frame contains a column named by the value of `item_name_column`
                and consisting of values from `main_data[item_name_column]`. Defaults to dict().

            session_observable_data (Optional[Dict[str, pd.DataFrame]], optional): a dictionary with keys as the name of each
                observable. The values should be a pandas data-frame contains a column named by the value of `session_index_column`
                and consisting of values from `main_data[session_index_column]`. Defaults to dict().

            price_observable_data (Optional[Dict[str, pd.DataFrame]], optional): a dictionary with keys as the name of each
                observable. The values should be a pandas data-frame contains (1) a column named by the value of `session_index_column`
                and consists of values from `main_data[session_index_column]` and (2) a column named by the value of `item_name_column`
                and consisting of values from `main_data[item_name_column]`. Defaults to dict().

            itemsession_observable_data (Optional[Dict[str, pd.DataFrame]], optional) is an alias for the
                price_observable_data argument for backward compatibility. All items of itemsession_observable_data
                will be added to price_observable_data. Defaults to dict().

            Another method to include observables is via *_observable_columns keywords, which takes column name(s) of the main_data
                data-frame. The data wrapper will derive observable data from the main_data data-frame.
                For example, with `user_observable_columns = ['feature_A', 'feature_B']`, this wrapper will create two user-specific
                observable tensors derived from main_data['feature_A'] and main_data['feature_B'].

            The itemsession_observable_column is an alias for the `price_observable_column` argument for backward compatibility,
                all elements of `itemsession_observable_columns` will be appended to `price_observable_column`.

            num_items (Optional[int], optional): the number of items in the dataset to pass to the ChoiceDataset. Defaults to None.

            num_users (Optional[int], optional): the number of users in the dataset to pass to the ChoiceDataset. Defaults to None.

            num_sessions (Optional[int], optional): the number of sessions in the dataset to pass to the ChoiceDataset. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if (useritem_observable_data is not None) or (useritemsession_observable_data is not None) or (useritem_observable_columns is not None) or (useritemsession_observable_columns is not None):
            raise NotImplementedError("The user-item and user-item-session observables are not yet supported in easy data wrapper. Please construct ChoiceDataset objects directly with tensors using the ChoiceDataset class to include these observables.")

        # read in data.
        self.main_data = main_data
        self.purchase_record_column = purchase_record_column
        # in alphabetical order of purchase record indices.
        # this is kept internally.
        self.purchase_record_index = main_data[purchase_record_column].unique()
        self.item_name_column = item_name_column
        self.choice_column = choice_column
        self.user_index_column = user_index_column
        self.session_index_column = session_index_column
        self.device = device

        # encode item name, user index, session index.
        self.encode()

        # adding alias itemsession_observable_data to price_observable_data.
        if itemsession_observable_data is not None:
            if price_observable_data is None:
                price_observable_data = itemsession_observable_data
            else:
                price_observable_data.extend(itemsession_observable_data)

        # re-format observable data-frames and set correct index.
        self.align_observable_data(item_observable_data,
                                   user_observable_data,
                                   session_observable_data,
                                   price_observable_data)

        if itemsession_observable_columns is not None:
            # merge the item-session observable columns to price_observable_columns.
            if price_observable_columns is None:
                price_observable_columns = itemsession_observable_columns
            else:
                price_observable_columns.extend(itemsession_observable_columns)

        # derive observables from columns of the main data-frame.
        self.derive_observable_from_main_data(item_observable_columns,
                                              user_observable_columns,
                                              session_observable_columns,
                                              price_observable_columns)

        self.observable_data_to_observable_tensors()

        # read in explicit numbers of items, users, and sessions.
        self._num_items = num_items
        self._num_users = num_users
        self._num_sessions = num_sessions
        self.create_choice_dataset()
        print('Finished Creating Choice Dataset.')

    def encode(self) -> None:
        """
        Encodes item names, user names, and session names to {0, 1, 2, ...} integers, item/user/session are encoded
        in alphabetical order.
        """
        # encode item names.
        self.item_name_encoder = LabelEncoder().fit(self.main_data[self.item_name_column].unique())

        # encode user indices.
        if self.user_index_column is not None:
            self.user_name_encoder = LabelEncoder().fit(self.main_data[self.user_index_column].unique())

        # encode session indices.
        if self.session_index_column is not None:
            self.session_name_encoder = LabelEncoder().fit(self.main_data[self.session_index_column].unique())

    def align_observable_data(self,
                              item_observable_data: Optional[Dict[str, pd.DataFrame]],
                              user_observable_data: Optional[Dict[str, pd.DataFrame]],
                              session_observable_data: Optional[Dict[str, pd.DataFrame]],
                              price_observable_data: Optional[Dict[str, pd.DataFrame]]) -> None:
        """This method converts observables in the dictionary format (observable name --> observable data frame), for
        each data frame of observables, this method set the appropriate corresponding index and subsets/permutes the data frames
        to have the same order as in the encoder.

        Args:
            item_observable_data (Optional[Dict[str, pd.DataFrame]]): _description_
            user_observable_data (Optional[Dict[str, pd.DataFrame]]): _description_
            session_observable_data (Optional[Dict[str, pd.DataFrame]]): _description_
            price_observable_data (Optional[Dict[str, pd.DataFrame]]): _description_
        """
        self.item_observable_data = dict()
        if item_observable_data is not None:
            for key, val in item_observable_data.items():
                # key: observable name.
                # val: data-frame of observable data.
                assert self.item_name_column in val.columns, f"{self.item_name_column} is not a column of provided item observable data-frame."
                for item in self.item_name_encoder.classes_:
                    assert item in val[self.item_name_column].values, f"item {item} is not in the {self.item_name_column} column of the item observable data-frame."

                self.item_observable_data['item_' + key] = val.set_index(self.item_name_column).loc[self.item_name_encoder.classes_]

        self.user_observable_data = dict()
        if user_observable_data is not None:
            for key, val in user_observable_data.items():
                # key: observable name.
                # val: data-frame of observable data.
                assert self.user_index_column is not None, "user observable data is provided but user index column is not provided."
                assert self.user_index_column in val.columns, f"{self.user_index_column} is not a column of provided user observable data-frame."
                for user in self.user_name_encoder.classes_:
                    assert user in val[self.user_index_column].values, f"user {user} is not in the {self.user_index_column} column of the user observable data-frame."

                self.user_observable_data['user_' + key] = val.set_index(self.user_index_column).loc[self.user_name_encoder.classes_]

        self.session_observable_data = dict()
        if session_observable_data is not None:
            for key, val in session_observable_data.items():
                # key: observable name.
                # val: data-frame of observable data.
                assert self.session_index_column is not None, "session observable data is provided but session index column is not provided."
                assert self.session_index_column in val.columns, f"{self.session_index_column} is not a column of provided session observable data-frame."
                for session in self.session_name_encoder.classes_:
                    assert session in val[self.session_index_column].values, f"session {session} is not in the {self.session_index_column} column of the session observable data-frame."

                self.session_observable_data['session_' + key] = val.set_index(self.session_index_column).loc[self.session_name_encoder.classes_]

        self.price_observable_data = dict()
        if price_observable_data is not None:
            for key, val in price_observable_data.items():
                # key: observable name.
                # val: data-frame of observable data.
                assert self.session_index_column is not None, "price observable data is provided but session index column is not provided."
                assert self.session_index_column in val.columns, f"{self.session_index_column} is not a column of provided price observable data-frame."
                assert self.item_name_column in val.columns, f"{self.item_name_column} is not a column of provided price observable data-frame."

                for session in self.session_name_encoder.classes_:
                    assert session in val[self.session_index_column].values, f"session {session} is not in the {self.session_index_column} column of the price observable data-frame."

                for item in self.item_name_encoder.classes_:
                    assert item in val[self.item_name_column].values, f"item {item} is not in the {self.item_name_column} column of the price observable data-frame."

                # need to re-index since some (session, item) pairs are allowed to be unavailable.
                # complete index = Cartesian product of all sessions and all items.
                complete_index = pd.MultiIndex.from_product([self.session_name_encoder.classes_, self.item_name_encoder.classes_],
                                                            names=[self.session_index_column, self.item_name_column])
                self.price_observable_data['itemsession_' + key] = val.set_index([self.session_index_column, self.item_name_column]).reindex(complete_index)

    def derive_observable_from_main_data(self,
                                         item_observable_columns: Optional[List[str]],
                                         user_observable_columns: Optional[List[str]],
                                         session_observable_columns: Optional[List[str]],
                                         price_observable_columns: Optional[List[str]]) -> None:
        """
        Generates data-frames of observables using certain columns in the main dataset. This is a complementary method for programers to supply variables.
        """
        if item_observable_columns is not None:
            for obs_col in item_observable_columns:
                # get the value of `obs_col` for each item.
                # note: values in self.main_data[self.item_name_column] are NOT encoded, they are raw values.
                self.item_observable_data['item_' + obs_col] = self.main_data.groupby(self.item_name_column).first()[[obs_col]].loc[self.item_name_encoder.classes_]

        if user_observable_columns is not None:
            for obs_col in user_observable_columns:
                # TODO: move to sanity check part.
                assert self.user_index_column is not None, "user observable data is required but user index column is not provided."
                self.user_observable_data['user_' + obs_col] = self.main_data.groupby(self.user_index_column).first()[[obs_col]].loc[self.user_name_encoder.classes_]

        if session_observable_columns is not None:
            for obs_col in session_observable_columns:
                self.session_observable_data['session_' + obs_col] = self.main_data.groupby(self.session_index_column).first()[[obs_col]].loc[self.session_name_encoder.classes_]

        if price_observable_columns is not None:
            for obs_col in price_observable_columns:
                val = self.main_data.groupby([self.session_index_column, self.item_name_column]).first()[[obs_col]]
                complete_index = pd.MultiIndex.from_product([self.session_name_encoder.classes_, self.item_name_encoder.classes_],
                                                            names=[self.session_index_column, self.item_name_column])
                self.price_observable_data['itemsession_' + obs_col] = val.reindex(complete_index)

    def observable_data_to_observable_tensors(self) -> None:
        """Convert all self.*_observable_data to self.*_observable_tensors for PyTorch."""
        self.item_observable_tensors = dict()
        for key, val in self.item_observable_data.items():
            assert all(val.index == self.item_name_encoder.classes_), "item observable data is not alighted with user name encoder."
            self.item_observable_tensors[key] = torch.tensor(val.values, dtype=torch.float32)

        self.user_observable_tensors = dict()
        for key, val in self.user_observable_data.items():
            assert all(val.index == self.user_name_encoder.classes_), "user observable data is not alighted with user name encoder."
            self.user_observable_tensors[key] = torch.tensor(val.values, dtype=torch.float32)

        self.session_observable_tensors = dict()
        for key, val in self.session_observable_data.items():
            assert all(val.index == self.session_name_encoder.classes_), "session observable data is not aligned with session name encoder."
            self.session_observable_tensors[key] = torch.tensor(val.values, dtype=torch.float32)

        self.price_observable_tensors = dict()
        for key, val in self.price_observable_data.items():
            tensor_slice = list()
            # if there are multiple columns (i.e., multiple observables) in the data-frame, we stack them together.
            for column in val.columns:
                df_slice = val.reset_index().pivot(index=self.session_index_column, columns=self.item_name_column, values=column)
                tensor_slice.append(torch.tensor(df_slice.values, dtype=torch.float32))

                assert np.all(df_slice.index == self.session_name_encoder.classes_)
                assert np.all(df_slice.columns == self.item_name_encoder.classes_)

            # (num_sessions, num_items, num_params)
            self.price_observable_tensors[key] = torch.stack(tensor_slice, dim=-1)

    def create_choice_dataset(self) -> None:
        print('Creating choice dataset from stata format data-frames...')
        # get choice set in each purchase record.
        choice_set_size = self.main_data.groupby(self.purchase_record_column)[self.item_name_column].nunique()
        s = choice_set_size.value_counts()
        # choice set size might be different in different purchase records due to unavailability of items.
        rep = dict(zip([f'size {x}' for x in s.index], [f'occurrence {x}' for x in s.values]))
        if len(np.unique(choice_set_size)) > 1:
            print(f'Note: choice sets of different sizes found in different purchase records: {rep}')
            self.item_availability = self.get_item_availability_tensor()
        else:
            # None means all items are available.
            self.item_availability = None

        # get the name of item bought in each purchase record.
        assert all(self.main_data[self.main_data[self.choice_column] == 1].groupby(self.purchase_record_column).size() == 1)
        item_bought = self.main_data[self.main_data[self.choice_column] == 1].set_index(self.purchase_record_column).loc[self.purchase_record_index, self.item_name_column].values
        # encode names of item bought.
        self.item_index = self.item_name_encoder.transform(item_bought)

        # user index
        if self.user_index_column is None:
            # no user index is supplied.
            self.user_index = None
        else:
            # get the user index of each purchase record.
            self.user_index = self.main_data.groupby(self.purchase_record_column)[self.user_index_column].first().loc[self.purchase_record_index].values
            # encode user indices.
            self.user_index = self.user_name_encoder.transform(self.user_index)

        # session index
        if self.session_index_column is None:
            # print('Note: no session index provided, assign each case/purchase record to a unique session index.')
            self.session_index = None
        else:
            # get session index of each purchase record.
            self.session_index = self.main_data.groupby(self.purchase_record_column)[self.session_index_column].first().loc[self.purchase_record_index].values
            self.session_index = self.session_name_encoder.transform(self.session_index)

        self.choice_dataset = ChoiceDataset(item_index=torch.LongTensor(self.item_index),
                                            user_index=torch.LongTensor(self.user_index) if self.user_index is not None else None,
                                            session_index=torch.LongTensor(self.session_index) if self.session_index is not None else None,
                                            item_availability=self.item_availability,
                                            num_items=self._num_items,
                                            num_users=self._num_users,
                                            num_sessions=self._num_sessions,
                                            # keyword arguments for observables.
                                            **self.item_observable_tensors,
                                            **self.user_observable_tensors,
                                            **self.session_observable_tensors,
                                            **self.price_observable_tensors)

        self.choice_dataset.to(self.device)

    def get_item_availability_tensor(self) -> torch.BoolTensor:
        """Get the item availability tensor from the main_data data-frame."""
        if self.session_index_column is None:
            raise ValueError(f'Item availability cannot be constructed without session index column.')
        A = self.main_data.pivot(index=self.session_index_column, columns=self.item_name_column, values=self.choice_column)
        return torch.BoolTensor(~np.isnan(A.values))

    def __len__(self):
        return len(self.purchase_record_index)

    def summary(self):
        print(f'* purchase record index range:', self.purchase_record_index[:3], '...', self.purchase_record_index[-3:])
        print(f'* Space of {len(self.item_name_encoder.classes_)} items:\n', pd.DataFrame(data={'item name': self.item_name_encoder.classes_}, index=np.arange(len(self.item_name_encoder.classes_))).T)
        print(f'* Number of purchase records/cases: {len(self)}.')
        print('* Preview of main data frame:')
        print(self.main_data)
        print('* Preview of ChoiceDataset:')
        print(self.choice_dataset)
