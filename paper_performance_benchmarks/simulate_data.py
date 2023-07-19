'''
This is a script to generate simulated data to test torch-choice on.
'''
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List
import os
import pickle as pkl
import shutil

def myjoin(a, b, on, how, **kwargs):
    return (a.set_index(on).join(b.set_index(on), how=how, **kwargs).reset_index())

np.random.seed(0)
config = {
    'num_users' : 50,
    'num_categories' : 100,
    'num_items_pcat': 5,
    'num_weeks' : 104,
    'trips_per_customer' : 80,
    'min_trips_per_customer' : 10, # take a uniform draw between min and max per user
    # common params
    'user_item' : True,
    'user_price' : True,
    'latent_dim' : 30,
    'cat_latent_dim' : 30,
    'user_price' : True,
    'cat_price' : False,
    'user_cat_price' : False,
    'item_price' : False,
    # mean and std of price coeffs, normal distribution
    'loc' : -1.5,
    'scale' : 0.3,
    'train_frac' : 0.8,
    'val_frac' : 0.1,
    'test_frac' : 0.1,
    'out_dir' : '/scratch/users/akanodia/sims23',
    'store_id' : '1',
    # intercept for category inside option
    'category_base_rate' : 7.5,
    # the number of logical user category chunks, such that user in chunk i only buys from categories in chunk i with high likelihood,
    'chunks' : 5,
    'write_latents_to_csv' : True,
}

category_to_item = {}
for cat_id in range(config['num_categories']):
    category_to_item[cat_id] = [cat_id*config['num_items_pcat'] + elem for elem in list(range(config['num_items_pcat']))]

config['num_days'] = 104*7
config['num_items'] = config['num_categories']*config['num_items_pcat'],
item_latents = np.random.normal(size=(config['num_categories']*config['num_items_pcat'], config['latent_dim']))
item_intercepts = np.random.normal(size=(config['num_categories']*config['num_items_pcat'], 1))
category_base_rate = config['category_base_rate']
category_ilatents = category_base_rate + np.random.uniform(low=0.1, high=0.9, size=(config['num_categories'], config['cat_latent_dim']))
category_iintercepts = np.random.normal(size=(config['num_categories'], 1)) - category_base_rate
category_iivs_coefs = np.random.uniform(low=0.3, high=0.9, size=(config['num_categories'], 1))
category_to_item = {}
for cat_id in range(config['num_categories']):
    category_to_item[cat_id] = [cat_id*config['num_items_pcat']+ elem for elem in list(range(config['num_items_pcat']))]

pre_params = {}
pre_params['item_latents'] = item_latents
pre_params['item_intercepts'] = item_intercepts
pre_params['category_to_item'] = category_to_item
pre_params['category_ilatents'] = category_ilatents
pre_params['category_iintercepts'] = category_iintercepts
pre_params['category_iivs_coefs'] = category_iivs_coefs
pre_params['upc_utility'] = 'lambda_item + theta_user * beta_item + gamma_user * price_obs'
pre_params['category_utility'] = 'lambda_item + theta_user * beta_item + gamma_user * price_obs + delta_item * session_dayofweek + mu_item * session_week'

params_per_store = ['user_latents', 'user_price_coeffs', 'user_latents_cat', 'category_weekday_effects', 'category_week_effects']

def mylogsumexp(data):
    maxval = np.max(data)
    datanew = data - maxval
    logsumexp = np.log(np.sum(np.exp(datanew)))
    return logsumexp + maxval

def one_availability(inventory_rates, potential_drops=3, drop_rate=0.5):
    ans = np.ones(inventory_rates.shape)
    for ii, row in enumerate(inventory_rates):
        # candidates = np.random.multinomial(1, row, size=potential_drops)
        candidates = np.random.choice(len(row), size=potential_drops, p=row)
        drop_indicators = np.random.binomial(1, drop_rate, size=(potential_drops,))
        for jj in range(len(candidates)):
            if drop_indicators[jj] == 1:
                ans[ii, candidates[jj]] = 0
    return ans

def sim_data(config, pre_params, debug=False, seed=0):
    np.random.seed(seed)
    num_users = config['num_users']
    user_ids = list(range(num_users))
    user_ids = [int(config['store_id'] + str(elem)) for elem in user_ids]
    user_ids_df = pd.DataFrame(user_ids, columns=['user_id'])
    num_weeks = config['num_weeks']
    trips_per_customer = config['trips_per_customer']
    trips_observed = np.random.choice(list(range(config['min_trips_per_customer'], config['trips_per_customer'] + 1)), num_users, replace=True)
    # trips_observed = {ii:elem for ii, elem in enumerate(trips_observed)}
    num_days = num_weeks * 7
    num_sessions = num_days
    num_categories = config['num_categories']
    num_items_pcat = config['num_items_pcat']

    inventory_rates = []
    for ii in range(num_categories):
        this_rate = np.random.uniform(low=0.5, high=30.0, size=(num_items_pcat))
        this_rate = this_rate / np.sum(this_rate)
        inventory_rates.append(this_rate)
    inventory_rates = np.array(inventory_rates)
    inventory = []
    for w in range(num_weeks):
        to_append = one_availability(inventory_rates).reshape(-1)
        for dd in range(7):
            inventory.append(to_append)
    inventory = np.array(inventory)
    inventory_utils = (1 - inventory) * -1000000.0
    weekly_prices = np.random.choice([0.5, 0.75, 1.00, 1.25, 1.50], size=(num_weeks, config['num_categories']* config['num_items_pcat']))
    daily_prices = np.repeat(weekly_prices, 7, axis=0)
    session_ids = list(range(num_days))
    session_ids_store = [int(config['store_id'] + str(elem)) for elem in session_ids]
    session_ids_df = pd.DataFrame(session_ids_store, columns=['session_id'])
    session_days = np.asarray(session_ids)
    session_weekdays = np.asarray([elem % 7 for elem in session_ids])
    session_weekids = np.asarray([elem // 7 for elem in session_ids])
    users_trips = []
    for user in range(num_users):
        user_trips = np.random.choice(session_ids, trips_observed[user], replace=False)
        user_trips = [[user, elem] for elem in sorted(user_trips)]
        users_trips.append(user_trips)
    users_trips = np.vstack(users_trips)
    trips_prices = daily_prices[users_trips[:, 1]]
    trips_inventory_utils = inventory_utils[users_trips[:, 1]]
    trips_days = session_days[users_trips[:, 1]]
    trips_weekdays = session_weekdays[users_trips[:, 1]]
    trips_weekids = session_weekids[users_trips[:, 1]]
    category_to_item = pre_params['category_to_item']

    latent_dim = config['latent_dim']
    user_latents = np.random.normal(size=(num_users, latent_dim))
    item_latents = pre_params['item_latents']
    item_intercepts = pre_params['item_intercepts']
    params = {
        'user_latents' : user_latents,
        'item_latents' : item_latents,
        'item_intercepts' : item_intercepts
    }

    user_latents_trips = user_latents[users_trips[:, 0]]

    ui_latents_trips =  user_latents_trips @ item_latents.T
    i_latents_trips = np.ones((user_latents_trips.shape[0], 1)) @ item_intercepts.T

    if debug:
        print(user_latents_trips.shape, item_latents.shape, ui_latents_trips.shape, i_latents_trips.shape)

    if config['cat_price']:
        cat_price_coeffs = np.random.normal(loc=config['loc'], scale=config['scale'], size=(config['num_categories'], 1))
        item_price_coeffs = np.repeat(cat_price_coeffs, config['num_items_pcat'], axis=0)
        params['cat_price_coeffs'] = cat_price_coeffs
        params['item_price_coeffs'] = item_price_coeffs
    elif config['item_price']:
        item_price_coeffs = np.random.normal(loc=config['loc'], scale=config['scale'], size=(config['num_categories']* config['num_items_pcat'], 1))
        params['item_price_coeffs'] = item_price_coeffs
    elif config['user_price']:
        user_price_coeffs = np.random.normal(loc=config['loc'], scale=config['scale'], size=(num_users, 1))
        params['user_price_coeffs'] = user_price_coeffs
    elif config['user_cat_price']:
        user_price_coeffs = np.random.normal(loc=config['loc'], scale=config['scale'], size=(num_users, 1))
        params['user_price_coeffs'] = user_price_coeffs
        cat_price_coeffs = np.random.normal(loc=config['loc2'], scale=config['scale2'], size=(config['num_categories'], 1))
        params['cat_price_coeffs'] = cat_price_coeffs
        user_cat_price_coeffs = user_price_coeffs @ np.repeat(cat_price_coeffs, config['num_items_pcat'], axis=0).T
        params['user_cat_price_coeffs'] = user_cat_price_coeffs
    else:
        raise ValueError("Exactly one of the price coeff flags must be set to true")


    if debug:
        if config['item_price']or config['cat_price']:
            print(item_price_coeffs.shape, item_latents.shape)
        elif config['user_price']:
            print(user_price_coeffs.shape)
        elif config['user_cat_price']:
            print(user_price_coeffs.shape)
            print(cat_price_coeffs.shape)
            print(user_cat_price_coeffs.shape)

    if config['cat_price'] or config['item_price']:
        trips_item_pcoeffs = np.repeat(item_price_coeffs.T, trips_prices.shape[0], axis=0)
        obs_utils = i_latents_trips + trips_prices*trips_item_pcoeffs
    elif config['user_price']:
        trips_pcoeffs = user_price_coeffs[users_trips[:, 0]]
        obs_utils = i_latents_trips + trips_prices*trips_pcoeffs
    elif config['user_cat_price']:
        trips_pcoeffs = user_cat_price_coeffs[users_trips[:, 0]]
        obs_utils = i_latents_trips + trips_prices*trips_pcoeffs

    if config['user_item']:
        ll_term = user_latents@item_latents.T
        ll_term = ll_term[users_trips[:, 0], :]
        obs_utils += ll_term

    mu_gumbel = 0.0
    beta_gumbel = 1.0
    EUL_MAS_CONST = 0.5772156649
    loc = mu_gumbel + beta_gumbel * EUL_MAS_CONST
    gumbel_samples = np.random.gumbel(loc=loc, scale=1.0, size=obs_utils.shape)
    all_utils = obs_utils + gumbel_samples + trips_inventory_utils

    if debug:
        i_latents_trips.shape,  daily_prices.shape, trips_prices.shape, obs_utils.shape, gumbel_samples.shape, all_utils.shape

    all_items_purchased = np.argmax(all_utils.reshape(all_utils.shape[0], config['num_categories'], config['num_items_pcat']), axis=2)

    item_indices = []
    for row in all_items_purchased:
        trip_item_indices = [cat*config['num_items_pcat']+ elem for cat, elem in enumerate(row)]
        item_indices.extend(trip_item_indices)

    cat_ivs = np.apply_along_axis(mylogsumexp, 2, all_utils.reshape(all_utils.shape[0], config['num_categories'], config['num_items_pcat']))
    user_latents_cat = category_base_rate + np.random.uniform(low=0.1, high=0.9, size=(num_users, config['cat_latent_dim']))
    chunks = config['chunks']
    assert num_users % chunks == 0, "Number of users must be divisible by number of chunks"
    assert num_categories % chunks == 0, "Number of categories must be divisible by number of chunks"
    chunk_size_latent = config['cat_latent_dim'] // chunks
    user_chunk_map = {int(config['store_id'] + str(ii)):ii//(num_users//chunks) for ii in range(num_users)}
    cat_chunk_map = {ii:ii//(num_categories//chunks) for ii in range(num_categories)}
    # zero out all but one chunk for each user
    for ii in range(user_latents_cat.shape[0]):
        # The first block of users goes to the first chunk, the second block to the second chunk, etc.
        chunk_index_latent = ii // (num_users // chunks)
        block_vector = np.zeros(config['cat_latent_dim'])
        block_vector[chunk_index_latent*chunk_size_latent:(chunk_index_latent+1)*chunk_size_latent] = 1
        user_latents_cat[ii, :] *= block_vector
    category_ilatents = pre_params['category_ilatents']
    # zero out all but one chunk for each category
    for ii in range(category_ilatents.shape[0]):
        # The first block of categories goes to the first chunk, the second block to the second chunk, etc.
        chunk_index_latent = ii // (num_categories // chunks)
        block_vector = np.zeros(config['cat_latent_dim'])
        block_vector[chunk_index_latent*chunk_size_latent:(chunk_index_latent+1)*chunk_size_latent] = 1
        category_ilatents[ii, :] *= block_vector

    category_iintercepts = pre_params['category_iintercepts']
    category_iivs_coefs = pre_params['category_iivs_coefs']
    category_weekday_effects = np.random.normal(size=(7, config['num_categories']))
    category_week_effects = np.random.normal(size=(num_weeks, config['num_categories']))
    params['user_latents_cat'] = user_latents_cat
    params['category_weekday_effects'] = category_weekday_effects
    params['category_week_effects'] = category_week_effects

    user_latents_trips_cat = user_latents_cat[users_trips[:, 0]]
    uc_latents_trips =  user_latents_trips_cat @ category_ilatents.T
    ci_latents_trips = np.ones((user_latents_trips_cat.shape[0], 1)) @ category_iintercepts.T
    trips_civcoeffs = np.repeat(category_iivs_coefs.T, cat_ivs.shape[0], axis=0)
    weekday_effects = category_weekday_effects[trips_weekdays]
    week_effects = category_week_effects[trips_weekids]
    cat_obs_utils = ci_latents_trips + cat_ivs*trips_civcoeffs + uc_latents_trips# + weekday_effects + week_effects
    gumbel_samples_cats = np.random.gumbel(loc=loc, scale=1.0, size=cat_obs_utils.shape) - np.random.gumbel(loc=loc, scale=1.0, size=cat_obs_utils.shape)
    cat_utils = cat_obs_utils + gumbel_samples_cats
    cat_utils_outside = np.zeros(cat_utils.shape)
    cat_utils_all = np.stack((cat_utils_outside, cat_utils), axis=2)
    all_cats_purchased = np.argmax(cat_utils_all, axis=2)
    # count the fraction of inside option purchases per user per chunk
    chunk_size = num_users // chunks
    cat_indices = []
    for row in all_cats_purchased:
        trip_cat_indices = [cat*10 + elem for cat, elem in enumerate(row)]
        cat_indices.extend(trip_cat_indices)

    users_trips_upc = users_trips.copy()
    users_trips_upc[:, 0] = [int(config['store_id'] + str(row[0])) for row in users_trips_upc]
    users_trips_upc[:, 1] = [int(config['store_id'] + str(row[1])) for row in users_trips_upc]

    users_trips_cat = users_trips.copy()
    users_trips_cat[:, 0] = [int(config['store_id'] + str(row[0])) for row in users_trips_cat]
    users_trips_cat[:, 1] = [int(config['store_id'] + str(num_days*row[0] + row[1])) for row in users_trips_cat]

    user_index_upc = np.repeat(users_trips_upc[:, 0], config['num_categories'])
    user_index_cat = np.repeat(users_trips_cat[:, 0], config['num_categories'])

    session_index_upc = np.repeat(users_trips_upc[:, 1], config['num_categories'])
    session_index_cat = np.repeat(users_trips_cat[:, 1], config['num_categories'])

    item_index_upc = np.asarray(item_indices)
    item_index_cat = np.asarray(cat_indices)

    cat_data = pd.DataFrame()
    cat_data['user_id'] = user_index_cat
    cat_data['cat_session_id'] = session_index_cat
    cat_data['item_id'] = item_index_cat
    cat_data['quantity'] = 1
    cat_data['category_id'] = item_index_cat // 10
    cat_data['user_chunk'] = [user_chunk_map[elem] for elem in cat_data['user_id']]
    cat_data['cat_chunk'] = [cat_chunk_map[elem] for elem in cat_data['category_id']]
    cat_data['inside_option'] = item_index_cat % 10
    # mean purchase rates in each chunk pair
    chunk_purchase_matrix = cat_data[['user_chunk', 'cat_chunk', 'inside_option']].groupby(['user_chunk', 'cat_chunk'])['inside_option'].mean()
    chunk_purchase_matrix = chunk_purchase_matrix.values.reshape((chunks, chunks))

    upc_data = pd.DataFrame()
    upc_data['user_id'] = user_index_upc
    upc_data['item_id'] = item_index_upc
    upc_data['upc_session_id'] = session_index_upc
    upc_data['category_purchased'] = item_index_cat
    upc_data['quantity'] = 1

    cat_ivs_df_inside = pd.DataFrame()
    cat_ivs_df_inside['item_id'] = list(range(config['num_categories'])) * len(users_trips)
    cat_ivs_df_inside['item_id'] = cat_ivs_df_inside['item_id']*10 + 1
    cat_ivs_df_inside['session_id'] = session_index_cat
    cat_ivs_df_inside['iv'] = cat_ivs.reshape(-1)
    cat_ivs_df_outside = cat_ivs_df_inside.copy()
    cat_ivs_df_outside['item_id'] -= 1
    cat_ivs_df_outside['iv'] = 0.0

    cat_ivs = pd.concat([cat_ivs_df_outside, cat_ivs_df_inside])

    item_sess_price = pd.DataFrame()
    item_sess_price['item_id'] = np.asarray(list(range(config['num_categories']*config['num_items_pcat']))).repeat(num_days)
    item_sess_price['session_id'] = list(range(num_days)) * (config['num_categories'] * config['num_items_pcat'])
    item_sess_price['session_id'] = [int(config['store_id'] + str(elem)) for elem in item_sess_price['session_id']]
    item_sess_price['price'] = daily_prices.transpose().reshape(-1)

    inventory_df = pd.DataFrame()
    inventory_df['item_id'] = np.asarray(list(range(config['num_categories']*config['num_items_pcat']))).repeat(num_days)
    inventory_df['session_id'] = list(range(num_days)) * (config['num_categories'] * config['num_items_pcat'])
    inventory_df['session_id'] = [int(config['store_id'] + str(elem)) for elem in inventory_df['session_id']]
    inventory_df['availability'] = inventory.transpose().reshape(-1)
    inventory_df = inventory_df[inventory_df['availability'] == 1]
    inventory_df = inventory_df[['session_id', 'item_id']]

    sess_days_upc = pd.DataFrame()
    sess_days_upc['session_id'] = session_ids
    sess_days_upc['session_id'] = [int(config['store_id'] + str(elem)) for elem in sess_days_upc['session_id']]
    sess_days_upc['week'] = session_weekids
    sess_days_upc['weekday'] = session_weekdays
    sess_days_upc['hour'] = 1
    sess_days_upc['store_id'] = int(config['store_id'])
    sess_days_upc = sess_days_upc.drop_duplicates()

    sess_days_cat = pd.DataFrame()
    sess_days_cat['session_id'] = session_index_cat
    sess_days_cat['user_id'] = user_index_cat
    sess_days_cat['date'] = trips_days.repeat(config['num_categories'])
    sess_days_cat['date'] = [int(config['store_id'] + str(elem)) for elem in sess_days_cat['date']]
    sess_days_cat['week'] = trips_weekids.repeat(config['num_categories'])
    sess_days_cat['weekday'] = trips_weekdays.repeat(config['num_categories'])
    sess_days_cat['hour'] = 1
    sess_days_cat['store_id'] = int(config['store_id'])
    sess_days_cat = sess_days_cat.drop_duplicates()

    session_map = sess_days_cat.copy()
    sess_days_cat = sess_days_cat.drop(['user_id', 'date'], axis=1)

    trip_order = np.random.uniform(size=users_trips.shape[0])
    trip_info = pd.DataFrame()
    trip_info['user_id'] = users_trips_upc[:, 0]
    trip_info['upc_session_id'] = users_trips_upc[:, 1]
    trip_info['cat_session_id'] = users_trips_cat[:, 1]
    trip_info['rand'] = trip_order
    trip_info = trip_info.sort_values(by=['user_id', 'rand'])

    train_cum = config['train_frac']
    val_cum = config['val_frac'] + config['train_frac']

    train_trips = trip_info[(trip_info['rand'] <= train_cum)]
    val_trips = trip_info[(trip_info['rand'] > train_cum) & (trip_info['rand'] <= val_cum)]
    test_trips = trip_info[(trip_info['rand'] > val_cum)]

    train_trips_upc = train_trips.copy()[['user_id', 'upc_session_id']]
    val_trips_upc = val_trips.copy()[['user_id', 'upc_session_id']]
    test_trips_upc = test_trips.copy()[['user_id', 'upc_session_id']]

    outdir = f'{config["out_dir"]}/{config["store_id"]}'
    # Write data to disk
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cat_outdir = f'{outdir}/category_level'

    if not os.path.exists(cat_outdir):
        os.makedirs(cat_outdir)

    # Subset on only those upc data rows for which the inside option is purchased
    upc_data_all = upc_data.copy().drop(columns=['category_purchased'])
    train_upc = myjoin(upc_data_all, train_trips_upc, ['user_id', 'upc_session_id'], 'inner')
    val_upc = myjoin(upc_data_all, val_trips_upc, ['user_id', 'upc_session_id'], 'inner')
    test_upc = myjoin(upc_data_all, test_trips_upc, ['user_id', 'upc_session_id'], 'inner')
    train_upc = train_upc[['user_id', 'item_id', 'upc_session_id', 'quantity']]
    val_upc = val_upc[['user_id', 'item_id', 'upc_session_id', 'quantity']]
    test_upc = test_upc[['user_id', 'item_id', 'upc_session_id', 'quantity']]

    train_upc.to_csv(f'{outdir}/train_all.tsv', index=False, sep='\t', header=False)
    val_upc.to_csv(f'{outdir}/validation_all.tsv', index=False, sep='\t', header=False)
    test_upc.to_csv(f'{outdir}/test_all.tsv', index=False, sep='\t', header=False)

    print('Subsetting on only those upc data rows for which the inside option is purchased')
    print('Before: ', upc_data.shape)
    upc_data = upc_data[upc_data['category_purchased'].apply(lambda x: (str(x).endswith('1')))]
    print('After: ', upc_data.shape)
    upc_data = upc_data.drop(columns=['category_purchased'])

    train_upc = myjoin(upc_data, train_trips_upc, ['user_id', 'upc_session_id'], 'inner')
    val_upc = myjoin(upc_data, val_trips_upc, ['user_id', 'upc_session_id'], 'inner')
    test_upc = myjoin(upc_data, test_trips_upc, ['user_id', 'upc_session_id'], 'inner')
    train_upc = train_upc[['user_id', 'item_id', 'upc_session_id', 'quantity']]
    val_upc = val_upc[['user_id', 'item_id', 'upc_session_id', 'quantity']]
    test_upc = test_upc[['user_id', 'item_id', 'upc_session_id', 'quantity']]

    train_trips_cat = train_trips.copy()[['user_id', 'cat_session_id']]
    val_trips_cat = val_trips.copy()[['user_id', 'cat_session_id']]
    test_trips_cat = test_trips.copy()[['user_id', 'cat_session_id']]

    train_cat = myjoin(cat_data, train_trips_cat, ['user_id', 'cat_session_id'], 'inner')
    val_cat = myjoin(cat_data, val_trips_cat, ['user_id', 'cat_session_id'], 'inner')
    test_cat = myjoin(cat_data, test_trips_cat, ['user_id', 'cat_session_id'], 'inner')
    train_cat = train_cat[['user_id', 'item_id', 'cat_session_id', 'quantity']]
    val_cat = val_cat[['user_id', 'item_id', 'cat_session_id', 'quantity']]
    test_cat = test_cat[['user_id', 'item_id', 'cat_session_id', 'quantity']]

    ivs_df = pd.DataFrame()
    ivs_df['date'] = users_trips_upc[:, 1]
    ivs_df['user_id'] = users_trips_upc[:, 0]
    ivs_df['session_counter'] = users_trips_cat[:, 1]

    item_group_upc = pd.DataFrame()
    item_group_upc['item_id'] = list(range(config['num_categories']*config['num_items_pcat']))
    item_group_upc['category_id'] = item_group_upc['item_id'].apply(lambda x: x // config['num_items_pcat'])
    item_ids = list(range(config['num_categories']*config['num_items_pcat']))
    item_ids_df = pd.DataFrame(item_ids, columns=['item_id'])
    category_ids = list(range(config['num_categories']))
    category_ids_df = pd.DataFrame(category_ids, columns=['category_id'])

    item_group_category_inside = pd.DataFrame()
    item_group_category_inside['category_id'] = list(range(config['num_categories']))
    item_group_category_inside['item_id'] = item_group_category_inside['category_id'].apply(lambda x: x * 10)
    item_group_category_outside = pd.DataFrame()
    item_group_category_outside['category_id'] = list(range(config['num_categories']))
    item_group_category_outside['item_id'] = item_group_category_inside['category_id'].apply(lambda x: x * 10 + 1)
    item_group_category = pd.concat([item_group_category_inside, item_group_category_outside])
    item_group_category = item_group_category[['item_id', 'category_id']]
    category_item_ids_df = pd.DataFrame(item_group_category['item_id'], columns=['item_id'])

    train_upc.to_csv(f'{outdir}/train.tsv', index=False, sep='\t', header=False)
    val_upc.to_csv(f'{outdir}/validation.tsv', index=False, sep='\t', header=False)
    test_upc.to_csv(f'{outdir}/test.tsv', index=False, sep='\t', header=False)
    sess_days_upc.to_csv(f'{outdir}/sess_days.tsv', index=False, sep='\t', header=False)
    item_sess_price.to_csv(f'{outdir}/item_sess_price.tsv', index=False, sep='\t', header=False)
    inventory_df.to_csv(f'{outdir}/availabilityList.tsv', index=False, sep='\t', header=False)
    item_group_upc.to_csv(f'{outdir}/itemGroup.tsv', index=False, sep='\t', header=False)
    item_group_category.to_csv(f'{outdir}/cat_itemGroup.tsv', index=False, sep='\t', header=False)
    item_group_upc.to_csv(f'{outdir}/itemGroup.csv', index=False)
    item_group_category.to_csv(f'{outdir}/cat_itemGroup.csv', index=False)
    user_ids_df.to_csv(f'{outdir}/all_users.csv', index=False)
    session_ids_df.to_csv(f'{outdir}/all_sessions.csv', index=False)
    item_ids_df.to_csv(f'{outdir}/all_items.csv', index=False)
    category_ids_df.to_csv(f'{outdir}/all_categories.csv', index=False)
    category_item_ids_df.to_csv(f'{outdir}/all_category_items.csv', index=False)
    cat_session_ids_df = session_map[['session_id']]
    session_map.to_csv(f'{outdir}/sessionMap.csv', index=False)
    cat_session_ids_df.to_csv(f'{outdir}/all_category_sessions.csv', index=False)
    ivs_df.to_csv(f'{outdir}/ivs_df.csv', index=False)
    # print user_latents_df to user_latents.csv
    if config['write_latents_to_csv']:
        if config['cat_price'] or config['item_price'] or config['user_cat_price']:
            raise ValueError('Not supported with write latents')

        user_latents_df = pd.DataFrame(user_latents, columns=list(range(config['latent_dim'])))
        user_latents_df['user_id'] = user_ids
        if config['user_price']:
            user_latents_df['price_coef'] = user_price_coeffs[:, 0]
        user_latents_df.to_csv(f'{outdir}/user_latents.csv', index=False)
        # print item_latents_df to user_latents.csv
        item_latents_df = pd.DataFrame(item_latents, columns=list(range(config['latent_dim'])))
        item_latents_df['item_id'] = item_ids
        item_latents_df.to_csv(f'{outdir}/item_latents.csv', index=False)
        # print category_latents_df to category_latents.csv
        category_ilatents_df = pd.DataFrame(category_ilatents, columns=list(range(config['latent_dim'])))
        category_ilatents_df['category_id'] = category_ids
        category_ilatents_df['intercept'] = category_iintercepts[:, 0]
        category_ilatents_df['iv_coef'] = category_iivs_coefs
        category_ilatents_df.to_csv(f'{outdir}/category_ilatents.csv', index=False)


    train_cat.to_csv(f'{cat_outdir}/train.tsv', index=False, sep='\t', header=False)
    val_cat.to_csv(f'{cat_outdir}/validation.tsv', index=False, sep='\t', header=False)
    test_cat.to_csv(f'{cat_outdir}/test.tsv', index=False, sep='\t', header=False)
    cat_ivs.to_csv(f'{cat_outdir}/item_sess_price.tsv', index=False, sep='\t', header=False)
    cat_ivs.to_csv(f'{cat_outdir}/availabilityList.tsv', index=False, sep='\t', header=False, columns=['session_id', 'item_id'])
    sess_days_cat.to_csv(f'{cat_outdir}/sess_days.tsv', index=False, sep='\t', header=False)
    session_map.to_csv(f'{cat_outdir}/sessionMap.csv', index=False)
    item_group_category.to_csv(f'{cat_outdir}/itemGroup.tsv', index=False, sep='\t', header=False)
    user_ids_df.to_csv(f'{cat_outdir}/all_users.csv', index=False)
    category_item_ids_df.to_csv(f'{cat_outdir}/all_items.csv', index=False)
    category_ids_df.to_csv(f'{cat_outdir}/all_categories.csv', index=False)
    category_item_ids_df.to_csv(f'{cat_outdir}/all_category_items.csv', index=False)
    cat_session_ids_df.to_csv(f'{cat_outdir}/all_category_sessions.csv', index=False)

    params_dfs = {}
    price_coeffs = pd.DataFrame()
    price_coeffs['user_id'] = user_ids_df['user_id']
    price_coeffs['0'] = params['user_price_coeffs']
    params_dfs['user_coeff'] = price_coeffs
    item_const = pd.DataFrame()
    item_const['item_id'] = list(range(config['num_categories']*config['num_items_pcat']))
    item_const['0'] = pre_params['item_intercepts']
    params_dfs['lambda'] = item_const
    item_latents = pd.DataFrame()
    item_latents['item_id'] = list(range(config['num_categories']*config['num_items_pcat']))
    item_latents[list(range(config['latent_dim']))] = pre_params['item_latents']
    params_dfs['beta'] = item_latents
    user_latents = pd.DataFrame()
    user_latents['user_id'] = user_ids_df['user_id']
    user_latents[list(range(config['latent_dim']))] = params['user_latents']
    params_dfs['theta_user'] = user_latents

    params_dfs_cat = {}
    price_coeffs_inside = pd.DataFrame()
    price_coeffs_inside['category_id'] = list(range(config['num_categories']))
    price_coeffs_inside['0'] = pre_params['category_iivs_coefs']
    params_dfs_cat['nfact'] = price_coeffs_inside
    item_const_inside = pd.DataFrame()
    item_const_inside['item_id'] = list(range(config['num_categories']))
    item_const_inside['item_id'] = item_const_inside['item_id'].apply(lambda x: x * 10 + 1)
    item_const_inside['0'] = pre_params['category_iintercepts']
    item_const_outside = pd.DataFrame()
    item_const_outside['item_id'] = list(range(config['num_categories']))
    item_const_outside['item_id'] = item_const_outside['item_id'].apply(lambda x: x * 10)
    item_const_outside['0'] = 0.0
    item_const_cat = pd.concat([item_const_inside, item_const_outside])
    params_dfs_cat['lambda'] = item_const_inside
    item_latents_inside = pd.DataFrame()
    item_latents_inside['item_id'] = list(range(config['num_categories']))
    item_latents_inside['item_id'] = item_latents_inside['item_id'].apply(lambda x: x * 10 + 1)
    item_latents_inside[list(range(config['cat_latent_dim']))] = pre_params['category_ilatents']
    item_latents_outside = pd.DataFrame()
    item_latents_outside['item_id'] = list(range(config['num_categories']))
    item_latents_outside['item_id'] = item_latents_outside['item_id'].apply(lambda x: x * 10)
    item_latents_outside[list(range(config['cat_latent_dim']))] = np.zeros((config['num_categories'], config['cat_latent_dim']))
    item_latents_cat = pd.concat([item_latents_inside, item_latents_outside])
    params_dfs_cat['beta'] = item_latents_inside
    user_latents_cat = pd.DataFrame()
    user_latents_cat['user_id'] = user_ids_df['user_id']
    user_latents_cat[list(range(config['cat_latent_dim']))] = params['user_latents_cat']
    params_dfs_cat['theta_user'] = user_latents_cat
    delta_inside = pd.DataFrame()
    delta_inside['item_id'] = list(range(config['num_categories']))
    delta_inside['item_id'] = delta_inside['item_id'].apply(lambda x: x * 10 + 1)
    delta_inside[list(range(7))] = params['category_weekday_effects'].transpose()
    delta_outside = pd.DataFrame()
    delta_outside['item_id'] = list(range(config['num_categories']))
    delta_outside['item_id'] = delta_outside['item_id'].apply(lambda x: x * 10)
    delta_outside[list(range(7))] = np.zeros((config['num_categories'], 7))
    delta = pd.concat([delta_inside, delta_outside])
    params_dfs_cat['delta'] = delta_inside
    mu_inside = pd.DataFrame(params['category_week_effects'].transpose(), columns=list(range(num_weeks)))
    mu_inside['item_id'] = [elem * 10 + 1 for elem in list(range(config['num_categories']))]
    mu_inside['item_id'] = mu_inside['item_id'].apply(lambda x: x * 10 + 1)
    mu_outside = pd.DataFrame()
    mu_inside = pd.DataFrame(np.zeros((config['num_categories'], num_weeks)), columns=list(range(num_weeks)))
    mu_outside['item_id'] = list(range(config['num_categories']))
    mu_outside['item_id'] = mu_outside['item_id'].apply(lambda x: x * 10)
    mu_outside[list(range(num_weeks))] = np.zeros((config['num_categories'], num_weeks))
    mu = pd.concat([mu_inside, mu_outside])
    params_dfs_cat['mu'] = mu_inside

    diff_params = set(params_per_store) - set(params.keys())
    assert diff_params == set(), f"{diff_params} \nparams are missing that need to be generated per store"
    write_params = params.copy()
    write_params.update(pre_params)
    pkl.dump(write_params, open(f'{outdir}/params.pkl', 'wb'))
    pkl.dump(params_dfs, open(f'{outdir}/params_dfs.pkl', 'wb'))
    pkl.dump(params_dfs_cat, open(f'{outdir}/category_level/params_dfs_cat.pkl', 'wb'))
    return

def merge_store_data(config, store_ids, pre_params):
    '''
    Merge store data for all stores store_ids
    This assumes that all store sims for each data was generated using the same `pre_params`
    '''
    out_suffix = '-'.join(store_ids)
    upc_outdir = f'{config["out_dir"]}/{out_suffix}'
    cat_outdir = f'{config["out_dir"]}/{out_suffix}/category_level'

    if not os.path.exists(upc_outdir):
        os.makedirs(upc_outdir)

    if not os.path.exists(cat_outdir):
        os.makedirs(cat_outdir)

    params_store = pkl.load(open(f'{config["out_dir"]}/{store_ids[0]}/params.pkl', 'rb'))
    params = {k: params_store[k] for k in params_per_store}
    for store_id in store_ids[1:]:
        print(f'Merging store data params for store {store_id}')
        params_store = pkl.load(open(f'{config["out_dir"]}/{store_id}/params.pkl', 'rb'))
        for key in params.keys():
            params[key] = np.concatenate([params[key], params_store[key]])
    params.update(pre_params)

    params_dfs = pkl.load(open(f'{config["out_dir"]}/{store_ids[0]}/params_dfs.pkl', 'rb'))
    skip_keys = {'lambda', 'beta'}
    for store_id in store_ids[1:]:
        print(f'Merging store data dfs for store {store_id}')
        params_dfs_store = pkl.load(open(f'{config["out_dir"]}/{store_id}/params_dfs.pkl', 'rb'))
        for key in params_dfs.keys():
            if key not in skip_keys:
                params_dfs[key] = pd.concat((params_dfs[key], params_dfs_store[key]), axis=0)

    params_dfs_cat = pkl.load(open(f'{config["out_dir"]}/{store_ids[0]}/category_level/params_dfs_cat.pkl', 'rb'))
    skip_keys = {'ifact', 'gfact', 'nfact', 'lambda', 'beta'}
    for store_id in store_ids[1:]:
        print(f'Merging store data dfs for store {store_id}')
        params_dfs_cat_store = pkl.load(open(f'{config["out_dir"]}/{store_id}/category_level/params_dfs_cat.pkl', 'rb'))
        for key in params_dfs_cat.keys():
            if key not in skip_keys:
                params_dfs_cat[key] = pd.concat((params_dfs_cat[key], params_dfs_cat_store[key]), axis=0)

    # To concat
    to_concat_files_upc = [
        'train.tsv', 'test.tsv', 'validation.tsv', 'sess_days.tsv',
        'item_sess_price.tsv', 'all_users.csv', 'ivs_df.csv']

    to_concat_files_cat = to_concat_files_upc.copy() + ['sessionMap.csv']

    to_concat_files_upc.extend(['train_all.tsv', 'test_all.tsv', 'validation_all.tsv'])

    to_copy_files_common = ['itemGroup.tsv']

    pkl.dump(params, open(f'{upc_outdir}/params.pkl', 'wb'))
    pkl.dump(params_dfs, open(f'{upc_outdir}/params_dfs.pkl', 'wb'))
    pkl.dump(params_dfs_cat, open(f'{cat_outdir}/params_dfs_cat.pkl', 'wb'))

    def concat_single(filename, out_dir, category_level=False):
        if filename.endswith('.tsv'):
            sep = '\t'
            header = None
        elif filename.endswith('.csv'):
            sep = ','
            header = 0
        if category_level:
            category_suffix = 'category_level'
        else:
            category_suffix = ''
        df = pd.read_csv(f'{config["out_dir"]}/{store_ids[0]}/{category_suffix}/{filename}', sep=sep, header=header)
        for store_id in store_ids[1:]:
            df_store = pd.read_csv(f'{config["out_dir"]}/{store_id}/{category_suffix}/{filename}', sep=sep, header=header)
            df = pd.concat([df, df_store])
        index=False
        if filename.endswith('.tsv'):
            sep = '\t'
            header = False
        elif filename.endswith('.csv'):
            sep = ','
            header = True
        df.to_csv(f'{out_dir}/{filename}', index=index, sep=sep, header=header)

    for filename in to_copy_files_common:
        shutil.copyfile(f'{config["out_dir"]}/{store_ids[0]}/{filename}', f'{upc_outdir}/{filename}')
        shutil.copyfile(f'{config["out_dir"]}/{store_ids[0]}/category_level/{filename}', f'{cat_outdir}/{filename}')

    for filename in to_concat_files_upc:
        print(f'Concatenating {filename}')
        concat_single(filename, out_dir=upc_outdir)

    for filename in to_concat_files_cat:
        print(f'Concatenating {filename}')
        concat_single(filename, out_dir=cat_outdir, category_level=True)

    return

if __name__ == "__main__":
    store_ids = ['1']
    for store_id in store_ids:
        config['store_id'] = store_id
        sim_data(config, pre_params, seed=int(store_id))
        tsvdir = f'{config["out_dir"]}/{store_id}'
        outdir = f'{config["out_dir"]}/{store_id}/category_level'
        store_effects = True
