


if __name__ == '__main__':
user_obs = torch.rand(num_users, 3)
item_obs = torch.rand(num_items, 3)
session_obs = torch.rand(num_sessions, 3)
# price_obs = torch.randn(num_sessions, num_items, 12)
item_index = torch.LongTensor(np.random.choice(num_items, size=N))
user_index = torch.LongTensor(np.random.choice(num_users, size=N))
session_index = torch.LongTensor(np.random.choice(num_sessions, size=N))
item_availability = torch.ones(num_sessions, num_items).bool()

    item_index = list()
    for n in tqdm(range(N)):
        u, s = user_index[n], session_index[n]
        if np.random.rand() <= rational_prob:
            # (num_items, 1)
            # utilities = lambda_item + (beta_user[u].view(1, -1).expand(num_items, -1) * item_obs).sum(dim=-1) + (gamma_constant.view(1, -1).expand(num_items, -1) * session_obs[s].view(1, -1).expand(num_items, -1)).sum(dim=-1)
            utilities = lambda_item
            p = torch.nn.functional.softmax(utilities, dim=0).detach().numpy()
            item_index.append(np.random.choice(num_items, p=p))
            # item_index.append(int(np.argmax(utilities)))
        else:
            item_index.append(int(np.random.choice(num_items, size=1)))
    item_index = torch.LongTensor(item_index)