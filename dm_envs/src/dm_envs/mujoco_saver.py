import pickle


def save(env, action):
    state = env._physics.data
    data = {
        'action': action,
        'state':  state,
    }
    with open("data.pkl", 'wb') as f:
        pickle.dump(data, f)
