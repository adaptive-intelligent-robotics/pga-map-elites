'''
Copyright (c) 2020 Scott Fujimoto 
ReplayBuffer Based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implementation by Scott Fujimoto https://github.com/sfujim/TD3 Paper: https://arxiv.org/abs/1802.09477
Modified by:
    Olle Nilsson: olle.nilsson19@imperial.ac.uk
    Felix Chalumeau: felix.chalumeau20@imperial.ac.uk
    Manon Flageat: manon.flageat18@imperial.ac.uk
'''



import numpy as np
import torch
import pickle




class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, bd_dim=2, max_size=int(1e6), load=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.additions = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.behaviour_descriptor = np.zeros((max_size, bd_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, transitions):
        """
        Add all the transitions given to the replay buffer. 

        Determine the indexes of the transitions thanks to the actual position
        of the pointer and the maximum size of the buffer. Then, simply add the 
        diverse elements of the transitions into the replay buffer.

        transition = state, action, next_state, reward, done, bd
        transitions = states, actions, next_states, rewards, dones, bds

        TODO: handle the behaviour descriptor - DONE
        """
        # get number of transitions
        l = len(transitions[0])

        # compute indexes
        idx = np.arange(self.ptr, self.ptr + l) % self.max_size

        # simply add the elements to the storage
        self.state[idx] = transitions[0]
        self.action[idx] = transitions[1]
        self.next_state[idx] = transitions[2]
        self.reward[idx] = transitions[3]
        self.not_done[idx] = 1. - transitions[4]
        self.behaviour_descriptor[idx] = transitions[5]

        # update the metrics
        self.ptr = (self.ptr + l) % self.max_size
        self.size = min(self.size + l, self.max_size)
        self.additions += 1

        # end of the function


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def sample_state(self, batch_size, steps):
        states = []
        for _ in range(steps):
            ind = np.random.randint(0, self.size, size=batch_size)
            # get states
            states.append(self.state[ind])

        return torch.FloatTensor(states).to(self.device)


    def save(self, filename):
        with open(f"{filename}", 'wb') as replay_buffer_file:
            pickle.dump(self, replay_buffer_file)

    
    def load(self, filename):
        with open(f"{filename}", 'rb') as replay_buffer_file:
            replay_buffer = pickle.load(replay_buffer_file)
        return replay_buffer
