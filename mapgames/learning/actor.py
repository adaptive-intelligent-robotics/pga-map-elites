"""
Copyright (c) 2020 Scott Fujimoto
Based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implementation by Scott Fujimoto https://github.com/sfujim/TD3
Paper: https://arxiv.org/abs/1802.09477
Modified by:
    Olle Nilsson: olle.nilsson19@imperial.ac.uk
    Felix Chalumeau: felix.chalumeau20@imperial.ac.uk
    Manon Flageat: manon.flageat18@imperial.ac.uk
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        neurons_list,
        normalise=False,
        affine=False,
        init=False,
    ):

        super(Actor, self).__init__()
        self.weight_init_fn = nn.init.xavier_uniform_
        self.num_layers = len(neurons_list)
        self.normalise = normalise
        self.affine = affine

        if self.num_layers == 1:
            self.l1 = nn.Linear(state_dim, neurons_list[0], bias=(not self.affine))
            self.l2 = nn.Linear(neurons_list[0], action_dim, bias=(not self.affine))

            if self.normalise:
                self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=self.affine)
                self.n2 = nn.LayerNorm(action_dim, elementwise_affine=self.affine)

        if self.num_layers == 2:
            self.l1 = nn.Linear(state_dim, neurons_list[0], bias=(not self.affine))
            self.l2 = nn.Linear(
                neurons_list[0], neurons_list[1], bias=(not self.affine)
            )
            self.l3 = nn.Linear(neurons_list[1], action_dim, bias=(not self.affine))

            if self.normalise:
                self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=self.affine)
                self.n2 = nn.LayerNorm(neurons_list[1], elementwise_affine=self.affine)
                self.n3 = nn.LayerNorm(action_dim, elementwise_affine=self.affine)

        self.max_action = max_action

        for param in self.parameters():
            param.requires_grad = False

        self.apply(self.init_weights)

        self.type = None
        self.id = None
        self.parent_1_id = None
        self.parent_2_id = None

        self.novel = False  # considered not novel until proven

        # Compare to previous elite in the niche
        self.delta_f = 0  # set to zero until the niche is filled

        # Compare to parent
        self.parent_fitness = None
        self.parent_bd = None
        self.delta_bd = 0  # zero until measured
        self.parent_delta_f = 0

        self.behaviour_descriptor = None
        self.variation_type = None

    def forward(self, state):
        if self.num_layers == 1:
            if self.normalise:
                a = func.relu(self.n1(self.l1(state)))
                return self.max_action * torch.tanh(self.n2(self.l2(a)))
            else:
                a = func.relu(self.l1(state))
                return self.max_action * torch.tanh(self.l2(a))
        if self.num_layers == 2:
            if self.normalise:
                a = func.relu(self.n1(self.l1(state)))
                a = func.relu(self.n2(self.l2(a)))
                return self.max_action * torch.tanh(self.n3(self.l3(a)))

            else:
                a = func.relu(self.l1(state))
                a = func.relu(self.l2(a))
                return self.max_action * torch.tanh(self.l3(a))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self(state).cpu().data.numpy().flatten()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device("cpu")))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.weight_init_fn(m.weight)
        if isinstance(m, nn.LayerNorm):
            pass

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def return_copy(self):
        return copy.deepcopy(self)
