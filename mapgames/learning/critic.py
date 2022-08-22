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


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = func.relu(self.l1(sa))
        q1 = func.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = func.relu(self.l4(sa))
        q2 = func.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = func.relu(self.l1(sa))
        q1 = func.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class Critic(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        expl_noise=0.2,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4,
    ):

        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.actor = None
        self.lr = lr

    def has_greedy(self):
        """Is greedy actor defined."""
        return self.actor is not None

    def update_greedy(self, new_actor):
        """
        Update current greedy actor.
        Input: new_actor {Actor} - candidate new actor
        Output: /
        """
        if new_actor != self.actor:
            a = copy.deepcopy(new_actor)
            for param in a.parameters():
                param.requires_grad = True
            a.parent_1_id = new_actor.id
            a.parent_2_id = None
            a.type = "critic_training"

            self.actor = a
            self.actor_target = copy.deepcopy(a)
            self.actor_optimiser = torch.optim.Adam(a.parameters(), lr=self.lr)

    def train(self, replay_buffer, nr_of_steps_crit, train_batch_size=256):
        """
        Main training function, used to train the critic and its actor(s).
        Input:
                - replay_buffer {ReplayBuffer} - used to sample transitions
                to train the critic
                - nr_of_steps_crit {int} - number of steps to train the critic
                - train_batch_size {int} - size of the batch sampled for each
                training step of the critic
        Output: critic_loss {float} - last critic loss
        """

        for _ in range(nr_of_steps_crit):
            self.total_it += 1

            # Sample replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(
                train_batch_size
            )

            # Conpute Q target
            with torch.no_grad():

                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )

                next_action = (self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = func.mse_loss(current_Q1, target_Q) + func.mse_loss(
                current_Q2, target_Q
            )

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimiser.zero_grad()
                actor_loss.backward()
                self.actor_optimiser.step()

                # Update the target models
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                # Update the target models
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        return critic_loss

    def save(self, filename):
        """Save the critic networks and optimizers"""
        torch.save(self.critic.state_dict(), filename)
        torch.save(self.critic_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        """Load the critic networks and optimizers"""
        self.critic.load_state_dict(
            torch.load(filename + "_critic", map_location=torch.device("cpu"))
        )
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=torch.device("cpu"))
        )
        self.critic_target = copy.deepcopy(self.critic)
