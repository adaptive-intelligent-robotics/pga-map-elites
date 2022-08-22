import copy
from math import floor
from multiprocessing import Process, Queue

import torch


class PGVariation:
    def __init__(self, num_cpu, lr=1e-3, nr_of_steps_act=10):
        """
        Initialise the PG variation.
        Input:
            - num_cpu {int} - number of cpu for parallel Actor training.
            - lr {float} - Actor learning rate
            - nr_of_steps_act {int} - Number of steps for Actor training
        Output: /
        """

        self.label = "pg_" + str(lr) + "_" + str(nr_of_steps_act)
        self.n_processes = num_cpu
        self.lr = lr
        self.nr_of_steps_act = nr_of_steps_act

        # Setup queues
        self.actors_train_in_queue = Queue()
        self.actors_train_out_queue = Queue()

        # Setup paralell processes
        self.processes = [
            Process(
                target=pgvariation_worker,
                args=(
                    process_id,
                    self.actors_train_in_queue,
                    self.actors_train_out_queue,
                ),
            )
            for process_id in range(self.n_processes)
        ]

        # Start paralell processes
        for p in self.processes:
            p.daemon = True
            p.start()

    def update_lr(self, lr_prop=1.0):
        self.lr = self.lr * lr_prop

    def update_nr_of_steps_act(self, nr_of_steps_act_prop=1.0):
        self.nr_of_steps_act = floor(self.nr_of_steps_act * nr_of_steps_act_prop)

    def close(self):
        """Close parallel processes."""
        for p in self.processes:
            p.terminate()

    def __call__(self, parent_controllers, critic=False, states=False):
        """
        Evolve a batch of individuals in parallel.
        Input:
            - parent_controllers {list} - individuals to evolve
            - critic {Critic} - critic used for pg variations
            - states {list} - list of states for learning
        Output: offspring_controllers {list} - list of offspring evolved controllers
        """
        n_parents = len(parent_controllers)
        offspring_controllers = [None] * n_parents
        for n in range(n_parents):
            self.actors_train_in_queue.put(
                (
                    n,
                    copy.deepcopy(parent_controllers[n].x),
                    copy.deepcopy(critic),
                    copy.deepcopy(states.detach()),
                    self.lr,
                    self.nr_of_steps_act,
                )
            )

        for _ in range(n_parents):
            n, actor_z = self.actors_train_out_queue.get()
            offspring_controllers[n] = actor_z

        return offspring_controllers


def pgvariation_worker(
    process_id,
    actors_train_in_queue,
    actors_train_out_queue,
):

    """
    Function that runs the parallel processes for the variation operator.
    Input:
        - process_id {int} - ID of the process so it can be identified
        - actors_train_in_queue {Queue object} - queue for incoming actors
        - actors_train_out_queue {Queue object} - queue for outgoing actors
    Output: /
    """

    # Start process loop
    while True:
        try:
            # get an id, a controller to evolve, the critic,
            # states from the replay_buffer and a nb of steps
            (
                n,
                actor_x,
                critic,
                states,
                lr,
                nr_of_steps_act,
            ) = actors_train_in_queue.get()

            # prepare the new actor
            actor_z = copy.deepcopy(actor_x)
            actor_z.type = "grad"
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = None

            # Enable grad
            for param in actor_z.parameters():
                param.requires_grad = True

            # this means we have to define an optimizer each time
            # i guess we're loosing some benefits of adam optimizer, no?
            optimizer = torch.optim.Adam(actor_z.parameters(), lr=lr)

            # gradient descent loop
            for i in range(nr_of_steps_act):
                # get a batch of states
                state = states[i]

                # compute loss
                actor_loss = -critic.Q1(state, actor_z(state)).mean()

                # update the controller
                optimizer.zero_grad()
                actor_loss.backward()
                optimizer.step()

            # Disable grad so can sent across proceeses
            for param in actor_z.parameters():
                param.requires_grad = False

            # put the actor back in the queue
            actors_train_out_queue.put((n, copy.deepcopy(actor_z)))

        except KeyboardInterrupt:
            break
