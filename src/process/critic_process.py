# external imports
import copy
from multiprocessing import Event, Pipe, Process


class CriticProcess:
    """Critics training process."""

    def __init__(self, worker_fn):
        # connexion Pipe
        self.remote, self.local = Pipe()

        # events
        self.global_sync = Event()
        self.close_process = Event()

        # create the critic process (and hence the critic)
        self.critic_process = Process(
            target=worker_fn, args=(self.remote, self.global_sync, self.close_process)
        )

        self.critic_process.daemon = True

        # start the critic process
        self.critic_process.start()

    def update_greedy(self, new_actor):
        """
        Update the critics greedy actor.
        Inputs: new_actor {Actor} - candidate new greedy actor
        Outputs: /
        """
        self.local.send(copy.deepcopy(new_actor))

    def get_critic(self):
        """
        Get critic, along with other information, from the Remote connexion.
        Inputs: /
        Outputs:
            - critic {Critic}
            - actor {Actor} - greedy actor
            - states {list} - list of states to use for the pg variation
            - time {float} - train time
        """
        # set event flag to True
        self.global_sync.set()
        # print("Global sync flag set to True")
        # receive data from Remote - prepared in critic_worker worker function
        critic, actor, states, critic_loss, time = self.local.recv()
        print(f"  Critic and greedy total training time (s): {time}")
        return critic, actor, states, time

    def close(self):
        """
        Close the critic process.
        Inputs: /
        Outputs:
            - critic {Critic}
            - replay_buffer {ReplayBuffer}
        """

        # set the close_process flag to True
        self.close_process.set()

        # terminate the critic process
        self.critic_process.terminate()

