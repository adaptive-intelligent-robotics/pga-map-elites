# Class defining the lower-level instance: the individuals
class Individual:
    def __init__(self, x, desc, fitness, centroid=None):
        """
        Initialise the individual
        Input:
          - x {Actor} - controller
          - desc {list}
          - fitness {float}
          - centroid {int} - id of corresponding centroid
        Output: /
        """
        x.id = next(self._ids)  # get a unique id for each individual
        x.behaviour_descriptor = desc
        Individual.current_id = x.id
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid

        # Compute parent-offspring distance
        self.parent_delta_f = 0  # for simplicity
        if x.parent_fitness is not None:
            self.parent_delta_f = fitness - x.parent_fitness
        self.parent_delta_bd = 0  # for simplicity
        if x.parent_bd is not None:
            self.parent_delta_bd = sum([abs(x) for x in (desc - x.parent_bd)])

    @classmethod
    def get_counter(cls):
        """Get the individual unique id"""
        return cls._ids
