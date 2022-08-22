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
        self.id = next(self._ids)  # get a unique id for each individual
        Individual.current_id = self.id
        x.id = self.id
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid

    @classmethod
    def get_counter(cls):
        """Get the individual unique id"""
        return cls._ids
