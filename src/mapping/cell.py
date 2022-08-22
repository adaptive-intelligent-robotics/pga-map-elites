from random import randint, uniform


# MAP-Elites cells
class Cell:
    def __init__(self, max_depth):
        self._content = []
        self.max_depth = max_depth
        self._update_attributes()

    def can_add(self, s):
        """Return 1 if individual s will be added to cell in add()
        Return 0 otherwise"""
        return (len(self._content) < self.max_depth) or (
            s.fitness > self._content[self.max_depth - 1].fitness
        )

    def add(self, s):
        """Add solution s to cell"""

        # If cell not full, add ordered by fitness
        if len(self._content) < self.max_depth:
            self._content.append(s)
            if len(self._content) > 1:
                self._ordered(s)
            self._update_attributes()
            return 1

        # If cell full and worste than worste indiv, do not add
        if s.fitness < self._content[self.max_depth - 1].fitness:
            return 0

        # If cell full and better than worste indiv, add
        self._ordered(s)
        self._update_attributes()
        return 1

    def select(self):
        """Select a solution s from cell"""
        return self._content[0]

    def _update_attributes(self):
        """Update attributes of the cell for interface with the object"""
        if len(self._content) == 0:
            self.x = None
            self.desc = None
            self.fitness = None
            self.centroid = None
        else:  # Used when computing metrics and saving archive
            self.x = self._content[0].x
            self.desc = self._content[0].desc
            self.fitness = self._content[0].fitness
            self.centroid = self._content[0].centroid

    def _ordered(self, s):
        """Add indiv to the content ordered by fitness."""
        i = len(self._content) - 1
        while i > 0 and s.fitness > self._content[i - 1].fitness:
            self._content[i] = self._content[i - 1]
            i -= 1
        self._content[i] = s


# Deep-grid cells
class DeepGridCell:
    def __init__(self, max_depth):
        self._content = []
        self.max_depth = max_depth
        self._update_attributes()

    def can_add(self, s):
        """Return 1 if individual s will be added to cell in add()
        Return 0 otherwise"""
        return 1

    def add(self, s):
        """Add solution s to cell"""
        # If cell not full, add at the end
        if len(self._content) < self.max_depth:
            n = len(self._content)
            self._content.append(s)

        # If cell full, replace randomly
        else:
            n = randint(0, self.max_depth - 1)
            self._content[n] = s

        if len(self._content) > 1:
            self._ordered(n, s)
        self._update_attributes()
        return 1

    def select(self):
        """Select a solution s from cell"""
        assert len(self._content) > 0, "Selecting from an empty cell"
        if len(self._content) == 1:
            return self._content[0]

        # Normalised fitnesses of the cell
        norm_fitness = [
            (s.fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
            for s in self._content
        ]

        # Choose indidivual fitness-proportionally
        rand = uniform(0, sum(norm_fitness))
        i = 0
        value = norm_fitness[i]
        while value < rand and i < (len(self._content) - 1):
            i += 1
            value += norm_fitness[i]
        return self._content[i]

    def _update_attributes(self):
        """Update attributes of the cell for interface with the object"""
        if len(self._content) == 0:
            self.x = None
            self.desc = None
            self.fitness = None
            self.min_fitness = None
            self.max_fitness = None
            self.centroid = None
        else:  # Used when computing metrics and saving archive
            self.x = self._content[0].x
            self.desc = self._content[0].desc
            self.fitness = self._content[0].fitness
            self.centroid = self._content[0].centroid
            self.min_fitness = self._content[-1].fitness
            self.max_fitness = self._content[0].fitness

    def _ordered(self, n, s):
        """Add indiv to the content ordered by fitness."""
        i = n
        while i < len(self._content) - 1 and s.fitness < self._content[i + 1].fitness:
            self._content[i] = self._content[i + 1]
            i += 1
        while i > 0 and s.fitness > self._content[i - 1].fitness:
            self._content[i] = self._content[i - 1]
            i -= 1
        self._content[i] = s
