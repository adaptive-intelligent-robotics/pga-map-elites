"""
Copyright 2019, INRIA
SBX, iso_dd and polynomial mutation operators based on pymap_elites
https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by:
    Olle Nilsson: olle.nilsson19@imperial.ac.uk
    Felix Chalumeau: felix.chalumeau20@imperial.ac.uk
    Manon Flageat: manon.flageat18@imperial.ac.uk
"""

import copy


class BasicGAVariation:
    """
    A basic GA variation is made of one or zero mutation and one or zero crossover.
    """

    def __init__(self, selection_op, mutation=False, crossover=False):
        """
        Initialise the GA variation.
        Input:
            - selection_op {ArchiveSelector} - selector used for crossover operations
            - mutation {Mutation} - mutation operator (set to False if only crossover)
            - crossover {Crossover} - crossover operator (set to False if only mutation)
        Output: /
        """

        if not mutation:
            self.label = crossover.label
        if not crossover:
            self.label = mutation.label
        if mutation and crossover:
            self.label = mutation.label + "_" + crossover.label

        self.mutation = mutation
        self.crossover = crossover
        self.selection_op = selection_op

    def __call__(self, parent_controllers, archive):
        """
        Evolve a batch of individuals.
        Input:
            - parent_controllers {list} - individuals to evolve
            - archive {dict} - archive used to help variation
        Output: offspring_controllers {list} - evolved individuals
        """

        offspring_controllers = []
        for controller in parent_controllers:

            # Apply variation
            if self.crossover:
                parent_y = self.selection_op(archive, 1)[0]
                offspring_controllers.append(
                    self.evolve_individual(controller.x, parent_y.x)
                )
            else:
                offspring_controllers.append(self.evolve_individual(controller.x))

        return offspring_controllers

    def evolve_individual(self, actor_x, actor_y=False):
        """
        Evolve one individual.
        Input:
            - actor_x {Actor} - individual to evolve
            - actor_y {Actor} - facultative individual used for evolution
        Output: actor_z {Actor} - evolved individual
        """

        # Create actor_z
        actor_z = copy.deepcopy(actor_x)
        actor_z.optimizer = None
        actor_z.type = "evo"

        # If crossover type
        if self.crossover:
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = actor_y.id
            actor_z_state_dict = self.crossover.apply_to_state_dict(
                actor_x.state_dict(), actor_y.state_dict()
            )
            if self.mutation:
                actor_z_state_dict = self.mutation.apply_to_state_dict(
                    actor_z_state_dict
                )

        # Elif mutation type
        elif self.mutation:
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = None
            actor_z_state_dict = self.mutation.apply_to_state_dict(actor_x.state_dict())

        actor_z.load_state_dict(actor_z_state_dict)
        return actor_z

    def update_mutation_rate(self, mutation_rate_prop):
        if self.mutation is not False:
            self.mutation.update_mutation_rate(mutation_rate_prop)
            return True
        return False
