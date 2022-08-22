from src.variation.ga_crossover import IsoDDCrossover, SBXCrossover
from src.variation.ga_mutation import (
    GaussianMutation,
    PolynomialMutation,
    UniformMutation,
)
from src.variation.ga_operator import BasicGAVariation
from src.variation.pg_operator import PGVariation


class VariationScheduler:
    """
    The variation scheduler handle variations during the process,
    through 3 main methods:
        - initialise_variations : use the user args to create the
        variations and store it in the scheduler.
        - update : update the params used to determine the scheduling.
        - evolve : one of the main function of this repo! use the
        variations and the scheduling strategy.
    """

    def __init__(
        self,
        selection_op,
        proportion_evo=0.5,
    ):
        """
        Initialise the scheduler.
        Input:
            - selection_op {ArchiveSelector} - selector used for crossover
            - proportion_evo {float} - proportion of pg and ga variations
        Output: /
        """
        self.ga_variation = None
        self.pg_variation = None

        self.selection_op = selection_op
        self.proportion_evo = proportion_evo

    def initialise_variations(
        self,
        crossover_op="iso_dd",
        mutation_op=None,
        pg_op=False,
        num_cpu=False,
        lr=3e-4,
        nr_of_steps_act=50,
        mutation_rate=0.05,
        crossover_rate=0.75,
        eta_m=5.0,
        eta_c=10.0,
        sigma=0.2,
        max_uniform=0.1,
        iso_sigma=0.005,
        line_sigma=0.05,
    ):
        """
        Initialise all variation operators.
        Input:
            - crossover_op {str} - type of crossover (GA)
            - mutation_op {str} - type of mutation (GA)
            - pg_op {bool} - use pg op (PG)
            - num_cpu {int} - number of cpu (PG)
            - lr {float} - learning rate for actor learning (PG)
            - nr_of_steps_act {int} - number of steps for actor learning (PG)
            - mutation_rate {float} - probability of mutation (GA)
            - crossover_rate {float} - probability of crossover (GA)
            - eta_m {float} - amplitude of polynomial mutation (GA)
            - eta_c {float} - amplitude of sbx crossover (GA)
            - sigma {float} - standard deviation of gaussian mutation (GA)
            - max_uniform {float} - amplitude of uniform mutation (GA)
            - iso_sigma {float} - gaussian parameter in iso dd mutation (GA)
            - line_sigma {float} - line parameter in iso dd mutation (GA)
        Output: variation {list} - list of variation operators
        """

        # GA variation
        min_gene = False 
        max_gene = False
        if crossover_op == "iso_dd":
            crossover_operator = IsoDDCrossover(
                min_gene, max_gene, iso_sigma, line_sigma
            )
        elif crossover_op == "sbx" and crossover_rate != 0:
            crossover_operator = SBXCrossover(min_gene, max_gene, crossover_rate, eta_c)
        else:
            crossover_operator = False

        if mutation_op == "polynomial_mutation" and mutation_rate != 0:
            mutation_operator = PolynomialMutation(
                min_gene, max_gene, mutation_rate, eta_m
            )
        elif mutation_op == "uniform_mutation" and mutation_rate != 0:
            mutation_operator = UniformMutation(
                min_gene, max_gene, mutation_rate, max_uniform
            )
        elif mutation_op == "gaussian_mutation" and mutation_rate != 0:
            mutation_operator = GaussianMutation(
                min_gene, max_gene, mutation_rate, sigma
            )
        else:
            mutation_operator = False

        if mutation_operator is not False or crossover_operator is not False:
            self.ga_variation = BasicGAVariation(
                self.selection_op, mutation_operator, crossover_operator
            )

        # PG variation
        if pg_op:
            self.pg_variation = PGVariation(num_cpu, lr, nr_of_steps_act)
        else:
            print("\n!!!WARNING!!! This algorithm does not use any PG variation.\n")

    def update(self, n_evals):
        """
        Update the mutations. Not used here.
        Input: n_evals {int} - curent number of eval
        Output: {bool} - true if the mutations have been modified
        """
        return False

    def evolve(self, archive, nb_controllers, critic=False, states=False):
        """
        Apply variations.
        Input:
            - archive {dict} - archive to sample individuals to evolve
            - nb_controllers {int} - nr of individuals to evolve
            - critic {Critic} - critic used for pg variations
            - states {list} - list of states to use for the pg variation
        Output:
            - offspring_controllers {list} - list of offspring evolved controllers
            - controllers_variations {list} - list of variations applied to each parents
        """

        # If no variation set, return no parent
        if self.pg_variation is None and self.ga_variation is None:
            return [], []

        # Initialisation
        offspring_controllers = []
        controllers_variations = []

        # Select parent controllers
        parent_controllers = self.selection_op(archive, nb_controllers)

        # Select part of controllers for GA and PG
        nb_controllers_ga = int(self.proportion_evo * nb_controllers)
        nb_controllers_pg = nb_controllers - nb_controllers_ga

        # First, evolve with GA
        if self.ga_variation is not None:
            controllers_to_evolve = parent_controllers[0:nb_controllers_ga]
            offspring_controllers += self.ga_variation(controllers_to_evolve, archive)
            controllers_variations += [self.ga_variation.label] * len(
                controllers_to_evolve
            )
        elif nb_controllers_ga != 0:
            print("\n!!!WARNING!!! No GA variation but n_evo > 0\n")

        # Second, evolve with PG
        if self.pg_variation is not None:
            controllers_to_evolve = parent_controllers[
                nb_controllers_ga : nb_controllers_ga + nb_controllers_pg
            ]
            offspring_controllers += self.pg_variation(
                controllers_to_evolve, critic, states
            )
            controllers_variations += [self.pg_variation.label] * len(
                controllers_to_evolve
            )
        elif nb_controllers_pg != 0:
            print("\n!!!WARNING!!! No PG variation but n_evo < 1\n")

        # Update offspring attributes using parents ones
        for parent, offspring in zip(parent_controllers, offspring_controllers):
            # offspring is an Actor instance - parent an Individual instance
            offspring.parent_fitness = parent.fitness
            offspring.parent_bd = parent.desc

        return offspring_controllers, controllers_variations

    def close(self):
        """Close the variation processes."""
        for variation in [self.ga_variation, self.pg_variation]:
            if hasattr(
                variation, "close"
            ):  # close only processes that need to be closed
                variation.close()
