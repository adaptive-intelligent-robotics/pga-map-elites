from mapgames.variation.ga_crossover import IsoDDCrossover, SBXCrossover
from mapgames.variation.ga_mutation import (
    GaussianMutation,
    PolynomialMutation,
    UniformMutation,
)
from mapgames.variation.ga_operator import BasicGAVariation
from mapgames.variation.pg_operator import PGVariation


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
        stop_pg=10000000,
        proportion_update=1.0,
        proportion_update_period=10000000,
        lr_update=1.0,
        lr_update_period=10000000,
        nr_of_steps_act_update=1.0,
        nr_of_steps_act_update_period=10000000,
        mutation_rate_update=1.0,
        mutation_rate_update_period=10000000,
    ):
        """
        Initialise the scheduler.
        Input:
            - selection_op {ArchiveSelector} - selector used for crossover
            - proportion_evo {float} - proportion of pg and ga variations
            - stop_pg {int} - nr evaluations to stop pg variation (GA/PG)
            - proportion_update {float} - proportion_evo decay (GA/PG)
            - proportion_update_period {float} - proportion_evo decay
                period (GA/PG)
            - lr_update {float} - lr decay (PG)
            - lr_update_period {float} - lr decay period (PG)
            - nr_of_steps_act_update {float} - nr_of_steps_act decay (PG)
            - nr_of_steps_act_update_period {float} - nr_of_steps_act decay
                period (PG)
            - mutation_rate_update {float} - mutation_rate decay (GA)
            - mutation_rate_update_period {float} - mutation_rate decay
                period (GA)
        Output: /
        """
        self.ga_variation = None
        self.pg_variation = None

        self.selection_op = selection_op
        self.proportion_evo = proportion_evo

        # Proportion evo update
        self.stop_pg = stop_pg
        self.proportion_update = proportion_update
        self.proportion_update_period = proportion_update_period
        self.proportion_update_evals = 0

        # PG update
        self.lr_update = lr_update
        self.lr_update_period = lr_update_period
        self.lr_update_evals = 0
        self.nr_of_steps_act_update = nr_of_steps_act_update
        self.nr_of_steps_act_update_period = nr_of_steps_act_update_period
        self.nr_of_steps_act_update_evals = 0

        # GA update
        self.mutation_rate_update = mutation_rate_update
        self.mutation_rate_update_period = mutation_rate_update_period
        self.mutation_rate_update_evals = 0

    def initialise_variations(
        self,
        crossover_op="iso_dd",
        mutation_op=None,
        pg_op=False,
        num_cpu=False,
        lr=3e-4,
        nr_of_steps_act=50,
        max_gene=False,
        min_gene=False,
        mutation_rate=0.05,
        crossover_rate=0.75,
        eta_m=5.0,
        eta_c=10.0,
        sigma=0.1,
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
            - max_gene {float} - max value for the genotype (GA)
            - min_gene {float} - min value for the genotype (GA)
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
        Update the mutations.
        Input: n_evals {int} - curent number of eval
        Output: {bool} - true if the mutations have been modified
        """
        print("\nEntering variation update")

        modified = False

        # First update proportion evo
        if n_evals >= self.stop_pg and self.proportion_evo != 1.0:
            self.proportion_evo = 1.0
            modified = True
            print(" -> Stoped PG")
        elif (
            n_evals < self.stop_pg
            and self.proportion_update != 1
            and (n_evals - self.proportion_update_evals)
            >= self.proportion_update_period
        ):
            self.proportion_evo = self.proportion_evo * self.proportion_update
            self.proportion_update_evals = n_evals
            modified = True
            print(" -> Reducing proportion of PG:", self.proportion_evo)

        # Second update the PG variation
        if self.pg_variation is not None:
            if (
                self.lr_update != 1
                and (n_evals - self.lr_update_evals) >= self.lr_update_period
            ):
                self.pg_variation.update_lr(self.lr_update)
                self.lr_update_evals = n_evals
                modified = True
                print(" -> Reducing lr:", self.pg_variation.lr)
            if (
                self.nr_of_steps_act_update != 1
                and (n_evals - self.nr_of_steps_act_update_evals)
                >= self.nr_of_steps_act_update_period
            ):
                self.pg_variation.update_nr_of_steps_act(self.nr_of_steps_act_update)
                self.nr_of_steps_act_update_evals = n_evals
                modified = True
                print(
                    " -> Reducing nr_of_steps_act:", self.pg_variation.nr_of_steps_act
                )

        # Second update the GA variation
        if self.ga_variation is not None:
            if (
                self.mutation_rate_update != 1
                and (n_evals - self.mutation_rate_update_evals)
                >= self.mutation_rate_update_period
            ):
                modified_ga = self.ga_variation.update_mutation_rate(
                    self.mutation_rate_update
                )
                self.mutation_rate_update_evals = n_evals
                if modified_ga:
                    modified = True
                    print(
                        " -> Reducing mutation_rate:",
                        self.ga_variation.mutation.mutation_rate,
                    )

        print("Exiting variation update\n")
        return modified

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
