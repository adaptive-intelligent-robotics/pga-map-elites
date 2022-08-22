"""
Copyright 2019, INRIA
SBX and ido_dd and polynomial mutation operators based on pymap_elites
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

import torch


class Mutation:
    """Base Mutation class"""

    def __init__(self, min_gene, max_gene, mutation_rate):
        self.min = min_gene
        self.max = max_gene
        self.mutation_rate = mutation_rate

    def apply_to_state_dict(self, controller_state_dict):
        z = copy.deepcopy(controller_state_dict)
        for tensor in controller_state_dict:
            if "weight" or "bias" in tensor:
                z[tensor] = self.apply(controller_state_dict[tensor])
        return z


class PolynomialMutation(Mutation):
    """Polynomial Mutation"""

    def __init__(self, min_gene, max_gene, mutation_rate, eta_m):
        super().__init__(min_gene, max_gene, mutation_rate)

        self.label = "poly_" + str(mutation_rate) + "_" + str(eta_m)

        self.eta_m = eta_m

    def apply(self, x):
        """
        Cf Deb 2001, p 124 ; param: eta_m
        """
        y = x.clone()
        m = torch.rand_like(y)
        index = torch.where(m < self.mutation_rate)
        r = torch.rand(index[0].shape)
        delta = torch.where(
            r < 0.5,
            (2 * r) ** (1.0 / (self.eta_m + 1.0)) - 1.0,
            1.0 - ((2.0 * (1.0 - r)) ** (1.0 / (self.eta_m + 1.0))),
        )
        if len(y.shape) == 1:
            y[index[0]] += delta
        else:
            y[index[0], index[1]] += delta

        if not self.max and not self.min:
            return y
        else:
            return torch.clamp(y, self.min, self.max)


class GaussianMutation(Mutation):
    """Gaussian Mutation"""

    def __init__(self, min_gene, max_gene, mutation_rate, sigma):
        super().__init__(min_gene, max_gene, mutation_rate)

        self.label = "gaussian_" + str(mutation_rate) + "_" + str(sigma)

        self.sigma = sigma

    def apply(self, x):
        y = x.clone()
        m = torch.rand_like(y)
        index = torch.where(m < self.mutation_rate)
        delta = torch.zeros(index[0].shape).normal_(mean=0, std=self.sigma)
        if len(y.shape) == 1:
            y[index[0]] += delta
        else:
            y[index[0], index[1]] += delta

        if not self.max and not self.min:
            return y
        else:
            return torch.clamp(y, self.min, self.max)


class UniformMutation(Mutation):
    """Uniform Mutation"""

    def __init__(self, min_gene, max_gene, mutation_rate, max_uniform):
        super().__init__(min_gene, max_gene, mutation_rate)

        self.label = "uniform_" + str(mutation_rate)

        self.max_uniform = max_uniform

    def apply(self, x):
        y = x.clone()
        m = torch.rand_like(y)
        index = torch.where(m < self.mutation_rate)
        delta = torch.zeros(index[0].shape).uniform_(
            -self.max_uniform, self.max_uniform
        )
        if len(y.shape) == 1:
            y[index[0]] += delta
        else:
            y[index[0], index[1]] += delta

        if not self.max and not self.min:
            return y
        else:
            return torch.clamp(y, self.min, self.max)
