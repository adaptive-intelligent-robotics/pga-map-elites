'''
Copyright 2019, INRIA
SBX and ido_dd and polynomilal mutauion variation operators based on pymap_elites framework
https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by:
    Olle Nilsson: olle.nilsson19@imperial.ac.uk
    Felix Chalumeau: felix.chalumeau20@imperial.ac.uk
    Manon Flageat: manon.flageat18@imperial.ac.uk
'''

import copy
import numpy as np
import torch

class Crossover():
    """ Base Crossover class """

    def __init__(self, min_gene, max_gene):
        self.min = min_gene
        self.max = max_gene

    def apply_to_state_dict(self, controller_x_state_dict, controller_y_state_dict):
        z = copy.deepcopy(controller_x_state_dict)
        for tensor in controller_x_state_dict:
            if "weight" or "bias" in tensor:
                z[tensor] = self.apply(controller_x_state_dict[tensor], controller_y_state_dict[tensor])
        return z


class IsoDDCrossover(Crossover):
    """ IsoDD Crossover class """

    def __init__(self, min_gene, max_gene, iso_sigma, line_sigma):
        super().__init__(min_gene, max_gene)

        self.label = "lineDD_" + str(iso_sigma) + "_" + str(line_sigma)

        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma

    def apply(self, x, y):
        '''
        Iso+Line
        Ref:
        Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
        GECCO 2018
        '''
        a = torch.zeros_like(x).normal_(mean=0, std=self.iso_sigma)
        b = np.random.normal(0, self.line_sigma)
        z = x.clone() + a + b * (y - x)

        if not self.max and not self.min:
            return z
        else:
            return torch.clamp(z, self.min, self.max)


class SBXCrossover(Crossover):
    """ SBX Crossover class """

    def __init__(self, min_gene, max_gene, crossover_rate, eta_c):
        super().__init__(min_gene, max_gene)

        self.label = "sbx_" + str(crossover_rate) + "_" + str(eta_c)

        self.crossover_rate = crossover_rate
        self.eta_c = eta_c

    def apply(self, x, y):
        if not self.max and not self.min:
            return self.__sbx_unbounded(x, y)
        else:
            return self.__sbx_bounded(x, y)

    def __sbx_unbounded(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover
        Unbounded version
        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        z = x.clone()
        c = torch.rand_like(z)
        index = torch.where(c < self.crossover_rate)
        r1 = torch.rand(index[0].shape)
        r2 = torch.rand(index[0].shape)

        if len(z.shape) == 1:
            diff = torch.abs(x[index[0]] - y[index[0]])
            x1 = torch.min(x[index[0]], y[index[0]])
            x2 = torch.max(x[index[0]], y[index[0]])
            z_idx = z[index[0]]
        else:
            diff = torch.abs(x[index[0], index[1]] - y[index[0], index[1]])
            x1 = torch.min(x[index[0], index[1]], y[index[0], index[1]])
            x2 = torch.max(x[index[0], index[1]], y[index[0], index[1]])
            z_idx = z[index[0], index[1]]

        beta_q = torch.where(r1 <= 0.5, (2.0 * r1) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 * (1.0 - r1))) ** (1.0 / (self.eta_c + 1)))

        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

        z_mut = torch.where(diff > 1e-15, torch.where(r2 <= 0.5, c2, c1), z_idx)

        if len(y.shape) == 1:
            z[index[0]] = z_mut
        else:
            z[index[0], index[1]] = z_mut
        return z


    def __sbx_bounded(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover
        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        z = x.clone()
        c = torch.rand_like(z)
        index = torch.where(c < self.crossover_rate)
        r1 = torch.rand(index[0].shape)
        r2 = torch.rand(index[0].shape)

        if len(z.shape) == 1:
            diff = torch.abs(x[index[0]] - y[index[0]])
            x1 = torch.min(x[index[0]], y[index[0]])
            x2 = torch.max(x[index[0]], y[index[0]])
            z_idx = z[index[0]]
        else:
            diff = torch.abs(x[index[0], index[1]] - y[index[0], index[1]])
            x1 = torch.min(x[index[0], index[1]], y[index[0], index[1]])
            x2 = torch.max(x[index[0], index[1]], y[index[0], index[1]])
            z_idx = z[index[0], index[1]]


        beta = 1.0 + (2.0 * (x1 - self.min) / (x2 - x1))
        alpha = 2.0 - beta ** - (self.eta_c + 1)
        beta_q = torch.where(r1 <= (1.0 / alpha), (r1 * alpha) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 - r1 * alpha)) ** (1.0 / (self.eta_c + 1)))

        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

        beta = 1.0 + (2.0 * (self.max - x2) / (x2 - x1))
        alpha = 2.0 - beta ** - (self.eta_c + 1)

        beta_q = torch.where(r1 <= (1.0 / alpha), (r1 * alpha) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 - r1 * alpha)) ** (1.0 / (self.eta_c + 1)))
        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

        c1 = torch.clamp(c1, self.min, self.max)
        c2 = torch.clamp(c2, self.min, self.max)

        z_mut = torch.where(diff > 1e-15, torch.where(r2 <= 0.5, c2, c1), z_idx)

        if len(y.shape) == 1:
            z[index[0]] = z_mut
        else:
            z[index[0], index[1]] = z_mut
        return z
