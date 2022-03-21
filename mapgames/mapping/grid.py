'''
Copyright 2019, INRIA
CVT Uility functions based on pymap_elites framework https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by:
    Olle Nilsson: olle.nilsson19@imperial.ac.uk
    Felix Chalumeau: felix.chalumeau20@imperial.ac.uk
    Manon Flageat: manon.flageat18@imperial.ac.uk
'''

import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from itertools import count
from sklearn.neighbors import KDTree

######### Main functions 

def add_to_archive(s, descriptor, archive, kdt, cell_fn, reeval = False):
    """
    Add individual to vanilla MAP-Elites archive
    Input:
    	- s {Individual} - individual to add
	- descriptor {np.array} - individual's descriptor
	- archive {dict} - archive to add to
	- kdt {KDTree} - used to add to archive
	- reeval {bool} - 0 when adding to reeval archives, to not modify controllers 
    Output: added {bool} - 1 if added, 0 if not
    """
    niche_index = kdt.query([descriptor], k=1)[1][0][0]
    niche = kdt.data[niche_index]

    # usable id of the individual's niche
    n = make_hashable(niche)
    s.centroid = n # add the centroid id info to the individual instance
    if not reeval: s.x.novel = False

    # if there is already an individual in this niche 
    if n in archive:
        # Try to add to cell
        if archive[n].can_add(s):
            if not reeval: s.x.delta_f = s.fitness - archive[n].fitness
            added = archive[n].add(s)
            assert added == 1, "Mismatch between can_add and add in Cell class"
            return 1
        else: 
            return 0

    # if it is the first time we encounter an individual in this niche 
    else:
        # Create the cell
        archive[n] = cell_fn()
        if archive[n].can_add(s):
            if not reeval:
                s.x.novel = True
                s.x.delta_f = s.fitness # maybe we should beware the cases where the fitness can be negative
            added = archive[n].add(s)
            assert added == 1, "Mismatch between can_add and add in Cell class"
            return 1
        else: 
            return 0


######### CVT-related functions

def make_hashable(array):
    return tuple(map(float, array))

def __centroids_filename(k, dim): 
    return 'CVT/centroids_' + str(k) + '_' + str(dim) + '.dat'

def write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ') 
            f.write('\n')


def cvt(k, dim, samples, cvt_use_cache = True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("\n!!!WARNING!!! using cached CVT:", fname, "\n")
            if dim == 1:
                if k == 1:
                    return np.expand_dims(np.expand_dims(np.loadtxt(fname), axis=0), axis=1)
                return np.expand_dims(np.loadtxt(fname), axis=1)
            else:
                if k == 1:
                    return np.expand_dims(np.loadtxt(fname), axis=0)
                return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)
    x = np.random.rand(samples, dim) 
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, max_iter=1000000, n_jobs=1, verbose=1, tol=1e-8) #Full is the proper Expectation Maximization algorithm
    k_means.fit(x)
    write_centroids(k_means.cluster_centers_)
    return k_means.cluster_centers_

