#cython: language_level=3, boundscheck=False
from libc.time cimport time

import numpy as np
import copy

from cython_gsl cimport gsl_rng, gsl_rng_alloc, gsl_rng_set, gsl_rng_free
from cython_gsl cimport gsl_rng_mt19937, gsl_rng_ranlux, gsl_rng_ranlxs2
from .cnodes cimport RandomVariableNode

cdef class Chain:


    cdef dict vars
    cdef int id
    cdef public dict stats
    cdef list trace_keys


    def __init__(self, model, chain_id):
        
        self.vars = copy.copy(model.vars)
        self.id = chain_id

        self.stats = {}
        self.trace_keys = model.trace_keys

        for key in self.trace_keys:
            self.stats[key] = { 'sum1': 0, 'sum2': 0, 'N': 0 }


    def burn_stats(self, burn_fraction=1.0):
        keep_fraction = 1. - burn_fraction
        for key in self.trace_keys:
            for stat_key, stat_val in self.stats[key].items():
                self.stats[key][stat_key] = stat_val * keep_fraction
        for vardict in self.vars.values():
            for node in vardict.values():
                node.burn_stats(burn_fraction)



    def sample(self, N, run_sampled_count=None, thin=1, quiet=True):

        steps_until_thin = thin

        # this will run in multiprocessing job, so we need to reset seed
        np.random.seed()
                
        updt_interval = max(1, N*0.0001)
        steps_until_updt = updt_interval

        # define the random number generator
        cdef gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2)
        # seed it with current time
        cdef unsigned int seed = np.random.randint(0, 2147483647)
        gsl_rng_set(rng, seed)

        cdef RandomVariableNode node

        for i in range(N):
            steps_until_updt -= 1
            if not steps_until_updt:
                if run_sampled_count is not None:
                    run_sampled_count[self.id] += updt_interval
                elif not quiet:
                    print("\rChain {} - Progress {: 7.2%}".format(self.id, i/N), end="")
                steps_until_updt = updt_interval

            steps_until_thin -= 1
            update_stats = steps_until_thin < 1

            for vardict in self.vars.values():
                for node in vardict.values():
                    node.sample(rng, update_stats)

            if update_stats:
                steps_until_thin = thin
        
        gsl_rng_free(rng)
        
        for vardict in self.vars.values():
            for node in vardict.values():
                try:
                    # if node is multinomial, value will be a numpy array
                    # have to set a list for each element in 'value'
                    for i, val in enumerate(node.value):
                        self.stats[f"{node.id}_{i}"]['sum1'] = node.valsum1[i]
                        self.stats[f"{node.id}_{i}"]['sum2'] = node.valsum2[i]
                        self.stats[f"{node.id}_{i}"]['N'] = node.valN
                except TypeError:
                    # value is no array, it won't be iterable
                    self.stats[node.id]['sum1'] = node.valsum1
                    self.stats[node.id]['sum2'] = node.valsum2
                    self.stats[node.id]['N'] = node.valN

        if not quiet:
            print(f"\rChain {self.id} - Sampling completed")
        if run_sampled_count is not None:
            run_sampled_count[self.id] = N
