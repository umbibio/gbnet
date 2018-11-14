#cython: language_level=3, boundscheck=False, profile=True
from cython_gsl cimport gsl_rng
cimport numpy as np


cdef class RandomVariableNode:


    cdef public str id, name
    cdef object uid
    cdef public list parents, children, in_edges

    cdef void sample(self, gsl_rng *rng, bint update_stats)


cdef class Multinomial(RandomVariableNode):

    cdef bint value_cached
    cdef list _value

    cdef unsigned int *value
    cdef unsigned int *possible_values
    cdef double *prob
    cdef double *logprob
    cdef unsigned int noutcomes
    
    cdef unsigned int *value_buff
    
    cdef double *pr

    cdef double *valsum1
    cdef double *valsum2
    cdef public double valN

    cdef void get_outcome_probs(self)
    cdef double get_loglikelihood(self)
    cdef void sample(self, gsl_rng *rng, bint update_stats)


cdef class ORNOR_YLikelihood(Multinomial):

    cdef double get_model_likelihood(self)
    cpdef double get_loglikelihood(self)


cdef class Noise(RandomVariableNode):

    cdef public np.ndarray table, value
    cdef public Beta a, b

    cdef double *valsum1
    cdef double *valsum2
    cdef public double valN

    cdef void sample(self, gsl_rng *rng, bint update_stats)


cdef class Beta(RandomVariableNode):

    cdef double l_clip, r_clip, scale
    cdef object dist
    cdef list params
    cdef public double value
    cdef double a, b

    cdef public double valsum1, valsum2, valN

    cdef double proposal_norm(self, gsl_rng * rng)
    cdef double get_loglikelihood(self)
    cdef void metropolis_hastings(self, gsl_rng *rng)
    cdef void sample(self, gsl_rng *rng, bint update_stats)
