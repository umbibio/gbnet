#cython: language_level=3, boundscheck=False, profile=True
import cython
from cython import sizeof

from libc.stdlib cimport malloc
from libc.string cimport strcpy, strlen
from libc.math cimport sqrt, exp, log, INFINITY
from libc.time cimport time

import numpy as np
cimport numpy as np

import scipy.stats as st

from cython_gsl cimport *

# define the random number generator
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
# seed it with current time
gsl_rng_set(r, time(NULL))

# define infinite constant
cdef double inf = INFINITY

cdef class RandomVariableNode:


    cdef public str id, name
    cdef object uid
    cdef public list parents, children, in_edges

    
    def __init__(self, str name, uid=None):

        if uid is not None:
            self.id = f"{name}__{uid}"
        else:
            self.id = name

        self.name = name
        self.uid = uid

        self.parents = []
        self.children = []
        
        # special attribute that groups parent nodes
        # that are associated within each other
        # For example an interaction X->Y will have
        # corresponding T and S nodes associated only to X
        self.in_edges = []

    def add_child(self, object elem):
        self.children.append(elem)

    def add_parent(self, object elem):
        self.parents.append(elem)

    def add_in_edges(self, object elem):
        self.in_edges.append(elem)


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


    def __init__(self, str name, uid, np.ndarray p,
                 np.ndarray value=np.array([], dtype=np.int32)):

        if not value.shape[0]:
            value = np.random.multinomial(1, p).astype(np.int32)

        cdef unsigned int size = p.shape[0]

        self.noutcomes = size

        self.value_buff = <unsigned int *> malloc(size * sizeof(unsigned int))
        self.pr = <double *> malloc(size * sizeof(double))

        self.prob = <double *> malloc(size * sizeof(double))
        self.logprob = <double *> malloc(size * sizeof(double))
        
        self.value = <unsigned int *> malloc(size * sizeof(unsigned int))
        self.possible_values = <unsigned int *> malloc(size * size * sizeof(unsigned int))

        self.valsum1 = <double *> malloc(size * sizeof(double))
        self.valsum2 = <double *> malloc(size * sizeof(double))
        self.valN = 0.


        cdef unsigned int i, j, v
        for i in range(size):
            self.value[i] = value[i]
            self.prob[i] = p[i]
            self.logprob[i] = log(p[i])
            self.valsum1[i] = 0.
            self.valsum2[i] = 0.
            for j in range(size):
                if i == j:
                    v = 1
                else:
                    v = 0
                self.possible_values[i*size + j] = v

        self.value_cached = 0

        RandomVariableNode.__init__(self, name, uid)


    @property
    def value(self):
        if not self.value_cached:
            self._value = [self.value[i] for i in range(self.noutcomes)]
            self.value_cached = 1
        return self._value

    @property
    def valsum1(self):
        return [self.valsum1[i] for i in range(self.noutcomes)]
    
    @property
    def valsum2(self):
        return [self.valsum2[i] for i in range(self.noutcomes)]
    
    @cython.cdivision(True)
    cdef void get_outcome_probs(self):

        cdef unsigned int size = self.noutcomes

        cdef double llik, lik
        cdef double sum_lik = 0

        cdef unsigned int i, j
        for i in range(size):
            self.value_buff[i] = self.value[i]

        for i in range(size):
            for j in range(size):
                self.value[j] = self.possible_values[i*size + j]
            llik = self.get_loglikelihood()
            lik = exp(llik)
            self.pr[i] = lik
            sum_lik += lik

        if sum_lik > 0:
            for i in range(size):
                self.pr[i] = self.pr[i] / sum_lik
        else:
            for i in range(size):
                self.pr[i] = self.prob[i]

        for i in range(size):
            self.value[i] = self.value_buff[i]

    
    cdef double get_loglikelihood(self):
        cdef double loglik = 0.

        cdef ORNOR_YLikelihood node

        for node in self.children:
            loglik += node.get_loglikelihood()
        loglik += gsl_ran_multinomial_lnpdf(self.noutcomes, self.prob, self.value)
        
        return loglik

    
    def sample(self, update_stats=False):
        self.get_outcome_probs()
        cdef unsigned int N = 1
        gsl_ran_multinomial(r, self.noutcomes, N, self.pr, self.value)
        self.value_cached = 0
        if update_stats:
            self.valN += 1.
            for i in range(self.noutcomes):
                self.valsum1[i] += <double> self.value[i]
                self.valsum2[i] += <double> self.value[i] * self.value[i]


cdef class ORNOR_YLikelihood(Multinomial):

    @property
    def prob(self):
        cdef unsigned int i
        return [self.prob[i] for i in range(self.noutcomes)]

    @prob.setter
    def prob(self, np.ndarray[double, ndim=1] value):
        cdef unsigned int i
        for i in range(self.noutcomes):
            self.prob[i] = value[i]


    cdef double get_model_likelihood(self):
        cdef double likelihood, pr0, pr1, pr2
        cdef Multinomial x, s
        cdef Beta t

        if self.value[0]:
            pr0 = 1.
            for x, t, s in self.in_edges:
                if s.value[0]:
                    pr0 *= 1. - t.value * x.value[1]
            pr0 = (1. - pr0)
            likelihood = pr0

        elif self.value[2]:
            pr0 = 1.
            pr2 = 1.
            for x, t, s in self.in_edges:
                if s.value[2]:
                    pr2 *= 1. - t.value * x.value[1]
                elif s.value[0]:
                    pr0 *= 1. - t.value * x.value[1]
            pr2 = (pr0 - pr2*pr0)
            likelihood = pr2

        else:
            pr1 = 1.
            for x, t, s in self.in_edges:
                if not s.value[1]:
                    pr1 *= 1. - t.value * x.value[1]
            likelihood = pr1
        
        return likelihood


    cdef public double get_loglikelihood(self):

        cdef unsigned int size = self.noutcomes

        cdef unsigned int i, j
        for i in range(size):
            self.value_buff[i] = self.value[i]
        
        cdef double likelihood = 0.
        for i in range(size):
            for j in range(size):
                self.value[j] = self.possible_values[i*size + j]
            likelihood += self.get_model_likelihood() * self.prob[i]

        for i in range(size):
            self.value[i] = self.value_buff[i]

        return log(likelihood)


    def sample(self):
        pass
        #self.value = self.dist.rvs(*self.params)


cdef class Noise(RandomVariableNode):


    cdef public np.ndarray table, value
    cdef public Beta a, b

    cdef double *valsum1
    cdef double *valsum2
    cdef public double valN


    def __init__(self, name, uid, a=0.050, b=0.001):
        self.a = Beta('a', 0, 5, 100, value=a, r_clip=0.5, scale=0.02)
        self.b = Beta('b', 0, 1, 100, value=b, r_clip=0.5, scale=0.02)
        self.table = np.eye(3, dtype=float)
        self.update()

        RandomVariableNode.__init__(self, name, uid)

        self.parents.append(self.a)
        self.parents.append(self.b)

        self.valsum1 = <double *> malloc(2 * sizeof(double))
        self.valsum2 = <double *> malloc(2 * sizeof(double))
        self.valN = 0.

        cdef unsigned int i
        for i in range(2):
            self.valsum1[i] = 0.
            self.valsum2[i] = 0.

    def update(self):
        a = self.a.value
        b = self.b.value

        self.table[0] = [1. - a - b,          a,          b]
        self.table[1] = [         a, 1. - 2 * a,          a]
        self.table[2] = [         b,          a, 1. - a - b]

        self.value = np.array([self.a.value, self.b.value])


    def sample(self, update_stats=False):
        self.a.sample()
        self.update()
        self.b.sample()
        self.update()
        
        for Ynod in self.children:
            y_prob = self.table[:, np.argwhere(Ynod.value)[0, 0]]
            Ynod.prob = y_prob

        if update_stats:
            self.valN += 1.
            for i in range(2):
                self.valsum1[i] += <double> self.value[i]
                self.valsum2[i] += <double> self.value[i] * self.value[i]

    @property
    def valsum1(self):
        return [self.valsum1[i] for i in range(2)]
    
    @property
    def valsum2(self):
        return [self.valsum2[i] for i in range(2)]
    

    def rvs(self):
        return np.array([self.a.value, self.b.value])

cdef class Beta(RandomVariableNode):


    cdef double l_clip, r_clip, scale
    cdef object dist
    cdef list params
    cdef public double value
    cdef double a, b

    cdef public double valsum1, valsum2, valN


    def __init__(self, name, uid, a, b, value=None, l_clip=0.0, r_clip=1.0, scale=0.02):
        
        self.dist = st.beta
        self.params = [a, b]
        self.a = a
        self.b = b

        if value is not None:
            self.value = value
        else:
            self.value = self.rvs()

        self.l_clip = l_clip
        self.r_clip = r_clip
        self.scale = scale

        self.valsum1 = 0.
        self.valsum2 = 0.
        self.valN = 0.
        
        RandomVariableNode.__init__(self, name, uid)

    @cython.cdivision(True)
    cdef double proposal_norm(self):

        cdef double proposal

        prev = self.value
        scale = self.scale
        
        lft, rgt = self.l_clip, self.r_clip
        a, b = (lft - prev) / scale, (rgt - prev) / scale

        proposal = st.truncnorm.rvs(a, b, prev, scale)
        
        return proposal


    cdef double get_loglikelihood(self):
        # perhaps can implement a depth limit, to only consider variables in the Markov blanket
        # this is not necessary for this model because it only has two levels
        cdef double loglik, lik
        cdef ORNOR_YLikelihood node

        lik = gsl_ran_beta_pdf(self.value, self.a, self.b)
        loglik = log(lik)

        for node in self.children:
            loglik += node.get_loglikelihood()

        
        return loglik


    def sample_from_prior(self):
        self.value = self.dist.rvs(*self.params)
    

    cdef void metropolis_hastings(self):

        cdef bint accept
        
        prev = self.value
        prev_loglik = self.get_loglikelihood()
        self.value = self.proposal_norm()
        prop_loglik = self.get_loglikelihood()
        
        if prev_loglik > -inf:
            logratio = prop_loglik - prev_loglik
            accept = logratio >= 0. or logratio > - gsl_ran_exponential(r, 1.0)
            if not accept:
                self.value = prev


    def sample(self, update_stats=False):
        self.metropolis_hastings()
        #self.sample_from_prior()
        if update_stats:
            self.valN += 1.
            self.valsum1 += self.value
            self.valsum2 += self.value * self.value

    def rvs(self):
        return self.dist.rvs(*self.params)
