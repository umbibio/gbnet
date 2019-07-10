#cython: language_level=3, boundscheck=False
import cython
from cython import sizeof
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy

from libc.stdlib cimport malloc
from libc.string cimport strcpy, strlen
from libc.math cimport sqrt, exp, log, INFINITY, pow
from libc.time cimport time

import numpy as np
cimport numpy as np

from cython_gsl cimport *

# define the random number generator
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
# seed it with current time
gsl_rng_set(r, time(NULL))

# define infinite constant
cdef double inf = INFINITY

cdef class RandomVariableNode:

    def __init__(self, str name, valSize, uid=None):

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

        self.valSize = valSize
        self.valsum1 = [0.] * self.valSize
        self.valsum2 = [0.] * self.valSize
        self.valN = 0.

    property valsum1:
        def __get__(self):
            return [self._valsum1[i] for i in range(self.valSize)]

        def __set__(self, values):
            assert self.valSize == len(values), "Assigned value doesn't match property size"
            self._valsum1 = <double *> PyMem_Malloc(sizeof(double) * self.valSize)
            if not self._valsum1:
                raise MemoryError()
            for i in range(self.valSize):
                self._valsum1[i] = values[i]

    cpdef bytes get_valsum1(self):
        if self._valsum1 == NULL:
            return None
        return <bytes>(<char *>self._valsum1)[:sizeof(double) * self.valSize]
    
    cpdef void set_valsum1(self, bytes valsum1):
        assert self.valSize > 0, "valSize has not been previously specified"
        PyMem_Free(self._valsum1)
        self._valsum1 = <double *> PyMem_Malloc(sizeof(double) * self.valSize)
        if not self._valsum1:
            raise MemoryError()
        memcpy(self._valsum1, <char *>valsum1, sizeof(double) * self.valSize)

    property valsum2:
        def __get__(self):
            return [self._valsum2[i] for i in range(self.valSize)]

        def __set__(self, values):
            assert self.valSize == len(values), "Assigned value doesn't match property size"
            self._valsum2 = <double *> PyMem_Malloc(sizeof(double) * self.valSize)
            if not self._valsum2:
                raise MemoryError()
            for i in range(self.valSize):
                self._valsum2[i] = values[i]

    cpdef bytes get_valsum2(self):
        if self._valsum2 == NULL:
            return None
        return <bytes>(<char *>self._valsum2)[:sizeof(double) * self.valSize]
    
    cpdef void set_valsum2(self, bytes valsum2):
        assert self.valSize > 0, "valSize has not been previously specified"
        PyMem_Free(self._valsum2)
        self._valsum2 = <double *> PyMem_Malloc(sizeof(double) * self.valSize)
        if not self._valsum2:
            raise MemoryError()
        memcpy(self._valsum2, <char *>valsum2, sizeof(double) * self.valSize)

    def __getstate__(self):
        return (
            self.id, self.name, self.uid, 
            self.parents, self.children, self.in_edges,
            self.valSize, self.get_valsum1(), self.get_valsum2(), self.valN,
        )
    
    def __setstate__(self, state):
        (
            self.id, self.name, self.uid, 
            self.parents, self.children, self.in_edges,
            self.valSize, tmp_valsum1, tmp_valsum2, self.valN,
        ) = state
        self.set_valsum1(tmp_valsum1)
        self.set_valsum2(tmp_valsum2)

    def __dealloc__(self):
        PyMem_Free(self._valsum1)
        PyMem_Free(self._valsum2)

    def burn_stats(self, burn_fraction=1.0):
        keep_fraction = 1. - burn_fraction
        self.valN *= keep_fraction
        for i in range(self.valSize):
            self._valsum1[i] *= keep_fraction
            self._valsum2[i] *= keep_fraction

    def add_child(self, object elem):
        self.children.append(elem)

    def add_parent(self, object elem):
        self.parents.append(elem)

    def add_in_edges(self, object elem):
        self.in_edges.append(elem)

    cdef void sample(self, gsl_rng *rng, bint update_stats):
        pass


cdef class Multinomial(RandomVariableNode):


    def __init__(self, str name, uid, np.ndarray p,
                 np.ndarray value=np.array([], dtype=np.int32)):

        if not value.shape[0]:
            value = np.random.multinomial(1, p).astype(np.int32)

        self.value_idx = np.argwhere(value)[0, 0]

        cdef unsigned int size = p.shape[0]

        self.noutcomes = size

        self._value_buff = <unsigned int *> malloc(size * sizeof(unsigned int))
        self._pr = <double *> malloc(size * sizeof(double))

        self._prob = <double *> malloc(size * sizeof(double))
        self._logprob = <double *> malloc(size * sizeof(double))
        
        self._value = self._value_buff
        self._possible_values = <unsigned int *> malloc(size * size * sizeof(unsigned int))

        cdef unsigned int i, j, v
        for i in range(size):
            self._value[i] = value[i]
            self._prob[i] = p[i]
            self._logprob[i] = log(p[i])
            for j in range(size):
                if i == j:
                    v = 1
                else:
                    v = 0
                self._possible_values[i*size + j] = v

        self._cache_value = [self._value[i] for i in range(self.noutcomes)]
        self.value_is_cached = 1

        RandomVariableNode.__init__(self, name, self.noutcomes, uid)

    def __getstate__(self):
        state = super(Multinomial, self).__getstate__()
        state = state + (
            [self._value[i] for i in range(self.valSize)], 
            [self._possible_values[i] for i in range(self.valSize**2)], 
            [self._prob[i] for i in range(self.valSize)], 
        )
        return state

    def __setstate__(self, state):
        super(Multinomial, self).__setstate__(state[:-3])
        (
            tmp_value, 
            tmp_possible_values, 
            tmp_prob, 
        ) = state[-3:]
        assert len(tmp_value) == len(tmp_prob) == self.valSize, "Wrong array size in assignment"
        assert len(tmp_possible_values) == self.valSize**2, "Wrong array size in assignment"

        self._value_buff = <unsigned int *>PyMem_Malloc(sizeof(unsigned int) * self.valSize)
        self._value = self._value_buff
        self._possible_values = <unsigned int *>PyMem_Malloc(sizeof(unsigned int) * self.valSize**2)
        self._prob = <double *>PyMem_Malloc(sizeof(double) * self.valSize)
        self._logprob = <double *>PyMem_Malloc(sizeof(double) * self.valSize)

        self._pr = <double *>PyMem_Malloc(sizeof(double) * self.valSize)

        if (not self._value or not self._possible_values or not self._value_buff or
            not self._prob or not self._logprob or not self._pr):
            raise MemoryError()

        for i in range(self.valSize):
            self._value[i] = tmp_value[i]
            for j in range(self.valSize):
                self._possible_values[i*self.valSize + j] = tmp_possible_values[i*self.valSize + j]
            self._prob[i] = tmp_prob[i]
            self._logprob[i] = log(tmp_prob[i])

        self.noutcomes = self.valSize
        self._cache_value = [self._value[i] for i in range(self.noutcomes)]
        self.value_is_cached = 1


    @cython.cdivision(True)
    cdef void get_outcome_probs(self):

        cdef unsigned int size = self.noutcomes

        cdef double llik, lik
        cdef double sum_lik = 0

        cdef unsigned int i, j

        for i in range(size):
            self._value = &self._possible_values[i*size]
            llik = self.get_loglikelihood()
            lik = exp(llik)
            self._pr[i] = lik
            sum_lik += lik

        if sum_lik > 0:
            for i in range(size):
                self._pr[i] = self._pr[i] / sum_lik
        else:
            for i in range(size):
                self._pr[i] = self._prob[i]

        self._value = self._value_buff

    
    cdef double get_loglikelihood(self):
        cdef double loglik = 0.

        cdef ORNOR_YLikelihood node

        for node in self.children:
            loglik += node.get_loglikelihood()
        loglik += gsl_ran_multinomial_lnpdf(self.noutcomes, self._prob, self._value)
        
        return loglik

    
    cdef void sample(self, gsl_rng *rng, bint update_stats):
        self.get_outcome_probs()
        cdef unsigned int N = 1
        gsl_ran_multinomial(rng, self.noutcomes, N, self._pr, self._value)
        self.value_is_cached = 0
        if update_stats:
            self.valN += 1.
            for i in range(self.noutcomes):
                self._valsum1[i] += <double> self._value[i]
                self._valsum2[i] += <double> self._value[i] * self._value[i]


cdef class ORNOR_YLikelihood(Multinomial):

    def __getstate__(self):
        state = super(ORNOR_YLikelihood, self).__getstate__()
        state = state + (
            self.Znode,
        )
        return state

    def __setstate__(self, state):
        super(ORNOR_YLikelihood, self).__setstate__(state[:-1])
        (
            self.Znode,
        ) = state[-1:]


    @property
    def prob(self):
        cdef unsigned int i
        return [self._prob[i] for i in range(self.noutcomes)]

    @prob.setter
    def prob(self, np.ndarray[double, ndim=1] value):
        cdef unsigned int i
        for i in range(self.noutcomes):
            self._prob[i] = value[i]

    def set_Znode(self, node):
        self.Znode = node

    cdef double get_model_likelihood(self):
        cdef double likelihood, pr0, pr1, pr2, q0, q1, q2, zvalue, zcompl, zcompl_pn
        cdef Multinomial x, s
        cdef Beta t, z

        zvalue = self.Znode.value
        zcompl = 1.- zvalue
        zcompl_pn = 1.

        # deg dependant, TODO: compute this values from input deg file
        q0 = 0.2
        q1 = 0.8
        q2 = 0.2

        if self._value[0]:
            pr0 = 1.
            for x, t, s in self.in_edges:
                if s._value[0]:
                    zcompl_pn *= zcompl
                    pr0 *= (1. - t.value * x._value[1]) * zvalue
                elif s._value[2]:
                    zcompl_pn *= zcompl
            pr0 = (1. - pr0) * (1. - zcompl_pn) + zcompl_pn * q0
            likelihood = pr0 

        elif self._value[2]:
            pr0 = 1.
            pr2 = 1.
            for x, t, s in self.in_edges:
                if s._value[2]:
                    zcompl_pn *= zcompl
                    pr2 *= (1. - t.value * x._value[1]) * zvalue
                elif s._value[0]:
                    zcompl_pn *= zcompl
                    pr0 *= (1. - t.value * x._value[1]) * zvalue
            pr2 = (pr0 - pr2*pr0) * (1. - zcompl_pn) + zcompl_pn * q2
            likelihood = pr2

        else:
            pr1 = 1.
            for x, t, s in self.in_edges:
                if not s._value[1]:
                    zcompl_pn *= zcompl
                    pr1 *= (1. - t.value * x._value[1]) * zvalue
            pr1 = pr1 * (1. - zcompl_pn) + zcompl_pn * q1
            likelihood = pr1
        
        return likelihood #* 0.997 + 0.001


    cpdef double get_loglikelihood(self):

        cdef unsigned int size = self.noutcomes

        cdef unsigned int i, j

        cdef double likelihood = 0.
        for i in range(size):
            self._value = &self._possible_values[i*size]

            likelihood += self.get_model_likelihood() * self._prob[i]

        self._value = self._value_buff

        # penalize if this target gene wasn't differentially expressed
        # likelihood *= 1. - self._value[1]*0.9999

        return log(likelihood)


    def sample(self):
        pass


cdef class Noise(RandomVariableNode):


    def __init__(self, name, uid, a=0.0050, b=0.0005):
        self.a = Beta('a', 0, 1/(1-a), 1/a, value=a, r_clip=0.5)
        self.b = Beta('b', 0, 1/(1-b), 1/b, value=b, r_clip=0.5)
        self.table = np.eye(3, dtype=float)
        self.update()

        RandomVariableNode.__init__(self, name, 2, uid)

        self.parents.append(self.a)
        self.parents.append(self.b)

    def __getstate__(self):
        state = super(Noise, self).__getstate__()
        state = state + (
            self.table, self.value, 
            self.a, self.b,
        )
        return state

    def __setstate__(self, state):
        super(Noise, self).__setstate__(state[:-4])
        (
            self.table, self.value, 
            self.a, self.b,
        ) = state[-4:]

    def update(self):
        a = self.a.value
        b = self.b.value

        self.table[0] = [1. - a - b,          a,          b]
        self.table[1] = [         a, 1. - 2 * a,          a]
        self.table[2] = [         b,          a, 1. - a - b]

        self.value = np.array([a, b])


    cdef void sample(self, gsl_rng *rng, bint update_stats):
        self.a.sample(rng, False)
        self.update()
        self.b.sample(rng, False)
        self.update()
        
        for Ynod in self.children:
            y_prob = self.table[:, Ynod.value_idx]
            Ynod.prob = y_prob

        if update_stats:
            self.valN += 1.
            for i in range(2):
                self._valsum1[i] += <double> self.value[i]
                self._valsum2[i] += <double> self.value[i] * self.value[i]

    def rvs(self):
        return self.value


cdef class Beta(RandomVariableNode):


    def __init__(self, name, uid, a, b, value=None, l_clip=0.0, r_clip=1.0, scale=-1.0):
        RandomVariableNode.__init__(self, name, 1, uid)

        if scale < 0.0:
            # use the standard deviation of beta distribution
            scale = np.sqrt(a*b/((a+b)**2*(a+b+1)))
        
        self.l_clip = l_clip
        self.r_clip = r_clip
        self.scale = scale

        self.params = [a, b]
        self.a = a
        self.b = b

        if value is not None:
            self.value = value
        else:
            self.value = self.rvs()

    def __getstate__(self):
        state = super(Beta, self).__getstate__()
        state = state + (
            self.l_clip, self.r_clip, self.scale,
            self.params, self.value, self.a, self.b, 
        )
        return state

    def __setstate__(self, state):
        super(Beta, self).__setstate__(state[:-7])
        (
            self.l_clip, self.r_clip, self.scale,
            self.params, self.value, self.a, self.b, 
        ) = state[-7:]

    @cython.cdivision(True)
    cdef double proposal_norm(self, gsl_rng * rng):

        cdef double proposal
        cdef bint good = False

        while not good:
            proposal = self.value + gsl_ran_gaussian(rng, self.scale)
            if proposal > self.l_clip and proposal < self.r_clip:
                good = True
        
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


    cdef void metropolis_hastings(self, gsl_rng * rng):

        cdef bint accept

        prev = self.value
        prev_loglik = self.get_loglikelihood()
        self.value = self.proposal_norm(rng)
        prop_loglik = self.get_loglikelihood()
        
        if prev_loglik > -inf:
            logratio = prop_loglik - prev_loglik
            accept = logratio >= 0. or logratio > - gsl_ran_exponential(rng, 1.0)
            if not accept:
                self.value = prev


    cdef void sample(self, gsl_rng * rng, bint update_stats):
        self.metropolis_hastings(rng)

        if update_stats:
            self.valN += 1.
            self._valsum1[0] += self.value
            self._valsum2[0] += self.value * self.value

    def rvs(self):
        good = False

        while not good:
            value = np.random.beta(self.a, self.b)
            if value > self.l_clip and value < self.r_clip:
                good = True

        return value
