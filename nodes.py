import numpy as np
import scipy.stats as st


class RandomVariableNode(object):


    __slots__ = [
        'id', 'name', 'uid', 'parents', 'children', 'blanket',
        'in_edges', 'value', 'total_sampled', 'dist', 'params']

    
    def __init__(self, name, uid=None, **kwargs):
        if uid is not None:
            self.id = f"{name}__{uid}"
        else:
            self.id = name
        self.name = name
        self.uid = uid

        try:
            self.value = kwargs['value']
        except KeyError:
            self.value = self.rvs()
        self.total_sampled = 0
        self.parents = []
        self.children = []
        self.blanket = []
        
        # special attribute that groups parent nodes
        # that are associated within each other
        # For example an interaction X->Y will have
        # corresponding T and S nodes associated only to X
        self.in_edges = []

    
    def rvs(self):
        return self.dist.rvs(*self.params) # pylint: disable=E1101


class Multinomial(RandomVariableNode):
    
    
    __slots__ = ['possible_values', 'prior_prob', 'prior_logprob']

    
    def __init__(self, *args, **kwargs):
        args, p = args[:-1], args[-1]
        self.dist = st.multinomial
        self.params = [1, p]
        self.prior_prob = p
        self.prior_logprob = np.log(p)
        self.possible_values = np.eye(len(p), dtype=np.int)
        RandomVariableNode.__init__(self, *args, **kwargs)


    def reset(self):
        self.value = np.zeros_like(self.value)
        self.value[np.argmax(self.params[1])] = 1


    def get_outcome_probs(self):

        curr_val = self.value
        
        logpr = np.zeros_like(curr_val, dtype=np.float64)
        for i, val in enumerate(self.possible_values):
            self.value = val
            logpr[i] = self.get_loglikelihood()
            
        pr = np.exp(logpr)

        if not pr.any():
            p = self.params[1]
        else:
            norm = pr.sum()
            p = pr / norm

        self.value = curr_val

        return p

    
    def get_loglikelihood(self):
        loglik = 0.
        for node in self.children:
            loglik += node.get_loglikelihood()
        #loglik += self.dist.logpmf(self.value, *self.params)
        loglik += self.prior_logprob[np.argmax(self.value)] # rely on N = 1
        
        return loglik

    
    def sample(self):
        p = self.get_outcome_probs()
        self.value = np.random.multinomial(1, p)
        self.total_sampled += 1


class Beta(RandomVariableNode):


    __slots__ = ['l_clip', 'r_clip', 'scale']


    def __init__(self, *args, **kwargs):
        args, a, b = args[:-2], args[-2], args[-1]
        self.dist = st.beta
        self.params = [a, b]

        try:
            self.l_clip = kwargs['l_clip']
        except KeyError:
            self.l_clip = 0.0

        try:
            self.r_clip = kwargs['r_clip']
        except KeyError:
            self.r_clip = 1.0

        try:
            self.scale = kwargs['scale']
        except KeyError:
            self.scale = 0.2

        RandomVariableNode.__init__(self, *args, **kwargs)


    def proposal_norm(self):
        prev = self.value
        scale = self.scale
        
        lft, rgt = self.l_clip, self.r_clip
        a, b = (lft - prev) / scale, (rgt - prev) / scale
        
        return st.truncnorm.rvs(a, b, prev, scale)


    def get_loglikelihood(self):
        # perhaps can implement a depth limit, to only consider variables in the Markov blanket
        # this is not necessary for this model because it only has two levels
        loglik = 0.
        for node in self.children:
            loglik += node.get_loglikelihood()
        loglik += self.dist.logpdf(self.value, *self.params) # pylint: disable=E1101
        
        return loglik


    def sample_from_prior(self):
        self.value = self.dist.rvs(*self.params) # pylint: disable=E1101
    

    def metropolis_hastings(self):
        
        prev = self.value
        prev_loglik = self.get_loglikelihood()
        self.value = self.proposal_norm()
        prop_loglik = self.get_loglikelihood()
        
        if prev_loglik > -np.inf:
            logratio = prop_loglik - prev_loglik
            accept = logratio >= 0. or logratio > -np.random.exponential()
            if not accept:
                self.value = prev


    def sample(self):
        self.metropolis_hastings()
        #self.sample_from_prior()

