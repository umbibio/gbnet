import numpy as np
import scipy.stats as st

class Chain(object):
    def __init__(self, model, chain_id):
        
        self.vars = model.vars
        self.id = chain_id
        
        self.chain = {}
        self.trace_keys = model.trace_keys


    def sample(self, N, total_sampled=None, thin=1):

        steps_until_thin = thin

        for key in self.trace_keys:
            self.chain[key] = []
        
        # this will run in multiprocessing job, so we need to reset seed
        np.random.seed()
                
        updt_interval = max(1, N*0.0001)
        steps_until_updt = updt_interval

        for i in range(N):
            steps_until_updt -= 1
            if not steps_until_updt:
                if total_sampled is not None:
                    total_sampled[self.id] += updt_interval
                else:
                    print("\rChain {} - Progress {: 7.2%}".format(self.id, i/N), end="")
                steps_until_updt = updt_interval

            for vardict in self.vars.values():
                for node in vardict.values():
                    node.sample()

            steps_until_thin -= 1
            if not steps_until_thin:
                steps_until_thin = thin

                for vardict in self.vars.values():
                    for node in vardict.values():
                        try:
                            # if node is multinomial, value will be a numpy array
                            # have to set a list for each element in 'value'
                            for i, val in enumerate(node.value):
                                self.chain[f"{node.id}_{i}"].append(val)
                        except TypeError:
                            # value is no array, it won't be iterable
                            self.chain[node.id].append(node.value)


        print(f"\rChain {self.id} - Sampling completed")
        if total_sampled is not None:
            total_sampled[self.id] = N
        
        return self

