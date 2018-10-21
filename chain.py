import numpy as np
import scipy.stats as st

class Chain(object):
    def __init__(self, model, chain_id):
        
        self.vars = model.vars
        self.id = chain_id
        
        self.stats = {}
        self.trace_keys = model.trace_keys

        self.reset_stats()


    def reset_stats(self):
        for key in self.trace_keys:
            self.stats[key] = { 'sum1': 0, 'sum2': 0, 'N': 0 }


    def sample(self, N, run_sampled_count=None, thin=1):

        steps_until_thin = thin

        # this will run in multiprocessing job, so we need to reset seed
        np.random.seed()
                
        updt_interval = max(1, N*0.0001)
        steps_until_updt = updt_interval

        for i in range(N):
            steps_until_updt -= 1
            if not steps_until_updt:
                if run_sampled_count is not None:
                    run_sampled_count[self.id] += updt_interval
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
                                self.stats[f"{node.id}_{i}"]['sum1'] += val
                                self.stats[f"{node.id}_{i}"]['sum2'] += val**2
                                self.stats[f"{node.id}_{i}"]['N'] += 1
                        except TypeError:
                            # value is no array, it won't be iterable
                            self.stats[node.id]['sum1'] += node.value
                            self.stats[node.id]['sum2'] += node.value**2
                            self.stats[node.id]['N'] += 1


        print(f"\rChain {self.id} - Sampling completed")
        if run_sampled_count is not None:
            run_sampled_count[self.id] = N
        
        return self

