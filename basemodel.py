import signal, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from gbnet.chain import Chain
from gbnet.aux import Reporter

class BaseModel(object):

    __slots__ = [
        'trace',
        'chains',
        'burn',
        'gelman_rubin',
        'max_gr',
        'vars',
        '_trace_keys',
        'rp',
    ]

    def __init__(self):
        
        self.trace = []
        self.chains = []
        
        self.burn = None
        self.gelman_rubin = {}
        self.max_gr = 0
        
        self.vars = {}
        self._trace_keys = None
        self.rp = Reporter()


    @property
    def trace_df(self):
        trace_length = len(next(iter(self.trace[0].values())))
        if self.burn < 1:
            burn = int(trace_length * self.burn)
        else:
            burn = self.burn
        return [pd.DataFrame(tr).iloc[burn:] for tr in self.trace]


    @property
    def trace_keys(self):
        if self._trace_keys is None:
            self._trace_keys = []
            for vardict in self.vars.values():
                for node in vardict.values():
                    try:
                        # if node is multinomial, value will be a numpy array
                        # have to set a list for each element in 'value'
                        for i in range(len(node.value)):
                            self._trace_keys.append(f"{node.id}_{i}")
                    except TypeError:
                        # value is no array, it won't have Attribute 'size'
                        self._trace_keys.append(node.id)
        return self._trace_keys


    def init_chains(self):
        chains = 2
        for ch in range(chains):
            self.chains.append(Chain(self, ch))
            self.trace.append({})
            for key in self.trace_keys:
                self.trace[ch][key] = []


    def rscore(self, x, num_samples):
        """Implementation taken from
        https://github.com/pymc-devs/pymc3/blob/master/pymc3/diagnostics.py
        
        Further reference:
        https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
        DOI: 10.1080/10618600.1998.10474787
        https://www.jstor.org/stable/2246093
        """
        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        return np.sqrt(Vhat / W)


    def get_trace(self, varname, chain=None, combine=False):
        trace_length = self.trace[0][varname].shape[0]
        if self.burn < 1:
            burn = int(trace_length * self.burn)
        else:
            burn = self.burn
        trace = []
        for t in self.trace:
            trace.append(t[varname][burn:, :])
        trace = np.array(trace)
        
        if combine:
            nvars = trace.shape[2]
            return trace.reshape((-1, nvars))
        
        if chain is not None:
            return trace[chain]
        
        return trace


    def run_convergence_test(self):
        
        if len(self.trace) < 2:
            print('Need at least two chains for the convergence test')
            return
        
        trace_df = self.trace_df

        num_samples = len(trace_df[0])

        # Calculate between-chain variance
        B = num_samples * pd.DataFrame([tr.mean() for tr in trace_df]).var(ddof=1)

        # Calculate within-chain variance
        W = pd.DataFrame([tr.var(ddof=1) for tr in trace_df]).mean()

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        var_table = pd.DataFrame({'B':B, 'W':W, 'Vhat':Vhat})

        gelman_rubin = var_table.apply(lambda r: np.sqrt(r.Vhat/r.W) if r.W > 0 else 1., axis=1)
        self.gelman_rubin = gelman_rubin


    def converged(self):

        if len(self.gelman_rubin) == 0:
            return False

        self.max_gr = self.gelman_rubin.max()
        
        if self.max_gr < 1.1:
            print("\nChains have converged")
            return True
        else:
            print(f"\nFailed to converge. "
                  f"Gelman-Rubin statistics was {self.max_gr: 7.4} for some parameter")
            return False


    def get_result(self, varname):
        trace = self.get_trace(varname, combine=True)
        mean = trace.mean(axis=0)
        std = trace.std(axis=0)
        return pd.DataFrame({'mean': mean, 'std': std})


    def sample(self, N=200, burn=None, thin=1, njobs=2):
        if burn is None:
            if self.burn is None:
                self.burn = 100
        else:
            self.burn = burn
            
        if njobs > 1:
            
            chains = len(self.chains)
            
            print(f"\nSampling {chains} chains in {njobs} jobs")
        
            # Want workers to ignore Keyboard interrupt
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Create the pool of workers
            pool = Pool(processes=njobs)
            # restore signals for the main process
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            try:
                manager = Manager()
                sampled = manager.list([0]*chains)
                mres = [pool.apply_async(chain.sample, (N, sampled, thin)) for chain in self.chains]
                pool.close()
                
                timer = 90 * 24 * 60 * 60 * 4 # 90 days timeout
                target_total = N * chains
                while timer:
                    time.sleep(1/4)
                    total_sampled = 0
                    for count in sampled:
                        total_sampled += count
                    progress = total_sampled / target_total
                    self.rp.report(f"Progress {progress: 7.2%}", schar='\r', lchar='', showlast=False)
                    if progress == 1:
                        break
                    timer -= 1
                self.rp.report(f"Progress {progress: 7.2%}", schar='\r')
                
                if timer <= 0:
                    raise TimeoutError

                self.chains = [res.get(timeout=3600) for res in mres]
            except KeyboardInterrupt:
                pool.terminate()
                print("\n\nCaught KeyboardInterrupt, workers have been terminated\n")
                raise SystemExit

            except TimeoutError:
                pool.terminate()
                print("\n\nThe workers ran out of time. Terminating simulation.\n")
                raise SystemExit
            
            pool.join()
        else:
            for chain in self.chains:
                chain.sample(N, thin=thin)
        
        for chain_id, trace in enumerate(self.trace):
            for key in self.trace_keys:
                trace[key] += self.chains[chain_id].chain[key]

        self.run_convergence_test()