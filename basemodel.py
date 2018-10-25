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
        
        self.chains = []
        
        self.gelman_rubin = {}
        
        self.vars = {}
        self._trace_keys = None
        self.rp = Reporter()


    def burn_stats(self, burn_fraction=1.0):
        for chain in self.chains:
            chain.burn_stats(burn_fraction=burn_fraction)


    def get_trace_stats(self, combine=False):
        nchains = len(self.chains)
        dfs = []
        for i in range(nchains):
            df = pd.DataFrame(self.chains[i].stats).transpose()
            df.index.name = 'name'
            dfs.append(df)

        num_samples = dfs[0]['N'][0]

        if combine:
            stats = dfs[0]
            for df in dfs[1:]:
                stats += df

            if num_samples > 0:

                stats = stats.assign(mean=stats.apply(lambda r: r.sum1/r.N, axis=1))
                stats = stats.assign(var=stats.apply(lambda r: r.sum2/r.N - r['mean']**2, axis=1))
                stats = stats.assign(std=stats.apply(lambda r: np.sqrt(r['var']), axis=1))

        else:
            stats = []
            for df in dfs:
                if num_samples > 0:
                    df = df.assign(mean=df.apply(lambda r: r.sum1/r.N, axis=1))
                    df = df.assign(var=df.apply(lambda r: r.sum2/r.N - r['mean']**2, axis=1))
                    df = df.assign(std=df.apply(lambda r: np.sqrt(r['var']), axis=1))
                stats.append(df)

        return stats


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


    def init_chains(self, nchains=2):
        for ch in range(nchains):
            self.chains.append(Chain(self, ch))


    def get_gelman_rubin(self):

        nchains = len(self.chains)
        
        if nchains < 2:
            print('Need at least two chains for the convergence test')
            return
        
        trace_stats = self.get_trace_stats()

        num_samples = trace_stats[0]['N'][0]

        if num_samples == 0:
            return []

        # Calculate between-chain variance
        B = num_samples * pd.DataFrame([df['mean'] for df in trace_stats]).var(ddof=1)

        # Calculate within-chain variance
        W = pd.DataFrame([df['var'] for df in trace_stats]).mean()

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        var_table = pd.DataFrame({'W':W, 'Vhat':Vhat})

        gelman_rubin = var_table.apply(lambda r: np.sqrt(r.Vhat/r.W) if r.W > 0 else 1., axis=1)
        self.gelman_rubin = gelman_rubin

        return gelman_rubin


    def converged(self):

        gelman_rubin = self.get_gelman_rubin()

        if len(gelman_rubin) == 0:
            return False

        max_gr = gelman_rubin.max()
        
        if max_gr < 1.1:
            print("\nChains have converged")
            return True
        else:
            print(f"\nFailed to converge. "
                  f"Gelman-Rubin statistics was {max_gr: 7.4} for some parameter")
            return False


    def sample(self, N=200, thin=1, njobs=2):
            
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
