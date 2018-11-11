import numpy as np
import pandas as pd
from gbnet.basemodel import BaseModel
from gbnet.nodes import Beta, Multinomial, RandomVariableNode


class ORNOR_YLikelihood(Multinomial):


    __slots__ = []


    def get_model_likelihood(self):
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


    def get_loglikelihood(self):
        curr_val = self.value
        
        likelihood = np.zeros_like(curr_val, dtype=np.float64)
        for i, val in enumerate(self.possible_values):
            self.value = val
            likelihood[i] = self.get_model_likelihood() * self.prob[i]

        self.value = curr_val

        return np.log(likelihood.sum())


    def sample(self):
        pass
        #self.value = self.dist.rvs(*self.params)


class Noise(RandomVariableNode):


    __slots__ = ['table', 'a', 'b']


    def __init__(self, *args, a=0.050, b=0.001, **kwargs):
        self.a = Beta('a', 0, 5, 100, value=a, r_clip=0.5, step=0.02)
        self.b = Beta('b', 0, 1, 100, value=b, r_clip=0.5, step=0.02)
        self.table = np.eye(3, dtype=float)
        self.update_table()

        RandomVariableNode.__init__(self, *args, **kwargs)

        self.parents.append(self.a)
        self.parents.append(self.b)


    def update_table(self):
        a = self.a.value
        b = self.b.value
        self.table[0] = [1. - a - b,          a,          b]
        self.table[1] = [         a, 1. - 2 * a,          a]
        self.table[2] = [         b,          a, 1. - a - b]


    def sample(self):
        self.a.sample()
        self.update_table()
        self.b.sample()
        self.update_table()
        self.value = np.array([self.a.value, self.b.value])
        for Ynod in self.children:
            y_prob = self.table[:, np.argwhere(Ynod.value)[0, 0]]
            Ynod.prior_prob = y_prob


    def rvs(self):
        return np.array([self.a.value, self.b.value])


class ORNORModel(BaseModel):


    __slots__ = ['rels']


    def __init__(self, rels, DEG, nchains=2):
        BaseModel.__init__(self)

        self.rels = rels

        # identify TF considered in current network
        X_list = rels['srcuid'].unique()

        noiseNode = Noise('Noise', None, a=0.05, b=0.001)

        Ynodes = {}
        for trg, val in DEG.items():
            obs_val = np.array([0, 0, 0])
            obs_val[1 + val] = 1
            y_prob = noiseNode.table[:, np.argwhere(obs_val)[0, 0]]

            Ynodes[trg] = ORNOR_YLikelihood('Y', trg, y_prob, value=obs_val)
            noiseNode.a.children.append(Ynodes[trg])
            noiseNode.b.children.append(Ynodes[trg])
            noiseNode.children.append(Ynodes[trg])

        Xnodes, Tnodes = {}, {}
        for src in X_list:
            Xnodes[src] = Multinomial('X', src, [0.99, 0.01])
            Tnodes[src] = Beta('T', src, 2, 2)

        Snodes = {}
        for edg in rels.index:
            
            src, trg = edg
            
            Snodes[edg] = Multinomial('S', edg, [0.1, 0.8, 0.1])
            Snodes[edg].children.append(Ynodes[trg])
            
            Xnodes[src].children.append(Ynodes[trg])
            Tnodes[src].children.append(Ynodes[trg])
            
            Ynodes[trg].parents.append(Xnodes[src])
            Ynodes[trg].parents.append(Tnodes[src])
            Ynodes[trg].parents.append(Snodes[edg])
            
            Ynodes[trg].in_edges.append([Xnodes[src], Tnodes[src], Snodes[edg]])

        self.vars['Noise'] = {0: noiseNode}
        self.vars['X'] = Xnodes
        self.vars['T'] = Tnodes
        self.vars['S'] = Snodes
        self.init_chains(nchains)


    def result(self, Xgt=None):

        result = self.get_trace_stats(combine=True).drop(columns=['N', 'sum1', 'sum2', 'var'])

        rels = self.rels
        src_uids = rels.srcuid.unique()

        Noiseres = result.loc[[f'Noise_{i}' for i in [0,1]]]

        Xres = result.loc[[f'X__{src}_1' for src in src_uids]]
        Xres = Xres.assign(srcuid=src_uids)
        Xres['pred'] = Xres.apply(lambda r: 1 if r['mean']>0.5 else 0, axis =1)
        Xres = Xres.assign(idx=[f'X__{src}' for src in src_uids])
        Xres = Xres.set_index('idx')
        if Xgt is not None:
            Xres = Xres.assign(gt=[Xgt[src] for src in src_uids])

        Tres = result.loc[[f'T__{src}' for src in src_uids]]
        Tres = Tres.assign(srcuid=src_uids)

        Sres0 = result.loc[[f'S__{edge}_0' for edge in rels.index]]['mean'].to_frame('-')
        Sres0 = Sres0.assign(idx=[f'S__{edge}' for edge in rels.index])
        Sres0 = Sres0.set_index('idx')
        Sres2 = result.loc[[f'S__{edge}_2' for edge in rels.index]]['mean'].to_frame('+')
        Sres2 = Sres2.assign(idx=[f'S__{edge}' for edge in rels.index])
        Sres2 = Sres2.set_index('idx')

        Sres = pd.concat([Sres0, Sres2], sort=False, axis=1)
        Sres = Sres.assign(srcuid=list(rels['srcuid']))
        Sres = Sres.assign(trguid=list(rels['trguid']))
        Sres = Sres.assign(pred=Sres.apply(lambda r: 1 if r['+']>0.5 else (-1 if r['-']>0.5 else 0), axis=1))
        try:
            Sres = Sres.assign(gt=list(rels['val']))
        except KeyError:
            pass

        return {
            'Noise': Noiseres,
            'X': Xres,
            'T': Tres,
            'S': Sres,
        }
