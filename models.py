import numpy as np
import pandas as pd
from gbnet.basemodel import BaseModel
from gbnet.nodes import Beta, Multinomial


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
            likelihood[i] = self.get_model_likelihood() * self.prior_prob[i]

        self.value = curr_val

        return np.log(likelihood.sum())


    def sample(self):
        self.value = self.dist.rvs(*self.params)


class ORNORModel(BaseModel):

    __slots__ = ['rels']


    def __init__(self, rels, DEG, nchains=2):
        BaseModel.__init__(self)

        self.rels = rels

        # identify TF considered in current network
        X_list = rels['srcuid'].unique()

        # define a conditional probability table for the observed values
        # H is hidden true value of gene, Z is observed value in DEG
        a = 0.02
        b = 0.005

        PHZTable = np.empty(shape=(3,3), dtype=np.float64)
        PHZTable[0] = [1. - a - b,        a,          b]
        PHZTable[1] = [         a, 1. - 2*a,          a]
        PHZTable[2] = [         b,        a, 1. - a - b]

        Ynodes = {}
        for trg, val in DEG.items():
            obs_val = [0, 0, 0]
            obs_val[1 + val] = 1
            y_prob = PHZTable[:, np.argwhere(obs_val)[0, 0]]

            Ynodes[trg] = ORNOR_YLikelihood('Y', trg, y_prob, value=obs_val)

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

        self.vars['X'] = Xnodes
        self.vars['T'] = Tnodes
        self.vars['S'] = Snodes
        self.init_chains(nchains)


    def result(self, Xgt=None):

        result = self.get_trace_stats(combine=True).drop(columns=['N', 'sum1', 'sum2', 'var'])

        rels = self.rels
        src_uids = rels.srcuid.unique()

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
            'X': Xres,
            'T': Tres,
            'S': Sres,
        }
