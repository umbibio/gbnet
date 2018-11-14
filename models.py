import numpy as np
import pandas as pd
from gbnet.basemodel import BaseModel
from gbnet.cnodes import Beta, RandomVariableNode, Noise
from gbnet.cnodes import Multinomial, ORNOR_YLikelihood


class ORNORModel(BaseModel):


    def generate_vars(self):

        rels = self.rels
        DEG = self.DEG

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
            Xnodes[src] = Multinomial('X', src, np.array([0.60, 0.40]))
            Tnodes[src] = Beta('T', src, 2, 2, scale=0.1)

        Snodes = {}
        for edg in rels.index:
            
            src, trg = edg
            
            Snodes[edg] = Multinomial('S', edg, np.array([0.1, 0.8, 0.1]))
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


class ORNORModel_2(BaseModel):


    def generate_vars(self):

        rels = self.rels
        DEG = self.DEG

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
            Xnodes[src] = Multinomial('X', src, np.array([0.99, 0.01]))

        Snodes = {}
        for edg in rels.index:

            Tnodes[edg] = Beta('T', edg, 2, 2, scale=0.1)

            src, trg = edg
            
            Snodes[edg] = Multinomial('S', edg, np.array([0.3, 0.4, 0.3]))
            Snodes[edg].children.append(Ynodes[trg])
            
            Xnodes[src].children.append(Ynodes[trg])
            Tnodes[edg].children.append(Ynodes[trg])
            
            Ynodes[trg].parents.append(Xnodes[src])
            Ynodes[trg].parents.append(Tnodes[edg])
            Ynodes[trg].parents.append(Snodes[edg])
            
            Ynodes[trg].in_edges.append([Xnodes[src], Tnodes[edg], Snodes[edg]])

        self.vars['Noise'] = {0: noiseNode}
        self.vars['X'] = Xnodes
        self.vars['S'] = Snodes
        self.vars['T'] = Tnodes


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

        Tres = result.loc[[f'T__{edge}' for edge in rels.index]]
        Tres = Tres.assign(edge=rels.index)

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
