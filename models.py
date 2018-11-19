import numpy as np
import pandas as pd
from gbnet.basemodel import BaseModel
from gbnet.cnodes import Beta, RandomVariableNode, Noise
from gbnet.cnodes import Multinomial, ORNOR_YLikelihood


class ORNORModel_0(BaseModel):


    def generate_vars(self):

        rels = self.rels
        DEG = self.DEG

        rels = rels.assign(DEG=[DEG[trg] for trg in rels['trguid']])

        # identify TF considered in current network
        X_list = rels['srcuid'].unique()

        X = pd.Series(X_list, name='srcuid').to_frame()
        X = X.assign(out_total=[len(rels[rels['srcuid']==src]) for src in X_list])
        X = X.assign(out_deg=[rels[rels.srcuid==src]['DEG'].abs().sum() for src in X_list])
        X = X.set_index('srcuid')
        X = X.assign(prior0=X.apply(lambda r: 0.95 - r['out_deg'] / r['out_total'] * 0.8, axis=1))

        noiseNode = Noise('Noise', None, a=0.005, b=0.0005)

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
            X0prior = X.loc[src]['prior0']
            X1prior = 1. - X0prior
            Xnodes[src] = Multinomial('X', src, np.array([X0prior, X1prior]))
            Tnodes[src] = Beta('T', src, 20, 1, scale=0.1)

        Snodes = {}
        for edg in rels.index:
            
            src, trg = edg

            if Ynodes[trg].value[1]:
                Sprior = np.array([0.2, 0.6, 0.2])
            else:
                Sprior = np.array([0.4, 0.2, 0.4])
            
            Snodes[edg] = Multinomial('S', edg, Sprior)
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


    def update_result(self, Xgt=None):

        result = self.get_trace_stats(combine=True).drop(columns=['N', 'sum1', 'sum2', 'var'])

        ents = self.ents
        rels = self.rels
        DEG = self.DEG
        rels = rels.assign(DEG=[DEG[trg] for trg in rels['trguid']])
        src_uids = rels.srcuid.unique()

        Noiseres = result.loc[[f'Noise_{i}' for i in [0,1]]]

        Tres = result.loc[[f'T__{src}' for src in src_uids]]
        Tres = Tres.assign(srcuid=src_uids)
        Tres = Tres.set_index('srcuid')

        Xres = result.loc[[f'X__{src}_1' for src in src_uids]]['mean'].to_frame('X')
        Xres = Xres.assign(srcuid=src_uids)
        Xres = Xres.set_index('srcuid')
        Xres = Xres.assign(T=Tres['mean'])
        Xres['pred'] = Xres.apply(lambda r: 1 if r['X']*r['T']>0.5 else 0, axis =1)
        if Xgt is not None:
            Xres = Xres.assign(gt=[Xgt[src] for src in src_uids])


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
        deg = pd.Series(self.DEG)
        Sres.assign(DEG=list(deg.loc[Sres['trguid']]))
        try:
            Sres = Sres.assign(gt=list(rels['val']))
        except KeyError:
            pass

        if ents is not None:
            Xres = Xres.assign(name=list(ents.loc[Xres.index].name))
            Tres = Tres.assign(name=list(ents.loc[Tres.index].name))
            Sres = Sres.assign(srcname=list(ents.loc[Sres['srcuid']].name))
            Sres = Sres.assign(trgname=list(ents.loc[Sres['trguid']].name))
        Sres = Sres.assign(DEG=list(rels['DEG']))
        Xres = Xres.assign(out_total=[len(rels[rels['srcuid']==src]) for src in src_uids])
        Xres = Xres.assign(out_deg=[rels[rels.srcuid==src]['DEG'].abs().sum() for src in src_uids])

        self.result = {
            'Noise': Noiseres,
            'X': Xres,
            'T': Tres,
            'S': Sres,
        }


class ORNORModel_1(BaseModel):


    def generate_vars(self):

        rels = self.rels
        DEG = self.DEG

        rels = rels.assign(DEG=[DEG[trg] for trg in rels['trguid']])

        # identify TF considered in current network
        X_list = rels['srcuid'].unique()

        X = pd.Series(X_list, name='srcuid').to_frame()
        X = X.assign(out_total=[len(rels[rels['srcuid']==src]) for src in X_list])
        X = X.assign(out_deg=[rels[rels.srcuid==src]['DEG'].abs().sum() for src in X_list])
        X = X.set_index('srcuid')
        X = X.assign(prior0=X.apply(lambda r: 0.95 - r['out_deg'] / r['out_total'] * 0.8, axis=1))

        noiseNode = Noise('Noise', None, a=0.005, b=0.0005)

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
            X0prior = X.loc[src]['prior0']
            X1prior = 1. - X0prior
            Xnodes[src] = Multinomial('X', src, np.array([X0prior, X1prior]))
            Tnodes[src] = Beta('T', src, 20, 1, scale=0.1)

        Snodes = {}
        for edg in rels.index:

            Tnodes[edg] = Beta('T', edg, 2, 2, scale=0.1)

            src, trg = edg
            
            if Ynodes[trg].value[1]:
                Sprior = np.array([0.2, 0.6, 0.2])
            else:
                Sprior = np.array([0.4, 0.2, 0.4])
            
            Snodes[edg] = Multinomial('S', edg, Sprior)
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


    def update_result(self, Xgt=None):

        result = self.get_trace_stats(combine=True).drop(columns=['N', 'sum1', 'sum2', 'var'])

        ents = self.ents
        rels = self.rels
        DEG = self.DEG
        rels = rels.assign(DEG=[DEG[trg] for trg in rels['trguid']])
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
        Tres = Tres.assign(srcuid=list(rels['srcuid']))
        Tres = Tres.assign(trguid=list(rels['trguid']))

        Sres0 = result.loc[[f'S__{edge}_0' for edge in rels.index]]['mean'].to_frame('S-')
        Sres0 = Sres0.assign(edge=rels.index)
        Sres0 = Sres0.set_index('edge')
        Sres2 = result.loc[[f'S__{edge}_2' for edge in rels.index]]['mean'].to_frame('S+')
        Sres2 = Sres2.assign(edge=rels.index)
        Sres2 = Sres2.set_index('edge')
        SresT = result.loc[[f'T__{edge}' for edge in rels.index]]['mean'].to_frame('T')
        SresT = SresT.assign(edge=rels.index)
        SresT = SresT.set_index('edge')

        Sres = pd.concat([Sres0, Sres2, SresT], sort=False, axis=1)
        Sres = Sres.assign(srcuid=list(rels['srcuid']))
        Sres = Sres.assign(trguid=list(rels['trguid']))
        Sres = Sres.assign(pred=Sres.apply(lambda r: 1 if r['S+']>0.5 else (-1 if r['S-']>0.5 else 0), axis=1))
        deg = pd.Series(self.DEG)
        Sres.assign(DEG=list(deg.loc[Sres['trguid']]))
        try:
            Sres = Sres.assign(gt=list(rels['val']))
        except KeyError:
            pass

        if ents is not None:
            Xres = Xres.assign(name=list(ents.loc[Xres.srcuid].name))
            Tres = Tres.assign(srcname=list(ents.loc[Tres.srcuid].name))
            Tres = Tres.assign(trgname=list(ents.loc[Tres.trguid].name))
            Sres = Sres.assign(srcname=list(ents.loc[Sres.srcuid].name))
            Sres = Sres.assign(trgname=list(ents.loc[Sres.trguid].name))
        Sres = Sres.assign(DEG=list(rels['DEG']))
        Xres = Xres.assign(out_total=[len(rels[rels['srcuid']==src]) for src in src_uids])
        Xres = Xres.assign(out_deg=[rels[rels.srcuid==src]['DEG'].abs().sum() for src in src_uids])

        self.result = {
            'Noise': Noiseres,
            'X': Xres,
            'T': Tres,
            'S': Sres,
        }


class ORNORModel(ORNORModel_0):
    pass
