import numpy as np
import pandas as pd
from .basemodel import BaseModel
from .cnodes import Beta, RandomVariableNode, Noise
from .cnodes import Multinomial, ORNOR_YLikelihood


class ORNORModel(BaseModel):


    def generate_vars(self):

        rels = self.rels
        DEG = self.DEG

        rels = rels.assign(DEG=[DEG[trg] for trg in rels['trguid']])

        # identify TF considered in current network
        X_list = rels['srcuid'].unique()

        noiseNode = Noise('Noise', None, a=0.005, b=0.0005)

        Znodes = {}

        mean = 0.0001
        Zprior_a, Zprior_b = 1/(1-mean),1/mean
        Znodes[0] = Beta('Z', 0, Zprior_a, Zprior_b, value=mean)

        mean = 0.9999
        Zprior_a, Zprior_b = 1/(1-mean),1/mean
        Znodes[1] = Beta('Z', 1, Zprior_a, Zprior_b, value=mean)

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
            try:
                X1prior = self.xpriors[src]
                X0prior = 1. - X1prior
            except KeyError:
                X0prior = 0.80
                X1prior = 1. - X0prior
            Xnodes[src] = Multinomial('X', src, np.array([X0prior, X1prior]))

            try:
                Tprior_a, Tprior_b = self.tpriors[src]
                Tprior_a, Tprior_b = min(Tprior_a, 20), min(Tprior_b, 20)
            except KeyError:
                mean = 0.90
                Tprior_a, Tprior_b = 1/(1-mean),1/mean
            Tnodes[src] = Beta('T', src, Tprior_a, Tprior_b, value=mean)

        Snodes = {}
        for edg, rel in rels.iterrows():
            
            src, trg = edg

            if Ynodes[trg].value[1]:
                # target not diff exp, less likely to be regulated?
                # Sprior = np.array([0.0005, 0.9990, 0.0005])
                Znod = Znodes[0]
            else:
                # Sprior = np.array([0.005, 0.990, 0.005])
                Znod = Znodes[1]

            Sprior = np.array([0.2, 0.6, 0.2])
            try:
                reltype = rel['type']
            except KeyError:
                reltype = 'unknown'

            if reltype != 'unknown':
                try:
                    if rel['is_trrust']:
                        if rel['type'] == 'increase':
                            Sprior = np.array([0.0, 0.4, 0.6])
                        elif rel['type'] == 'decrease':
                            Sprior = np.array([0.6, 0.4, 0.0])
                        else:
                            Sprior = np.array([0.3, 0.4, 0.3])
                    elif rel['is_chipatlas']:
                        Sprior = np.array([0.2, 0.6, 0.2])
                except KeyError:
                    pass
            
            try:
                # overwrite S prior if one was provided
                priorA = rel['prior']
                if priorA > 0:
                    priorI = 1. - priorA

                    Sprior = np.array([priorA/2., priorI, priorA/2.])

            except KeyError:
                pass

            Snodes[edg] = Multinomial('S', edg, Sprior)
            Snodes[edg].children.append(Ynodes[trg])
            
            Xnodes[src].children.append(Ynodes[trg])
            Tnodes[src].children.append(Ynodes[trg])
            Znod.children.append(Ynodes[trg])

            Ynodes[trg].set_Znode(Znod)
            
            Ynodes[trg].in_edges.append([Xnodes[src], Tnodes[src], Snodes[edg]])

        self.vars['Noise'] = {0: noiseNode}
        self.vars['T'] = Tnodes
        self.vars['X'] = Xnodes
        self.vars['S'] = Snodes
        self.vars['Z'] = Znodes


    def update_result(self, Xgt=None):

        result = self.get_trace_stats(combine=True).drop(columns=['N', 'sum1', 'sum2', 'var'])

        ents = self.ents
        rels = self.rels
        DEG = self.DEG
        rels = rels.assign(DEG=[DEG[trg] for trg in rels['trguid']])
        src_uids = rels.srcuid.unique()

        Noiseres = result.loc[[f'Noise_{i}' for i in [0,1]]]
        Zres = result.loc[[f'Z__{i}' for i in [0,1]]]

        Tres = result.loc[[f'T__{src}' for src in src_uids]]
        Tres = Tres.assign(srcuid=src_uids)
        Tres = Tres.set_index('srcuid')

        Xres = result.loc[[f'X__{src}_1' for src in src_uids]]['mean'].to_frame('X')
        Xres = Xres.assign(srcuid=src_uids)
        Xres = Xres.set_index('srcuid')
        Xres = Xres.assign(T=Tres['mean'])
        Xres['XT'] = Xres.apply(lambda r: r['X']*r['T'], axis =1)
        Xres['pred'] = Xres.apply(lambda r: 1 if r['X']>0.5 else 0, axis =1)
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
        Xres = Xres.sort_values(by='XT', ascending=False)

        self.result = {
            'Noise': Noiseres,
            'Z': Zres,
            'X': Xres,
            'T': Tres,
            'S': Sres,
        }

