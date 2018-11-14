import os
import psutil
from datetime import timedelta
import time

import numpy as np
import pandas as pd
from num2words import num2words

def genData(NX=3, num_active_tfs=2, NY=50, AvgNTF=0.5):
    # Start generating simulated data
    # The TFs: Number of TF and activation state of each

    # the target genes: Number of target genes and state of each
    
    Xgt = dict(zip(list(range(NX)), [0]*NX))
    Y = dict(zip(list(range(NX,NX+NY)), [0]*NY))
    
    # generate a random set of interations between TFs and genes
    edges = {}
    for trg in Y.keys():
        
        # randomize the number of TFs for this gene
        num_edges = np.random.binomial(NX, 1.0/NX*AvgNTF)

        # pick random TFs
        for src in np.random.choice(list(Xgt.keys()), size=num_edges, replace=False):
            # here an edge can be upregulator (1), downregulator (-1) or not valid (0)
            edges[(src, trg)] = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

    # randomize current activation state for TFs
    # and then determine the state of targeted genes
    for src in np.random.choice(list(Xgt.keys()), size=num_active_tfs, replace=False):
        # either active (1) or inactive (0)
        Xgt[src] = 1

    for src in Xgt.keys():
        for trg in Y.keys():
            edge = src, trg
            if edge in edges.keys():
                if edges[edge] < 0 and Xgt[src] > 0:
                    # if TF is inhibitor, make sure the response is definitive
                    Y[trg] += edges[edge]*Xgt[src]*10000
                else:
                    Y[trg] += edges[edge]*Xgt[src]

    a = 0.02
    b = 0.005

    # get only the signs for gene activation states
    for trg in Y.keys():
        sgn = np.sign(Y[trg])
        if sgn == -1:
            Y[trg] = np.random.choice([-1, 0, 1], p=[1.-a-b, a, b])
        if sgn == 0:
            Y[trg] = np.random.choice([-1, 0, 1], p=[a, 1.-a-a, a])
        if sgn == 1:
            Y[trg] = np.random.choice([-1, 0, 1], p=[b, a, 1.-a-b])
        
    # this is the possible associations data
    rels = pd.DataFrame(list(edges.keys()), columns=['srcuid', 'trguid'])
    rels = rels.assign(val=list(edges.values()))

    # extract a dataframe that only contains relevant interactions
    rels = rels[rels['trguid'].isin(Y.keys())]
    rels = rels.assign(edge=[(src, trg) for src, trg in zip(rels['srcuid'], rels['trguid'])])
    rels = rels.assign(srcactive=[Xgt[src] for src in rels['srcuid']])
    rels = rels.set_index('edge')

    ents = pd.DataFrame([num2words(i) for i in range(NX+NY)], columns=['name'])

    return Xgt, ents, rels, Y

def processTrace(model, Xgt=None, rels=None):
    
    Dx, ADx, Ds, ADs, Dy = model.dictionaries
    del Dy

    Xres = model.get_result('X')
    Xres = Xres.assign(gelman_rubin=model.gelman_rubin['X'])
    Xres = Xres.assign(srcuid=[ADx[i] for i in range(len(Dx))])
    Xres = Xres.set_index('srcuid')
    if Xgt is not None:
        Xres = Xres.assign(ground_truth=[Xgt[src] for src in Xres.index])

    Rres = model.get_result('R')
    Rres = Rres.assign(gelman_rubin=model.gelman_rubin['R'])
    Rres = Rres.assign(edge=[ADs[i] for i in range(len(Ds))])
    Rres = Rres.set_index('edge')
    srcuid, trguid = zip(*Rres.index.tolist())
    Rres = Rres.assign(srcuid=srcuid)
    Rres = Rres.assign(trguid=trguid)
    Rres = Rres[Rres['srcuid'].isin(Xres.index)]
    if rels is not None:
        Rres = Rres.assign(ground_truth=rels.loc[Rres.index, 'val'].abs())

    Sres = model.get_result('S')
    Sres = Sres.assign(gelman_rubin=model.gelman_rubin['S'])
    Sres = Sres.assign(edge=[ADs[i] for i in range(len(Ds))])
    Sres = Sres.set_index('edge')
    srcuid, trguid = zip(*Sres.index.tolist())
    Sres = Sres.assign(srcuid=srcuid)
    Sres = Sres.assign(trguid=trguid)
    Sres = Sres[Sres['srcuid'].isin(Xres.index)]
    if rels is not None:
        Sres = Sres.assign(ground_truth=rels.loc[Sres.index, 'val'])

    Xres = Xres.sort_values(by=['mean'])
    Rres = Rres.sort_values(by=['mean'])
    Sres = Sres.sort_values(by=['mean'])
   
    return Xres, Rres, Sres
    

def updateRes(Xres, Rres, Sres, lenient=[False, False, False], final=False):
    
    lenientX, lenientR, lenientS = lenient

    if final:
        conditions = [Xres['mean'] < 0.45, Xres['mean'] >= 0.7]
    elif lenientX:
        conditions = [Xres['mean'] < 0.4, Xres['mean'] >= 0.8]
    else:
        conditions = [Xres['mean'] < 0.2, Xres['mean'] >= 0.8]
    choices = [0, 1]
    Xres = Xres.assign(pred=np.select(conditions, choices, default=-99))
    try:
        Xres = Xres.assign(correctness=Xres['ground_truth']==Xres['pred'])
    except KeyError:
        pass

    if final:
        conditions = [Rres['mean'] < 0.40, Rres['mean'] >= 0.50]
    elif lenientR:
        conditions = [Rres['mean'] < 0.35, Rres['mean'] >= 0.65]
    else:
        conditions = [Rres['mean'] < 0.30, Rres['mean'] >= 0.70]
    choices = [0, 1]
    Rres = Rres.assign(pred=np.select(conditions, choices, default=-99))
    try:
        Rres = Rres.assign(correctness=Rres['ground_truth']==Rres['pred'])
    except KeyError:
        pass
    Rres = Rres.assign(srcactive=Xres.loc[Rres.srcuid, 'pred'].tolist())
    Rres = Rres[Rres['srcactive']>0]

    if final:
        conditions = [Sres['mean'] < 0.5, Sres['mean'] >= 0.5]
    elif lenientS:
        conditions = [Sres['mean'] < 0.35, Sres['mean'] >= 0.65]
    else:
        conditions = [Sres['mean'] < 0.30, Sres['mean'] >= 0.70]
    choices = [-1, 1]
    Sres = Sres.assign(prediction=np.select(conditions, choices, default=0))
    choices = [0, 1]
    Sres = Sres.assign(pred=np.select(conditions, choices, default=-99))
    try:
        Sres = Sres.assign(correctness=Sres['ground_truth']==Sres['prediction'])
    except KeyError:
        pass
    Sres = Sres.assign(srcactive=Xres.loc[Sres.srcuid, 'pred'].tolist())
    Sres = Sres[Sres['srcactive']>0]
    Sres = Sres[Sres.index.isin(Rres.index)]

    if len(Sres) > 0:
        # I think this might break. Be careful when rows differ in both dfs
        Sres['applicable'] = Rres['pred']
        Sres = Sres[Sres['applicable']>0]

    if not final:
        Xres = Xres[Xres['pred']!=-99]
        Rres = Rres[Rres['pred']!=-99]
        Sres = Sres[Sres['prediction']!=0]
    
    return Xres, Rres, Sres


class Reporter(object):
    def __init__(self):
        self.ctime = time.time()
        self.stime = self.ctime
        self.ltime = self.ctime
        self.process = psutil.Process(os.getpid())
        self.last_report = ""

    def reset(self):
        self.ctime = time.time()
        self.stime = self.ctime

    def report(self, string='', schar='', lchar='\n', showram=True, showlast=True):
        self.ctime = time.time()
        total_dt = timedelta(seconds=round(self.ctime - self.stime))
        last_dt = timedelta(seconds=round(self.ctime - self.ltime))
        out = f'{schar}{string}'

        if string:
            out += ' -- '

        if showlast:
            out += f"Last: {last_dt}, "

        out += f"Elapsed: {total_dt}"
            
        if showram:
            usage = 0
            usage += self.process.memory_info().rss-self.process.memory_info().shared
            for child in self.process.children():
                usage += child.memory_info().rss-child.memory_info().shared
            usage = usage/1073741824
            out += f", Mem usage: {usage:0.02f}GB"
        
        print(out, end=lchar)
        self.last_report = out
        if showlast:
            self.ltime = self.ctime


def mutate_data(Y, factor):

    a = (1 - factor)/2
    b = a + 0.5

    origY = Y.copy()
    Y = origY.copy()

    for key, val in Y.items():
        Y[key] = int(round(val*np.random.uniform(a, b))) 

    ndiff = 0
    for key in origY.keys():
        ndiff += int(Y[key] != origY[key])
    print(f"{ndiff} entries have been mutated")

    return Y


def mutate_data2(Y, factor):

    a = (1 - factor)/2
    b = a + 0.5

    origY = Y.copy()
    Y = origY.copy()

    for key, val in Y.items():
        if val == 0 and np.random.uniform() < factor:
            Y[key] = int(round(np.random.choice([-1,1])))
        else:
            Y[key] = int(round(val*np.random.uniform(a, b)))

    ndiff = 0
    for key in origY.keys():
        ndiff += int(Y[key] != origY[key])
    print(f"{ndiff} entries have been mutated")

    return Y
