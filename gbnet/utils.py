from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact
from gbnet import ModelORNOR


def get_tests(evid, rels):
    evid = evid.reset_index().set_index('uid')

    tmp = rels.merge(evid.loc[:, ['val']], how='outer', left_on='trguid', right_index=True)

    all_degp = len(evid.loc[evid.val == 1])    # number of positive deg
    all_degn = len(evid.loc[evid.val == -1])   # number of negative deg

    all_ydeg = all_degp + all_degn             # number of yes deg
    all_ndeg = len(evid.loc[evid.val == 0])    # number of not deg

    all_tot = all_ydeg + all_ndeg              # number of target genes

    # for enrichment
    trg_ydeg = tmp.loc[tmp.val != 0, ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('trg_ydeg')
    trg_ndeg = tmp.loc[tmp.val == 0, ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('trg_ndeg')

    # for mor binary and ternary test

    # mode of reg +1
    morp_degp = tmp.loc[(tmp.type ==  1) & (tmp.val ==  1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morp_degp')
    morp_degn = tmp.loc[(tmp.type ==  1) & (tmp.val == -1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morp_degn')
    # mode of reg -1
    morn_degp = tmp.loc[(tmp.type == -1) & (tmp.val ==  1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morn_degp')
    morn_degn = tmp.loc[(tmp.type == -1) & (tmp.val == -1), ['srcuid', 'trguid']].groupby('srcuid').count()['trguid'].rename('morn_degn')

    def enrichment_test(row):

        contingency_table =  [[row.trg_ydeg, all_ydeg - row.trg_ydeg],
                              [row.trg_ndeg, all_ndeg - row.trg_ndeg]]

        return fisher_exact(
           contingency_table,
            alternative = "greater")[1]

    def mor_binary_test(row):

        contingency_table = [[row.morp_degp, row.morn_degp],
                             [row.morp_degn, row.morn_degn]]

        return fisher_exact(
           contingency_table,
            alternative = "greater")[1]

    df = pd.DataFrame([trg_ydeg, trg_ndeg, morp_degp, morp_degn, morn_degp, morn_degn]).T.fillna(0).astype(int)
    df.index.rename('srcuid', True)
    df['enrichment'] = df.apply(enrichment_test, axis=1)
    df['mor_binary'] = df.apply(mor_binary_test, axis=1)

    rho = df.enrichment * df.mor_binary
    df['combined'] = rho.apply(lambda r: r * (1 - np.log(r)) if r > 0 else r)

    trg_tot = df.trg_ndeg + df.trg_ydeg
    tst_score = df.morp_degp + df.morn_degn - df.morp_degn - df.morn_degp
    df['trg_tot'] = trg_tot
    df['score'] = tst_score / df.trg_ydeg
    df['gt_act'] = False

    return df

def get_model(
    ents, rels, evid,
    const_params=False, noise_listen_children=False, comp_yprob=False,
    t_focus=2., t_lmargin=2., t_hmargin=2., zn_focus=2., zn_lmargin=8., zn_hmargin=2.,
    z_alpha=198, z_beta=2, z0_alpha=198, z0_beta=2, t_alpha=2, t_beta=2,
    sprior = [[ 0.99, 0.01,   0.0],
              [0.005, 0.99, 0.005],
              [  0.0, 0.01,  0.99]]
    ):
    
    return ModelORNOR(ents, rels, evid,
                      n_graphs=3, sprior=sprior,
                      z_alpha=z_alpha, z_beta=z_beta, z0_alpha=z0_alpha, z0_beta=z0_beta, t_alpha=t_alpha, t_beta=t_beta,
                      t_focus=t_focus, t_lmargin=t_lmargin, t_hmargin=t_hmargin, zn_focus=zn_focus, zn_lmargin=zn_lmargin, zn_hmargin=zn_hmargin,
                      noise_listen_children=noise_listen_children, comp_yprob=comp_yprob, const_params=const_params)

def sample_model(model, ents, tests, max_its=100, burn_its=3, verbosity=0):
    gt_act_src_uids = tests.loc[tests.sort_values('trg_tot').gt_act].index.tolist()
    gt_act_src_sym = ents.loc[ents.type == 'Protein'].set_index('uid').loc[gt_act_src_uids, 'name']

    it = 0
    for _ in range(burn_its):
        it += 1
        model.sample(20)
        if verbosity >= 1:
            mgr = model.get_max_gelman_rubin()
            df = pd.DataFrame([dict(symbol=symbol, val=val) for _, symbol, idx, val in model.get_posterior_means('X') if idx == '1']).set_index('symbol')
            gt_est = df.loc[gt_act_src_sym].val.values.round(2)
            max_val = df.val.max()
            params = np.array([v[-1] for v in model.get_posterior_means('Z')] + [v[-1] for v in model.get_posterior_means('T')])
            print(f"[{datetime.now()}] {mgr=: 10.4f}. {gt_est=}, {max_val=:.6f}, {params=}", flush=True)

    model.burn_stats()

    mgr = float('inf')
    while mgr > 1.1:
        it += 1
        model.sample(20)
        mgr = model.get_max_gelman_rubin()
        if verbosity >= 1:
            df = pd.DataFrame([dict(symbol=symbol, val=val) for _, symbol, idx, val in model.get_posterior_means('X') if idx == '1']).set_index('symbol')
            gt_est = df.loc[gt_act_src_sym].val.values.round(2)
            max_val = df.val.max()
            params = np.array([v[-1] for v in model.get_posterior_means('Z')] + [v[-1] for v in model.get_posterior_means('T')])
            print(f"[{datetime.now()}] {mgr=: 10.4f}. {gt_est=}, {max_val=:.6f}, {params=}", flush=True)
        if it >= max_its // 2:
            break

    model.burn_stats()

    mgr = float('inf')
    while mgr > 1.1:
        it += 1
        model.sample(20)
        mgr = model.get_max_gelman_rubin()
        if verbosity >= 1:
            df = pd.DataFrame([dict(symbol=symbol, val=val) for _, symbol, idx, val in model.get_posterior_means('X') if idx == '1']).set_index('symbol')
            gt_est = df.loc[gt_act_src_sym].val.values.round(2)
            max_val = df.val.max()
            params = np.array([v[-1] for v in model.get_posterior_means('Z')] + [v[-1] for v in model.get_posterior_means('T')])
            print(f"[{datetime.now()}] {mgr=: 10.4f}. {gt_est=}, {max_val=:.6f}, {params=}", flush=True)
        if it >= max_its:
            break

    df = pd.DataFrame([dict(symbol=symbol, val=val) for _, symbol, idx, val in model.get_posterior_means('X') if idx == '1']).set_index('symbol')
    df['uid'] = ents.loc[ents.type == 'Protein'].set_index('name').loc[df.index.values, 'uid']

    return tests.merge(df.reset_index().set_index('uid'), left_index=True, right_index=True).sort_values('val', ascending=False)
