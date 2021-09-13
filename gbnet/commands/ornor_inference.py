import argparse
import pandas as pd

from gbnet.utils import get_tests, get_model, sample_model


def main():
    parser = argparse.ArgumentParser(description='Specify parameters.')

    parser.add_argument('--ents', type=str, required=True)
    parser.add_argument('--rels', type=str, required=True)
    parser.add_argument('--evid', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)

    parser.add_argument('--const_params', action='store_true')
    parser.add_argument('--noise_listen_children', action='store_true')
    parser.add_argument('--comp_yprob', action='store_true')

    parser.add_argument('--t_focus', type=float, default=1.)
    parser.add_argument('--t_lmargin', type=float, default=2.)
    parser.add_argument('--t_hmargin', type=float, default=2.)
    parser.add_argument('--z_alpha', type=float, default=2000.)
    parser.add_argument('--z_beta', type=float, default=1.)
    parser.add_argument('--z0_alpha', type=float, default=200.)
    parser.add_argument('--z0_beta', type=float, default=1.)
    parser.add_argument('--s_leniency', type=float, default=0.01)

    parser.add_argument('--max_its', type=int, default=1000)

    params = vars(parser.parse_args())

    ents = pd.read_csv(params['ents'])
    rels = pd.read_csv(params['rels'])
    evid = pd.read_csv(params['evid'])
    out_path = params['out']
    del(params['ents'], params['rels'], params['evid'], params['out'])

    max_its = params['max_its']
    l = params['s_leniency']
    params['sprior'] = [
        [1.0 - l, 0.9 * l, 0.1 * l],
        [0.5 * l, 1.0 - l, 0.5 * l],
        [0.1 * l, 0.9 * l, 1.0 - l],
    ]
    del(params['max_its'], params['s_leniency'])

    tests = get_tests(evid, rels)
    model = get_model(ents, rels, evid, **params)
    result = sample_model(model, ents, tests, burn_its=5, max_its=max_its, verbosity=1)

    result.to_csv(out_path)
