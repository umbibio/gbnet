# distutils: language = c++

from ModelORNOR cimport *
import pandas as pd
# from cysignals.signals cimport sig_check

cdef class PyModelORNOR:
    cdef ModelORNOR *c_model  # Hold a C++ instance which we're wrapping

    def __cinit__(self, ents, rels, evid = None, set active_tf_set_py = set(), unsigned int n_graphs = 3):

        # Make sure the indices are appropriate
        assertion_msg = ("Please make sure that provided ents file is a "
                         "Pandas Dataframe with column 'uid' available")
        # for ents
        ents = ents[["uid", "name"]].dropna()
        if 'uid' in ents.columns:
            ents = ents.set_index('uid')
        assert ents.index.name == 'uid', assertion_msg

        # now build network
        rels["srcsymbol"] = ents.loc[rels.srcuid, "name"].values
        rels["trgsymbol"] = ents.loc[rels.trguid, "name"].values
        src_set = set(rels["srcsymbol"].values)
        trg_set = set(rels["trgsymbol"].values)
        network_py = {(src, trg):mor for src, trg, mor in rels[["srcsymbol", "trgsymbol", "type"]].values}

        # for DEG
        assertion_msg = ("Please make sure that provided DEG file is either a "
                         "dictionary or a Pandas Series with uids as keys/index")
        if isinstance(evid, pd.DataFrame):
            assert 'uid' in evid.columns, assertion_msg
            evid = evid[["uid", "val"]].dropna()
            evid["symbol"] = ents.loc[evid.uid, "name"].values
            evid = evid[evid["symbol"].isin(trg_set)]
            evidence_py = dict(zip(evid["symbol"].values, evid["val"].values))
        elif evid is None:
            evidence_py = {}
        else:
            raise TypeError("Input DEG evidence must be a DataFrame object")

        cdef src_trg_pair_t src_trg
        cdef network_edge_t network_edge
        cdef network_t network = network_t()
        for (src_uid, trg_uid), mor in network_py.items():
            src_trg = src_uid.encode('utf8'), trg_uid.encode('utf8')
            network_edge = src_trg, mor
            network.push_back( network_edge )

        cdef evidence_dict_t evidence = evidence_dict_t()
        for trg_uid, trg_de in evidence_py.items():
            evidence.insert((trg_uid.encode('utf8'), trg_de))

        cdef prior_active_tf_set_t active_tf_set = prior_active_tf_set_t()
        for src_uid in active_tf_set_py:
            active_tf_set.insert(src_uid.encode('utf8'))

        if len(evidence_py) > 0 and len(active_tf_set_py) > 0:
            self.c_model = new ModelORNOR(network, evidence, active_tf_set, n_graphs)
        elif len(evidence_py) > 0:
            self.c_model = new ModelORNOR(network, evidence, n_graphs)
        elif len(active_tf_set_py) > 0:
            self.c_model = new ModelORNOR(network, active_tf_set, n_graphs)

    def get_gelman_rubin(self):
        gr_list = self.c_model.get_gelman_rubin()
        return [id.decode('utf8').split('_')[:2]+[gr] for id, gr in gr_list]

    def get_max_gelman_rubin(self):
        return self.c_model.get_max_gelman_rubin()

    def sample(self, unsigned int N = 50000, unsigned int deltaN = 30):
        self.c_model.sample(N, deltaN)
        return

        # # TODO: KeyboardInterrupt is not working

        # n = 0
        # gr = float('inf')
        # while n < N and gr > 1.10:
        #     try:
        #         self.c_model.sample_n(deltaN)
        #         deltaN = min(deltaN, N-n)
        #         n += deltaN
        #         gr = self.c_model.get_max_gelman_rubin()
        #         # sig_check()

        #     except KeyboardInterrupt:
        #         print("\n\nCaught KeyboardInterrupt, workers have been terminated\n")

        # print("Drawed", n, "samples")
        # print("Max Gelman-Rubin statistics is", gr)

    def print_stats(self):
        self.c_model.print_stats()

    def burn_stats(self):
        self.c_model.burn_stats()

    def get_posterior_means(self, var_name):
        posterior_stat = self.c_model.get_posterior_means(var_name.encode('utf8'))
        return [id.decode('utf8').split('_')+[stat] for id, stat in posterior_stat]

    def get_posterior_sdevs(self, var_name):
        posterior_stat = self.c_model.get_posterior_sdevs(var_name.encode('utf8'))
        return [id.decode('utf8').split('_')+[stat] for id, stat in posterior_stat]

    def __dealloc__(self):
        if self.c_model is not NULL:
            del self.c_model
