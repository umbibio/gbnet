from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string
from libcpp.pair cimport pair as std_pair
from libcpp.map cimport map as std_map
from libcpp.set cimport set as std_set


ctypedef std_map[std_string, int] evidence_dict_t
ctypedef std_pair[std_string, std_string] src_trg_pair_t
ctypedef std_pair[src_trg_pair_t, int] network_edge_t
ctypedef std_vector[network_edge_t] network_t

ctypedef std_set[std_string] prior_active_tf_set_t

ctypedef std_pair[std_string, double] varid_stat_pair_t
ctypedef std_vector[varid_stat_pair_t] posterior_stat_vector_t

ctypedef std_vector[std_pair[std_string, double]] gelman_rubin_vector_t

# cdef extern from "src/ModelORNOR.cpp":
#     pass

# Declare the class with cdef
cdef extern from "../libgbnet/include/ModelORNOR.h" namespace "gbn":

    cdef cppclass ModelORNOR:
        ModelORNOR() except +
        ModelORNOR( const network_t, const evidence_dict_t, unsigned int, bint) except +
        ModelORNOR( const network_t, const prior_active_tf_set_t, unsigned int, bint) except +
        ModelORNOR( const network_t, const evidence_dict_t, const prior_active_tf_set_t, unsigned int, bint) except +
        gelman_rubin_vector_t get_gelman_rubin()
        double get_max_gelman_rubin()
        void sample(unsigned int, unsigned int)
        void sample_n(unsigned int)
        void burn_stats()
        void print_stats()
        posterior_stat_vector_t get_posterior_means(std_string)
        posterior_stat_vector_t get_posterior_sdevs(std_string)
        void set_signal(int)
