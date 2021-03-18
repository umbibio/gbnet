#ifndef GBNET_MODELBASE
#define GBNET_MODELBASE


#include <algorithm>
#include <cmath>
#include <csignal>
#include <time.h>

#include <gsl/gsl_rstat.h>


#include "GraphBase.h"


namespace gbn
{
    typedef std::pair<std::string, double> varid_stat_pair_t;
    typedef std::vector<varid_stat_pair_t> posterior_stat_vector_t;
    typedef std::vector<std::pair<std::string, double>> gelman_rubin_vector_t;

    class ModelBase
    {
        private:
        protected:
            int current_signal = 0;
        public:
            network_t network;
            evidence_dict_t evidence = evidence_dict_t();
            prior_active_tf_set_t active_tf_set = prior_active_tf_set_t();
            unsigned int n_graphs = 3;

            std::vector<GraphBase *> graphs;

            ModelBase(unsigned int = 3);
            virtual ~ModelBase();

            gelman_rubin_vector_t get_gelman_rubin();
            double get_max_gelman_rubin();

            void sample(unsigned int = 50000, unsigned int = 30);
            void sample_n(unsigned int);
            void burn_stats();

            void print_stats();

            posterior_stat_vector_t get_posterior_means(std::string);
            posterior_stat_vector_t get_posterior_sdevs(std::string);

            void set_signal(int);
    };
}

#endif