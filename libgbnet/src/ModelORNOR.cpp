#include "ModelORNOR.h"


namespace gbn
{
    ModelORNOR::ModelORNOR() : ModelBase() {}
    ModelORNOR::~ModelORNOR() {
        for (auto graph: this->graphs)
            delete graph;
        // std::cout << "Object at " << this << " destroyed, instance of  ModelORNOR\t" << std::endl;
    }

    ModelORNOR::ModelORNOR(
        const network_t network, const evidence_dict_t evidence, const prior_active_tf_set_t active_tf_set,
        const double SPRIOR[3 * 3], double z_alpha, double z_beta, double z0_alpha, double z0_beta, double t_alpha, double t_beta,
        unsigned int n_graphs, bool noise_listen_children, bool comp_yprob, bool const_params,
        double t_focus, double t_lmargin, double t_hmargin, double zn_focus, double zn_lmargin, double zn_hmargin
    ) : ModelBase(n_graphs) {

        unsigned int seed;
        GraphORNOR * graph;

        for (unsigned int i = 0; i < this->n_graphs; i++) {
            seed = (i + 1) * 42 + time(NULL);
            graph = new GraphORNOR(seed);
            graph->build_structure(network, evidence, active_tf_set, noise_listen_children, SPRIOR, z_alpha, z_beta, z0_alpha, z0_beta, t_alpha, t_beta, comp_yprob, const_params, t_focus, t_lmargin, t_hmargin, zn_focus, zn_lmargin, zn_hmargin);
            this->graphs.push_back(graph);
        }
    }
}