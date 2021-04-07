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
        const double SPRIOR[3 * 3],
        unsigned int n_graphs, bool noise_listen_children
    ) : ModelBase(n_graphs) {
        this->network = network;
        this->evidence = evidence;
        this->active_tf_set = active_tf_set;
        this->noise_listen_children = noise_listen_children;
        for (unsigned int i = 0; i < 9; i++) this->sprior[i] = SPRIOR[i];
        this->build_graphs();
    }


    void ModelORNOR::build_graphs()
    {
        unsigned int seed;
        GraphORNOR * graph;

        for (unsigned int i = 0; i < this->n_graphs; i++) {
            seed = (i + 1) * 42 + time(NULL);
            graph = new GraphORNOR(seed);
            graph->build_structure(this->network, this->evidence, this->active_tf_set, this->noise_listen_children, this->sprior);
            this->graphs.push_back(graph);
        }
    }
}