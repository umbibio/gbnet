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
        const network_t network, const evidence_dict_t evidence,
        unsigned int n_graphs, bool noise_listen_children
    ) : ModelBase() {
        this->network = network;
        this->evidence = evidence;
        this->build_graphs(noise_listen_children);
    }

    ModelORNOR::ModelORNOR(
        const network_t network, const prior_active_tf_set_t active_tf_set, 
        unsigned int n_graphs, bool noise_listen_children
    ) : ModelBase() {
        this->network = network;
        this->active_tf_set = active_tf_set;
        this->build_graphs(noise_listen_children);
    }

    ModelORNOR::ModelORNOR(
        const network_t network, const evidence_dict_t evidence, const prior_active_tf_set_t active_tf_set, 
        unsigned int n_graphs, bool noise_listen_children
    ) : ModelBase() {
        this->network = network;
        this->evidence = evidence;
        this->active_tf_set = active_tf_set;
        this->build_graphs(noise_listen_children);
    }


    void ModelORNOR::build_graphs(bool noise_listen_children)
    {
        unsigned int seed;
        GraphORNOR * graph;

        for (unsigned int i = 0; i < this->n_graphs; i++) {
            seed = (i + 1) * 42 + time(NULL);
            graph = new GraphORNOR(seed);
            graph->build_structure(this->network, this->evidence, this->active_tf_set, noise_listen_children);
            this->graphs.push_back(graph);
        }
    }
}