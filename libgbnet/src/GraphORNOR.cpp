#include "GraphORNOR.h"


namespace gbn
{

    GraphORNOR::GraphORNOR(unsigned int seed)
    : GraphBase(seed) {}

    GraphORNOR::~GraphORNOR()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  GraphORNOR\t" << std::endl;
    }

    void GraphORNOR::build_structure (
        network_t interaction_network, 
        evidence_dict_t evidence,
        prior_active_tf_set_t active_tf_set,
        bool noise_listen_children
    ) {
        // This means that evidence will be sampled from the graph, given the set of active TF
        bool is_simulation = evidence.size() == 0;

        // first, find unique src tfs and trg genes
        std::set<std::string> src_uid, trg_uid;
        std::string src, trg;
        src_trg_pair_t src_trg_pair;
        int mor;
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            src_uid.insert(src);
            trg_uid.insert(trg);
            if ( mor != -1 && mor != 0 && mor != 1) 
                throw std::out_of_range("MOR values can only be either -1, 0 or 1");
        }
        
        // temp variable pointers to create and manipulate nodes
        XNode * X;
        HNodeORNOR * H;
        YDataNode * Y;
        SNode * S;

        // collections are a wrapper for a dictionary
        NodeDictionary x_dictionary = NodeDictionary();
        NodeDictionary h_dictionary = NodeDictionary();

        // Theta
        TNode * T = new TNode((std::string) "", 50., 2., this->rng);
        T->listen_children = noise_listen_children;
        this->random_nodes.push_back(T);

        // One X for each TF
        for (auto uid: src_uid) {
            if (is_simulation) {
                if (active_tf_set.count(uid))
                    X = new XNode(uid, this->XPROB_ACTIVE, (unsigned int) 1);
                else
                    X = new XNode(uid, this->XPROB, (unsigned int) 0);
                this->norand_nodes.push_back(X);
            } else {
                if (active_tf_set.count(uid))
                    X = new XNode(uid, this->XPROB_ACTIVE, this->rng);
                else
                    X = new XNode(uid, this->XPROB, this->rng);
                this->random_nodes.push_back(X);
            }

            x_dictionary.include_node(X);
        }

        ZNode * Z0 = new ZNode((std::string) "0", 1., 200., 0.005);
        Z0->listen_children = noise_listen_children;
        this->random_nodes.push_back(Z0);
        ZNode * Z1 = new ZNode((std::string) "1", 200., 1., 0.995);
        Z1->listen_children = noise_listen_children;
        this->random_nodes.push_back(Z1);

        // Noise node: 3 rv Dirichlet K=3
        YNoiseNode * Noise[3];
        for (unsigned int i = 0; i < 3; i++) {
            Noise[i] = new YNoiseNode(i, this->YNOISE_ALPHA[i], this->rng);
            this->random_nodes.push_back(Noise[i]);
            Noise[i]->listen_children = noise_listen_children;
        }
        /* After testing this out it seems like this Noise node yields too easily, and shifts
        its values towards high error rates, crippling the effect of the observed data
        on the graphical model. Hence I am choosing to make this a hyperparameter, that doesn't
        take children likelihood for sampling */

        // The observed data and its corresponding hidden true state
        if (is_simulation) {
            // no evidence provided. Treat H and Y as random variables to sample

            for (auto uid: trg_uid) {
                // start out with no deg, so first samples for X are not bumped up
                H = new HNodeORNOR(uid, this->YPROB, 1);
                H->tvalue = &T->value;
                H->zvalue = &Z1->value;
                Y = new YDataNode(H);
                for (unsigned int i = 0; i < 3; i++) {
                    Y->noise[i] = Noise[i]->value;
                    Noise[i]->children.push_back(Y);
                }
                // Y->noise_value = this->YNOISE; // use this when skipping Noise nodes

                H->is_latent = false;
                Z1->append_h_child(H);
                h_dictionary.include_node(H);
                this->random_nodes.push_back(H);
                this->random_nodes.push_back(Y);
            }
        } else {
            // evidence available

            evidence_dict_t::iterator itr;
            unsigned int deg;

            // First compute deg proportions in data
            unsigned int deg_counts[3] = {0};
            for (itr = evidence.begin(); itr != evidence.end(); ++itr) {
                deg = (unsigned int) (itr->second + 1);
                deg_counts[deg] += 1;
            }
            // Now use that for Y and H nodes prior probabilities
            // take into account all target genes present in network
            this->YPROB[0] = (double) deg_counts[0] / trg_uid.size();
            this->YPROB[2] = (double) deg_counts[2] / trg_uid.size();
            this->YPROB[1] = (double) 1. - this->YPROB[0] - this->YPROB[2];

            // Finally initialize Y and H nodes
            for (auto uid: trg_uid) {
                itr = evidence.find(uid);
                if (itr != evidence.end())
                    deg = (unsigned int) (itr->second + 1);
                else
                    deg = 1; // 1 -> not DEG

                Y = new YDataNode(uid, this->YPROB, deg);
                for (unsigned int i = 0; i < 3; i++) {
                    Y->noise[i] = Noise[i]->value;
                    Noise[i]->children.push_back(Y);
                }
                // Y->noise_value = this->YNOISE; // use this when skipping Noise nodes

                H = new HNodeORNOR(Y);
                H->is_latent = true;
                H->tvalue = &T->value;
                if (deg == 1) {
                    H->zvalue = &Z0->value;
                    Z0->append_h_child(H);
                } else {
                    H->zvalue = &Z1->value;
                    Z1->append_h_child(H);
                }
                h_dictionary.include_node(H);
                this->norand_nodes.push_back(H);
                this->norand_nodes.push_back(Y);
            }
        }

        // Nodes for interaction S nodes 
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            X = (XNode *) x_dictionary.find_node(src);
            H = (HNodeORNOR *) h_dictionary.find_node(trg);
            std::string s_id = X->uid + "-->" + H->uid;

            unsigned int mor_idx = (unsigned int) (mor + 1);
            if (is_simulation) {
                S = new SNode(s_id, this->SPROB[mor_idx], mor_idx);
                this->norand_nodes.push_back(S);
            } else {
                S = new SNode(s_id, this->SPROB[mor_idx], this->rng);
                this->random_nodes.push_back(S);
            }

            H->append_parent(X, S);
        }
    }
}
