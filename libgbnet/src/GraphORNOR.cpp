#include <math.h>

#include "GraphORNOR.h"


namespace gbn
{

    GraphORNOR::GraphORNOR(unsigned int seed) : GraphBase(seed) {}
    GraphORNOR::~GraphORNOR() {}

    void GraphORNOR::build_structure (
        network_t interaction_network, 
        evidence_dict_t evidence,
        prior_active_tf_set_t active_tf_set,
        bool noise_listen_children,
        const double SPRIOR[3 * 3],
        double z_alpha, double z_beta,
        double z0_alpha, double z0_beta,
        double t_alpha, double t_beta,
        bool comp_yprob, bool const_params,
        double t_focus, double t_lmargin, double t_hmargin, double zn_focus, double zn_lmargin, double zn_hmargin
    ) {
        std::copy(SPRIOR, SPRIOR + 9, &this->SPROB[0][0]);

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
        TNode * T;
        HNodeORNOR * H;
        YDataNode * Y;
        SNode * S;

        // collections are a wrapper for a dictionary
        NodeDictionary x_dictionary = NodeDictionary();
        NodeDictionary t_dictionary = NodeDictionary();
        NodeDictionary h_dictionary = NodeDictionary();

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
        

        // Noise node: 3 rv Dirichlet K=3
        // YNoiseNode * Noise[3];
        // for (unsigned int i = 0; i < 3; i++) {
        //     Noise[i] = new YNoiseNode(i, this->YNOISE_ALPHA[i], this->rng);
        //     this->norand_nodes.push_back(Noise[i]);
        //     Noise[i]->listen_children = noise_listen_children;
        // }
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
                // H->tvalue = &T->value;
                // H->zvalue = &ZY->value;
                Y = new YDataNode(H);
                for (unsigned int i = 0; i < 3; i++)
                    Y->noise[i] = this->YNOISE[i];

                H->is_latent = false;
                // ZY->append_h_child(H);
                h_dictionary.include_node(H);
                this->random_nodes.push_back(H);
                this->random_nodes.push_back(Y);
            }
        } else {
            // evidence available

            evidence_dict_t::iterator itr;
            unsigned int deg;

            if (comp_yprob) {
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
            }

            // Finally initialize Y and H nodes
            for (auto uid: trg_uid) {
                itr = evidence.find(uid);
                if (itr != evidence.end())
                    deg = (unsigned int) (itr->second + 1);
                else
                    deg = 1; // 1 -> not DEG

                Y = new YDataNode(uid, this->YPROB, deg);
                for (unsigned int i = 0; i < 3; i++)
                    Y->noise[i] = this->YNOISE[i];

                H = new HNodeORNOR(Y);
                H->is_latent = true;
                // H->tvalue = &T->value;
                // if (deg == 1) {
                //     H->zvalue = &Z0->value;
                //     Z0->append_h_child(H);
                // } else {
                //     H->zvalue = &ZY->value;
                //     ZY->append_h_child(H);
                // }
                h_dictionary.include_node(H);
                this->norand_nodes.push_back(H);
                this->norand_nodes.push_back(Y);
            }
        }

        double z_mean = z_alpha / (z_alpha + z_beta);
        ZNode * ZY = new ZNode((std::string) "Y", z_alpha, z_beta, z_mean);
        ZY->listen_children = false;

        double z0_mean = z0_alpha / (z0_alpha + z0_beta);
        ZNode * ZN = new ZNode((std::string) "N", z0_alpha, z0_beta, z0_mean);
        ZN->listen_children = false;

        if (const_params)
        {
            this->norand_nodes.push_back(ZN);
            this->norand_nodes.push_back(ZY);
        } else {
            this->random_nodes.push_back(ZN);
            this->random_nodes.push_back(ZY);
        }

        const unsigned int n_zt_groups = 8;

        // ZN nodes
        // ZNode * ZNset[n_zt_groups];

        // Theta
        // TNode * Tset[n_zt_groups];

        // for (unsigned int i = 0; i < n_zt_groups; i++) {
        //     t_beta = (double) (i * t_focus + t_hmargin);
        //     t_alpha = (double) (n_zt_groups - i - 1) * t_focus + t_lmargin;
        //     double t_mean = t_alpha / (t_alpha + t_beta);
        //     Tset[i] = new TNode(std::to_string(i), t_alpha, t_beta, t_mean);
        //     Tset[i]->listen_children = noise_listen_children;

        //     // // double z0_alpha = (double) (i * zn_focus + zn_lmargin);
        //     // // double z0_beta = (double) (n_zt_groups - i - 1) * zn_focus + zn_hmargin;
        //     // double zn_mean = z0_alpha / (z0_alpha + z0_beta);
        //     // ZNset[i] = new ZNode(std::to_string(i), z0_alpha, z0_beta, zn_mean);
        //     // ZNset[i]->listen_children = noise_listen_children;

        //     if (const_params) {
        //         this->norand_nodes.push_back(Tset[i]);
        //         // this->norand_nodes.push_back(ZNset[i]);
        //     } else {
        //         this->random_nodes.push_back(Tset[i]);
        //         // this->random_nodes.push_back(ZNset[i]);
        //     }
        // }

        // Determine the number of targets for each TF
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            X = (XNode *) x_dictionary.find_node(src);
            X->n_h_child++;
        }

        for (auto& dict_item: x_dictionary.dictionary) {
            X = (XNode *) dict_item.second;
            double lognchild = log(X->n_h_child);

            t_alpha = t_lmargin + (n_zt_groups - lognchild - 1) * t_focus;
            t_beta = t_hmargin + lognchild * t_focus;
            // t_alpha = t_lmargin;
            // t_beta = t_hmargin + X->n_h_child * t_focus;

            double t_mean = t_alpha / (t_alpha + t_beta);
            T = new TNode(X->uid, t_alpha, t_beta, t_mean);
            T->listen_children = noise_listen_children;

            if (const_params) {
                this->norand_nodes.push_back(T);
            } else {
                this->random_nodes.push_back(T);
            }

            t_dictionary.include_node(T);
        }

        ZNode * Z;
        // Nodes for interaction S nodes 
        for (auto edge: interaction_network) {
            tie(src_trg_pair, mor) = edge;
            tie(src, trg) = src_trg_pair;
            X = (XNode *) x_dictionary.find_node(src);
            T = (TNode *) t_dictionary.find_node(src);
            H = (HNodeORNOR *) h_dictionary.find_node(trg);
            std::string s_id = X->uid + "-->" + H->uid;

            unsigned int mor_idx = (unsigned int) (mor + 1);
            S = new SNode(s_id, this->SPROB[mor_idx], mor_idx);
            if (is_simulation) {
                this->norand_nodes.push_back(S);
            } else {
                this->random_nodes.push_back(S);
            }

            if (X->n_h_child == 0) throw std::runtime_error("Included TF with no targets");
            unsigned int idx = (unsigned int) log(X->n_h_child);
            if (idx >= n_zt_groups) idx = n_zt_groups - 1;

            if (H->data->value != 1)
                Z = ZY;
            else
                Z = ZN;
                // Z = ZNset[idx];

            H->append_parent(Z, T, X, S);
        }
    }
}
