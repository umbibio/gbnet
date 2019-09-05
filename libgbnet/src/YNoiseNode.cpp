#include "YNoiseNode.h"
#include "YDataNode.h"


namespace gbn
{
    YNoiseNode::~YNoiseNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  YNoiseNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    YNoiseNode:: YNoiseNode (gsl_rng * rng)
    : Dirichlet ((std::string) "Noise", (std::string) "", 3)
    {
        this->prior_alpha = this->noise_alpha;
        this->set_init_attr();
        this->sample_from_prior(rng);
    }

    double YNoiseNode::get_children_loglikelihood ()
    {
        double loglik = 0.;

        if (this->listen_children)
            for (auto child: this->children)
                loglik += child->get_own_loglikelihood();

        return loglik;
    }
}
