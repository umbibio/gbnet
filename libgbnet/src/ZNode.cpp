#include "ZNode.h"

namespace gbn
{
    ZNode::~ZNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  ZNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    ZNode::ZNode (std::string uid, double a, double b, gsl_rng * rng)
    : Beta ((std::string) "Z", uid, a, b, rng) {}
    ZNode::ZNode (std::string uid, double a, double b, double value)
    : Beta ((std::string) "Z", uid, a, b, value) {}

    double ZNode::get_children_loglikelihood ()
    {
        double loglik = 0.;

        if (this->listen_children)
            for (auto child: this->children)
                loglik += child->get_own_loglikelihood();

        return loglik;
    }
}
