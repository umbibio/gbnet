#include "HNode.h"


#include "YDataNode.h"


namespace gbn
{
    HNode::~HNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  HNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    HNode::HNode(YDataNode * Y) 
    : Multinomial ((std::string) "H", Y->uid, Y->n_outcomes, Y->prob, Y->value)
    {
        this->data = Y;
        Y->hidden_truth = this;
    }

    HNode::HNode ( std::string uid, const double * prob, gsl_rng * rng )
    : Multinomial ((std::string) "H", uid, 3, prob, rng) {}

    HNode::HNode ( std::string uid, const double * prob, unsigned int value )
    : Multinomial ((std::string) "H", uid, 3, prob, value) {}

    double HNode::get_model_likelihood ()
    {
        // virtual method, must be overriden in derived class
        return 0.;
    }

    double HNode::get_own_likelihood ()
    {
        double likelihood;

        if (this->is_latent) {
            likelihood = 0.;
            for (unsigned int i = 0; i < this->n_outcomes; i++)
            {
                // iterate over possible values to compute 
                // make sure we preserve current value by the end of this loop
                this->value = i;
                likelihood += this->get_model_likelihood();
            }
        } else {
            likelihood = this->get_model_likelihood();
        }

        return likelihood;
    }
}
