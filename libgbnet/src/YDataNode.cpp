#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "YDataNode.h"


namespace gbn
{
    YDataNode::~YDataNode ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  YDataNode  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    YDataNode::YDataNode ( std::string uid, const double * prob, unsigned int value)
    : Multinomial ((std::string) "Y", uid, 3, prob, value) {}

    YDataNode::YDataNode(HNode * H) 
    : Multinomial ((std::string) "Y", H->uid, H->n_outcomes, H->prob, H->value)
    {
        H->data = this;
        this->hidden_truth = H;
    }

    double YDataNode::get_own_likelihood ()
    {
        unsigned int delta = abs((int)this->hidden_truth->value - (int)this->value);

        double lik;
        switch (this->value)
        {
        case 1:
            lik = delta == 0 ? 1. - 2. * this->noise_value[1] : this->noise_value[1];
            lik = std::max(0., lik);
            break;
        
        default:
            lik = this->noise_value[delta];
            break;
        }

        return lik;
    }
}
