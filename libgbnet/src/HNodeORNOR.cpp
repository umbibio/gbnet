#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "HNodeORNOR.h"


#include "YDataNode.h"
#include "XNode.h"
#include "SNode.h"


namespace gbn
{
    HNodeORNOR::~HNodeORNOR ()
    {
        // std::cout << "Object at " << this << " destroyed, instance of  HNodeORNOR  \t" << this->id << "\t" << typeid(this).name() << std::endl;
    }

    HNodeORNOR::HNodeORNOR (YDataNode * Y)
    : HNode(Y) {}

    HNodeORNOR::HNodeORNOR ( std::string uid, const double * prob, gsl_rng * rng )
    : HNode (uid, prob, rng) {}

    HNodeORNOR::HNodeORNOR ( std::string uid, const double * prob, unsigned int value )
    : HNode (uid, prob, value) {}

    void HNodeORNOR::append_parent (XNode * x, SNode * s) {
        xs_tuple parent(&x->value, &s->value);
        this->parents.push_back(parent);
        x->append_h_child(this);
        s->append_h_child(this);
    }

    double HNodeORNOR::get_model_likelihood ()
    {
        double pr0, pr1, pr2, zcompl_pn, likelihood;

        unsigned int * xvalue;
        unsigned int * svalue;

        double tvalue = *this->tvalue;
        double zvalue = *this->zvalue;
        double zcompl = 1.- zvalue;


        switch (this->value)
        {
        case 0:
            pr0 = 1.;
            zcompl_pn = 1.;
            for (auto xs_nodes: this->parents) {
                std::tie(xvalue, svalue) = xs_nodes;
                if (*svalue == 0) {
                    zcompl_pn *= zcompl;
                    pr0 *= (1. - tvalue * *xvalue) * zvalue;
                } else if (*svalue == 2) {
                    zcompl_pn *= zcompl;
                }
            }
            pr0 = (1. - pr0) * (1. - zcompl_pn) + zcompl_pn * this->prob[0];
            likelihood = pr0 * this->data->get_own_likelihood();
            break;
        
        case 2:
            pr0 = 1.;
            pr2 = 1.;
            zcompl_pn = 1.;
            for (auto xs_nodes: this->parents) {
                std::tie(xvalue, svalue) = xs_nodes;
                if (*svalue == 2) {
                    zcompl_pn *= zcompl;
                    pr2 *= (1. - tvalue * *xvalue) * zvalue;
                } else if (*svalue == 0) {
                    zcompl_pn *= zcompl;
                    pr0 *= (1. - tvalue * *xvalue) * zvalue;
                }
            }
            pr2 = (pr0 - pr2*pr0) * (1. - zcompl_pn) + zcompl_pn * this->prob[2];
            likelihood = pr2 * this->data->get_own_likelihood();
            break;
        
        case 1:
            pr1 = 1.;
            zcompl_pn = 1.;
            for (auto xs_nodes: this->parents) {
                std::tie(xvalue, svalue) = xs_nodes;
                if (*svalue != 1){
                    zcompl_pn *= zcompl;
                    pr1 *= (1. - tvalue * *xvalue) * zvalue;
                }
            }
            pr1 = pr1 * (1. - zcompl_pn) + zcompl_pn * this->prob[1];
            likelihood = pr1 * this->data->get_own_likelihood();
            break;
        
        default:
            throw std::out_of_range("Current node value is invalid");
            break;
        }

        return likelihood;
    }

}
