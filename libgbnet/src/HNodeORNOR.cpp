#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>


#include "HNodeORNOR.h"


#include "YDataNode.h"
#include "XNode.h"
#include "SNode.h"
#include "ZNode.h"


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

    void HNodeORNOR::append_parent (ZNode * z, TNode * t, XNode * x, SNode * s) {
        ztxs_tuple parent(&z->value, &t->value, &x->value, &s->value);
        this->parents.push_back(parent);
        z->append_h_child(this);
        t->append_h_child(this);
        x->append_h_child(this);
        s->append_h_child(this);
    }

    double HNodeORNOR::get_model_likelihood ()
    {
        double pr0, pr1, pr2, zcompl_pn, likelihood;

        double * zvalue;
        double * tvalue;
        unsigned int * xvalue;
        unsigned int * svalue;


        switch (this->value)
        {
        case 0:
            pr0 = 1.;
            zcompl_pn = 1.;
            for (auto ztxs_nodes: this->parents) {
                std::tie(zvalue, tvalue, xvalue, svalue) = ztxs_nodes;
                if (*svalue == 0) {
                    zcompl_pn *= (1. - *zvalue);
                    pr0 *= (1. - *tvalue * *xvalue) * *zvalue;
                } else if (*svalue == 2) {
                    zcompl_pn *= (1. - *zvalue);
                }
            }
            pr0 = (1. - pr0) * (1. - zcompl_pn) + zcompl_pn * this->prob[0];
            likelihood = pr0;
            break;
        
        case 2:
            pr0 = 1.;
            pr2 = 1.;
            zcompl_pn = 1.;
            for (auto ztxs_nodes: this->parents) {
                std::tie(zvalue, tvalue, xvalue, svalue) = ztxs_nodes;
                if (*svalue == 2) {
                    zcompl_pn *= (1. - *zvalue);
                    pr2 *= (1. - *tvalue * *xvalue) * *zvalue;
                } else if (*svalue == 0) {
                    zcompl_pn *= (1. - *zvalue);
                    pr0 *= (1. - *tvalue * *xvalue) * *zvalue;
                }
            }
            pr2 = (pr0 - pr2*pr0) * (1. - zcompl_pn) + zcompl_pn * this->prob[2];
            likelihood = pr2;
            break;
        
        case 1:
            pr1 = 1.;
            zcompl_pn = 1.;
            for (auto ztxs_nodes: this->parents) {
                std::tie(zvalue, tvalue, xvalue, svalue) = ztxs_nodes;
                if (*svalue != 1){
                    zcompl_pn *= (1. - *zvalue);
                    pr1 *= (1. - *tvalue * *xvalue) * *zvalue;
                }
            }
            pr1 = pr1 * (1. - zcompl_pn) + zcompl_pn * this->prob[1];
            likelihood = pr1;
            break;
        
        default:
            throw std::out_of_range("Current node value is invalid");
            break;
        }

        return likelihood;
    }

}
