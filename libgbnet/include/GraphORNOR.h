#ifndef GBNET_GRAPHORNOR
#define GBNET_GRAPHORNOR


#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <set>
#include <stdio.h>


#include <gsl/gsl_rng.h>


#include "RVNode.h"
#include "XNode.h"
#include "HNodeORNOR.h"
#include "YDataNode.h"
#include "SNode.h"
#include "TNode.h"
#include "ZNode.h"
#include "NodeDictionary.h"
#include "GraphBase.h"
#include "YNoiseNode.h"


namespace gbn
{
    const double SPRIOR[3 * 3] = {0.40, 0.40, 0.20,
                                  0.30, 0.40, 0.30,
                                  0.20, 0.40, 0.40};

    class GraphORNOR: public GraphBase
    {
        private:
        protected:
        public:
            const double XPROB[2] = {0.99, 0.01};
            const double XPROB_ACTIVE[2] = {0.10, 0.90};
            double SPROB[3][3] = {{0.900, 0.090, 0.010},
                                        {0.050, 0.900, 0.050},
                                        {0.010, 0.090, 0.900}};
            double YPROB[3] = {0.05, 0.90, 0.05};
            double YNOISE[3][3] = {{0.945, 0.050, 0.005},
                                   {0.050, 0.900, 0.050},
                                   {0.005, 0.050, 0.945}};
            const double YNOISE_ALPHA[3][3] = {{2000,  100,   10},
                                               { 100, 2000,  100},
                                               {  10,  100, 2000}};

            GraphORNOR (unsigned int);
            virtual ~GraphORNOR ();

            void build_structure (network_t, evidence_dict_t, prior_active_tf_set_t, bool = true,
                                  const double [3 * 3] = SPRIOR, double z_alpha = 25., double z_beta = 25., double z0_alpha = 25., double z0_beta = 25., double t_alpha = 25., double t_beta = 25., bool comp_yprob = true, bool const_params = false,
                                  double t_focus = 2., double t_lmargin = 2., double t_hmargin = 2., double zn_focus = 2., double zn_lmargin = 8., double zn_hmargin = 2.);
    };
}

#endif