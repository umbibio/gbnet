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
    
    class GraphORNOR: public GraphBase
    {
        private:
        protected:
        public:
            const double XPROB[2] = {0.99, 0.01};
            const double XPROB_ACTIVE[2] = {0.00, 1.00};
            const double SPROB[3][3] = {{0.99, 0.01, 0.0},
                                        {0.005, 0.99, 0.005},
                                        {0.0, 0.01, 0.99}};
            double YPROB[3] = {0.05, 0.90, 0.05};
            double YNOISE[3] = {0.945, 0.05, 0.005};
            const double YNOISE_ALPHA[3][3] = {{2000,  100,   10},
                                               { 100, 2000,  100},
                                               {  10,  100, 2000}};

            ~GraphORNOR () override;
            GraphORNOR (unsigned int);

            void build_structure (network_t, evidence_dict_t, prior_active_tf_set_t, bool = true);
    };
}

#endif