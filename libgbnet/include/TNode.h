#ifndef GBNET_TNODE
#define GBNET_TNODE


#include "Beta.h"
#include "HParentNode.h"


namespace gbn
{
    class TNode: public Beta, public HParentNode
    {
        private:

        protected:

        public:
            bool listen_children = true;

            ~TNode () override;
            TNode (std::string, double, double, gsl_rng *);
            TNode (std::string, double, double, double);
            double get_children_loglikelihood () override;
    };
}

#endif
