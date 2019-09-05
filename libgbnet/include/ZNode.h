#ifndef GBNET_ZNODE
#define GBNET_ZNODE


#include "Beta.h"
#include "HParentNode.h"


namespace gbn
{
    class ZNode: public Beta, public HParentNode
    {
        private:

        protected:

        public:
            bool listen_children = true;

            ~ZNode () override;
            ZNode (std::string, double, double, gsl_rng *);
            ZNode (std::string, double, double, double);
            double get_children_loglikelihood () override;
    };
}

#endif
