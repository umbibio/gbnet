#ifndef GBNET_XNODE
#define GBNET_XNODE


#include "Multinomial.h"
#include "HParentNode.h"


namespace gbn
{
    class XNode: public Multinomial, public HParentNode
    {
        private:

        protected:

        public:
            ~XNode () override;
            XNode (std::string, const double *, gsl_rng *);
            XNode (std::string, const double *, unsigned int);
            double get_children_loglikelihood () override;
    };
}

#endif
