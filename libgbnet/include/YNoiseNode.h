#ifndef GBNET_YNOISENODE
#define GBNET_YNOISENODE


#include "Dirichlet.h"


namespace gbn
{
    // forward declaration
    class YDataNode;

    class YNoiseNode: public Dirichlet
    {
        private:

        protected:

        public:
            std::vector<YDataNode *> children;
            bool listen_children = true;

            ~YNoiseNode () override;
            YNoiseNode (unsigned int, const double *, gsl_rng *);
            double get_children_loglikelihood () override;
    };
}

#endif
