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
            const double noise_alpha[3] = {2000, 100, 10};
            std::vector<YDataNode *> children;
            bool listen_children = true;

            ~YNoiseNode () override;
            YNoiseNode (gsl_rng *);
            double get_children_loglikelihood () override;
    };
}

#endif
