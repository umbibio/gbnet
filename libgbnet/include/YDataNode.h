#ifndef GBNET_YNODEDATA
#define GBNET_YNODEDATA


#include <string>
#include <vector>


#include "Multinomial.h"
#include "YNoiseNode.h"
#include "HNode.h"


namespace gbn
{

    class YDataNode: public Multinomial
    {
        private:

        protected:

        public:
            // Future: want to use these values for incluencing own likelihood
            double pvalue, abslogfc, *noise_value;

            HNode * hidden_truth;
            // YNoiseNode * noise;

            ~YDataNode () override;
            YDataNode (std::string, const double *, unsigned int);
            YDataNode (HNode *);

            double get_own_likelihood () override;
    };
}

#endif
