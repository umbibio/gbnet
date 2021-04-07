#ifndef GBNET_MODELORNOR
#define GBNET_MODELORNOR


#include <cmath>
#include <time.h>


#include <gsl/gsl_rstat.h>


#include "ModelBase.h"
#include "GraphORNOR.h"


namespace gbn
{

    class ModelORNOR: public ModelBase
    {
        private:
        protected:
            double sprior[3 * 3];
        public:
            ModelORNOR();
            ~ModelORNOR() override;
            ModelORNOR(const network_t, const evidence_dict_t, const prior_active_tf_set_t = prior_active_tf_set_t(),
                       const double [3 * 3] = gbn::SPRIOR,
                       unsigned int = 3, bool = true);

            void build_graphs();
    };
}

#endif