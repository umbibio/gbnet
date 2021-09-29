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
        public:
            ModelORNOR();
            ~ModelORNOR() override;
            ModelORNOR(const network_t, const evidence_dict_t, const prior_active_tf_set_t = prior_active_tf_set_t(),
                       const double [3 * 3] = gbn::SPRIOR, double z_alpha = 25., double z_beta = 25., double z0_alpha = 25., double z0_beta = 25., double t_alpha = 25., double t_beta = 25.,
                       unsigned int = 3, bool = true, bool = true, bool = false,
                       double = 2., double = 2., double = 2., double = 2., double = 8., double = 2.);
    };
}

#endif