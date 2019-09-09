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
            ModelORNOR(const network_t, const evidence_dict_t, unsigned int = 3, bool = true);
            ModelORNOR(const network_t, const prior_active_tf_set_t, unsigned int = 3, bool = true);
            ModelORNOR(const network_t, const evidence_dict_t, const prior_active_tf_set_t, unsigned int = 3, bool = true);

            void build_graphs(bool = true);
    };
}

#endif