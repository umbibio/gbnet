#ifndef GBNET_HNODE_ORNOR
#define GBNET_HNODE_ORNOR


#include <string>
#include <vector>
#include <tuple>


#include "HNode.h"


namespace gbn
{
    // forward declarations
    class ZNode;
    class TNode;
    class XNode;
    class SNode;
    class YDataNode;
    // class ZNode;

    // ?? consider to fill this tuple with pointers to values instead of whole nodes
    typedef std::tuple<double *, double *, unsigned int *, unsigned int *> ztxs_tuple;

    class HNodeORNOR: public HNode
    {
        private:

        protected:

        public:
            // ZNode * Z;
            // double * zvalue;
            // double * tvalue;

            std::vector< ztxs_tuple > parents;

            ~HNodeORNOR () override;
            HNodeORNOR(YDataNode *);
            HNodeORNOR (std::string, const double *, gsl_rng *);
            HNodeORNOR (std::string, const double *, unsigned int);

            void append_parent(ZNode *, TNode *, XNode *, SNode *);

            double get_model_likelihood () override;
    };
}

#endif
