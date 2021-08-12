#ifndef GBNET_HPARENTNODE
#define GBNET_HPARENTNODE


#include <set>
#include "HNode.h"


namespace gbn
{
    class HParentNode
    {
        private:

        protected:

        public:
            std::set<HNode *> children;
            unsigned int n_h_child = 0;

            HParentNode();
            virtual ~HParentNode();

            void append_h_child(HNode *);
    };
}

#endif
