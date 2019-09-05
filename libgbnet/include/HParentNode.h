#ifndef GBNET_HPARENTNODE
#define GBNET_HPARENTNODE


#include <vector>
#include "HNode.h"


namespace gbn
{
    class HParentNode
    {
        private:

        protected:

        public:
            std::vector<HNode *> children;

            HParentNode();
            virtual ~HParentNode();

            void append_h_child(HNode *);
    };
}

#endif
