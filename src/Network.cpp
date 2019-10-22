#include "Network.h"

Network::Network()
{
    // place holder
}

Network::~Network()
{
    // place holder
}

void Network::UpdateBatch()
{
    ForwardPropagate();
    BackwardPropagate();
    UpdateWeightsAndBias();
}

void Network::ForwardPropagate()
{
    for(auto &i: __layers)
        i->ForwardPropagate();
}

void Network::BackwardPropagate() 
{
    // backward
    size_t NLayers = __layers.size();
    for(size_t i=NLayers-1; i>=0;i--)
        __layers[i]->BackwardPropagate();
}

void Network::UpdateWeightsAndBias()
{
    for(auto &i: __layers)
        i->UpdateWeightsAndBias();
}
