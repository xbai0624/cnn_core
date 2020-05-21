#include "Network.h"

Network::Network()
{
    // place holder
}

Network::~Network()
{
    // place holder
}

void Network::Init()
{
    // init layers
    for(auto &i: __layers)
        i->Init();
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
    int NLayers = __layers.size();
    for(int i=NLayers-1; i>=0;i--)
        __layers[i]->BackwardPropagate();
}

void Network::UpdateWeightsAndBias()
{
    for(auto &i: __layers)
        i->UpdateWeightsAndBias();
}
