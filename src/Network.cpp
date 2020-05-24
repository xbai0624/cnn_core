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
    ForwardPropagateForBatch();
    BackwardPropagateForBatch();
    UpdateWeightsAndBias();
}

void Network::ForwardPropagateForBatch()
{
    int sample_size = 0;
    for(int i=0;i<sample_size;i++)
    {
	for(auto &i: __layers)
	    i->ForwardPropagateForSample();
    }
}

void Network::BackwardPropagateForBatch() 
{
    // backward
    int NLayers = __layers.size();
    for(int i=NLayers-1; i>=0;i--)
	__layers[i]->BackwardPropagateForBatch();
}

void Network::UpdateWeightsAndBias()
{
    for(auto &i: __layers)
	i->UpdateWeightsAndBias();
}
