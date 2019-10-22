#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Layer.h"

class Network 
{
public:
    Network();
    ~Network();

    // training procedures
    void UpdateBatch();
    void ForwardPropagate();
    void BackwardPropagate();
    void UpdateWeightsAndBias();
    float GetCost();

    // testing procedures
    float GetAccuracy();
    float GetError();

private:
    std::vector<Layer*> __layers;
};

#endif
