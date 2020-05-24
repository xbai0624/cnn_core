#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Layer.h"

class Network 
{
public:
    Network();

    template<typename T>
        Network(T l)
        {
            __layers.push_back(dynamic_cast<Layer*>(l));
        }

    template<typename T, typename... Args>
        Network(T l, Args... pars)
        {
            Network(pars...);
        }

    ~Network();

    // inits
    void Init();

    // training procedures
    void UpdateBatch();
    void ForwardPropagateForBatch();
    void BackwardPropagateForBatch();
    void UpdateWeightsAndBias();
    float GetCost();

    // testing procedures
    float GetAccuracy();
    float GetError();

private:
    std::vector<Layer*> __layers;
};

#endif
