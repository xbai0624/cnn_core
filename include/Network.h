#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Layer.h"
#include "Matrix.h"

class Network 
{
public:
    Network();
    template<typename T>
	Network(T l)
	{
	    __middleLayers.push_back(dynamic_cast<Layer*>(l));
	}

    template<typename T, typename... Args>
	Network(T l, Args... pars)
	{
	    Network(pars...);
	}
    ~Network();

    // inits
    void Init();
    void ConstructLayers();

    // training procedures
    void Train();
    void UpdateBatch();
    void ForwardPropagateForBatch();
    void BackwardPropagateForBatch();
    void UpdateWeightsAndBias();
    float GetCost();

    // testing procedures
    float GetAccuracy();
    float GetError();

    // work procedures
    std::vector<Matrix> Classify();

private:
    std::vector<Layer*> __middleLayers;
};

#endif
