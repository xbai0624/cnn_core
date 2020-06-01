#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Layer.h"
#include "Matrix.h"

class DataInterface;

struct LayerParameter
{
    // summary for layer parameters
    //     use this struct to pass parameters to each layers
    //     to avoid forget setting some parameters for layers

    LayerType _pLayerType;
    LayerDimension _pLayerdimension;

    DataInterface * _pDataInterface;

    size_t _nNeuronsFC;
    size_t _nKernels;

    float _glearningRate;

    bool _gUseDropout;
    bool _gUseL2Regularization;
    bool _gUseL1Regularization;

};

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
    void UpdateEpoch();
    void UpdateBatch();
    void ForwardPropagateForBatch();
    void BackwardPropagateForBatch();
    void UpdateWeightsAndBiasForBatch();
    float GetCost();

    // testing procedures
    float GetAccuracy();
    float GetError();

    // work procedures
    std::vector<Matrix> Classify();

private:
    std::vector<Layer*> __middleLayers;                  // save all middle layers
    Layer *__inputLayer=nullptr, *__outputLayer=nullptr; // input and output layers
    DataInterface *__dataInterface = nullptr;            // save data interface class

    int __numberOfEpoch = 100; // nuber of epochs
};

#endif
