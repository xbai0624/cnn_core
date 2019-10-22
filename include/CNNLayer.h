#ifndef CNNLAYER_H
#define CNNLAYER_H

#include "Layer.h"

class Neuron;

class CNNLayer:public Layer {
public:
    CNNLayer();
    ~CNNLayer();

    // external interfaces
    virtual void Init(); // overall init
    virtual void EpochInit();
    //after each batch process, we update weights and bias
    virtual void ProcessBatch();
    virtual void PostProcessBatch();

    // interal interfaces
    virtual void BatchInit();
    virtual void ProcessSample();
    // a helper
    virtual void InitNeurons();

    // extract a value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Matrix>& GetValueMatrixA();
    // extract z value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Matrix>& GetValueMatrixZ();
    // extract delta value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Matrix>& GetValueMatrixDelta();

    virtual void UpdateActiveWeightsAndBias(); // update active weights and bias matrix for active neurons/weight matrix

    // assign weights and bias to neurons
    virtual void AssignWeightsAndBiasToNeurons();

    // drop out
    virtual void DropOut();

    // update original weights and bias from active weights and bias
    virtual void TransferValueFromActiveToOriginal_WB();

    // helpers
    virtual void UpdateImageForCurrentTrainingSample();
    virtual void ClearImage();
    virtual NeuronCoord GetActiveNeuronDimension();

    // setters
    virtual void SetCNNStride(int);
    virtual void SetDropOutFactor(float);
    // getters
    virtual int GetCNNStride();
    virtual std::vector<Matrix>* GetWeightMatrix();
    virtual std::vector<double>* GetBiasVector();
    virtual LayerType GetType();
    virtual float GetDropOutFactor();
    virtual std::vector<std::vector<std::vector<bool>>>& GetActiveFlag();

private:
    LayerType __type = LayerType::fullyConnected;

    // this 3D matrix keeps all neurons in current layer
    std::vector<std::vector<std::vector<Neuron*>>> __neurons; // for MLPs, neurons will be in vertial vector form
    // weight matrix and vector bias; 
    // ### note: for CNN, matrix is kernel, and we have multiple kernels
    //           for MLP, # of kernel = # of neurons, and kernel dimension always is (1, M), where M is the # of active neurons in previous layer
    std::vector<Matrix> __weightMatrix;
    std::vector<double> __biasVector;
    // active weight matrix and active vector bias --for dropout algorithm
    std::vector<Matrix> __weightMatrixActive;
    std::vector<double> __biasVectorActive;

    // this 3D matrix flags active neurons for FC, active weight matrix element for CNN
    std::vector<std::vector<std::vector<bool>>> __activeFlag;

    // neuron images, for current training sample
    std::vector<Matrix> __valueMatrixA;
    std::vector<Matrix> __valueMatrixZ;
    std::vector<Matrix> __valueMatrixDelta;

    // total neuron dimension
    NeuronCoord __neuronDim;
    // active neuron dimension
    NeuronCoord __activeNeuronDim;
    // stride, x-y share the same stride for now
    int __cnnStride = 1; // for now, only use 1. Other values will be implemented with future improvements
    // dropout factor
    float __dropOut = 0.5;

    Layer* __prevLayer = nullptr;
    Layer* __nextLayer = nullptr;
};

#endif
