#ifndef FCLAYER_H
#define FCLAYER_H

#include "Layer.h"

class Neuron;

/***********************************************************************************
 *
 * Purely fully connected layer: 
 *     means: its previous layer is a vertical layer (single collumn of neurons)
 *            its next layer is also a vertical layer ( single collumn of neurons)
 *
 ***********************************************************************************/

class FCLayer: public Layer 
{
public:
    FCLayer();
    ~FCLayer();

    // external interfaces
    virtual void Init(); // overall init
    virtual void EpochInit();
    virtual void ForwardPropagate();
    virtual void BackwardPropagate();

    // update weights and bias, for external call
    virtual void UpdateWeightsAndBias();
    void UpdateWeightsAndBiasFC();
    void UpdateWeightsAndBiasCNN();
    void UpdateWeightsAndBiasPooling();
    // helpers, update weights and bias gradients vector for each training sample
    void UpdateWeightsAndBiasGradients();
    void UpdateWeightsAndBiasGradientsFC();
    void UpdateWeightsAndBiasGradientsCNN();
    void UpdateWeightsAndBiasGradientsPooling();

    // hyper parameters
    virtual void SetLearningRate(double);
    virtual void SetRegularizationMethod(Regularization);
    virtual void SetRegularizationParameter(double);

    //after each batch process, we update weights and bias
    virtual void ProcessBatch();
    virtual void PostProcessBatch();

    // interal interfaces
    virtual void BatchInit();
    virtual void ProcessSample();
    // a helper
    virtual void InitNeurons();

    // extract a value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesA();
    // extract z value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesZ();
    // extract delta value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesDelta();

    // get active neuron flags
    //virtual std::vector<std::vector<std::vector<NeuronCoord>>>& GetActiveNeuronFlags();
    virtual void UpdateCoordsForActiveNeuronFC(); // re-assign coordinates to active neurons

    virtual void UpdateActiveWeightsAndBias(); // update active weights and bias matrix for active neurons/weight matrix

    // assign weights and bias to neurons
    virtual void AssignWeightsAndBiasToNeurons();

    // drop out
    virtual void DropOut();

    // update active flag FC; heler function for FC; CNN layer doesn't need it,
    // the updateActiveFlagCNN() function are incoporated in DropOutCNN() function
    void __UpdateActiveFlagFC();
 
    // update original weights and bias from active weights and bias
    virtual void TransferValueFromActiveToOriginal_WB();

    // helpers
    virtual void UpdateImageForCurrentTrainingSample();
    virtual void ClearImage();
    virtual NeuronCoord GetActiveNeuronDimension();

    // setters
    virtual void SetDropOutFactor(float);
    // getters
    virtual std::vector<Matrix>* GetWeightMatrix();
    virtual std::vector<double>* GetBiasVector();
    virtual LayerType GetType();
    virtual float GetDropOutFactor();
    virtual std::vector<std::vector<std::vector<bool>>>& GetActiveFlag();

private:
    // 1):
    LayerType __type = LayerType::fullyConnected;

    // 2):
    // this 3D matrix keeps all neurons in current layer
    std::vector<std::vector<std::vector<Neuron*>>> __neurons; // for MLPs, neurons will be in vertical vector form 
    // we will follow exactly the Math form, so for FC layer, it has only one vertical form neurons
    // __neruons.size() = 1; __neurons[0].size() = N (layer size); __neruons[0][0].size() = 1;

    // weight matrix and vector bias; 
    // ### note: for CNN, matrix is kernel, and we have multiple kernels: __weightMatrix.sizes() = # of kernels
    //           for MLP, we have only 1 weight matrix, __weightMatrix.size() = 1
    //                    and matrix dimension is always (M, N), 
    //                    where N is the # of active neurons in previous layer
    //                          M is the current layer size
    std::vector<Matrix> __weightMatrix;
    // bias vector: for cnn, __biasVector.size() = # of kernels; for MLP, __biaseVector.size() = # of neurons
    std::vector<Matrix> __biasVector;

    // the following two vectors need to be updated in back propagation step, along with delta
    // saves weight gradient in each batch
    // Images.SampleOutputImage is a vector saves gradients for each kernel
    std::vector<Images> __wGradient; 
    // saves bias gradient in each batch
    std::vector<Images> __bGradient; 

    double __learningRate = 0.0; // learning rate

    // regularization always active, if you don't want it, set __regularization parameter to 0
    Regularization __regularizationMethod = Regularization::L2;
    double __regularizationParameter = 0.0; // default to 0. 


    // 3):
    // active weight matrix and active vector bias --for dropout algorithm
    std::vector<Matrix> __weightMatrixActive;
    std::vector<Matrix> __biasVectorActive;

    // 4):
    // this 3D matrix flags active neurons for FC, active weight matrix element for CNN
    std::vector<Filter2D> __activeFlag;

    // 5):
    // neuron images, Images is for sample
    // vector<Images> is for batch
    // size should = batch size
    std::vector<Images> __imageA;
    std::vector<Images> __imageZ;
    std::vector<Images> __imageDelta;

    // 6):
    // total neuron dimension
    NeuronCoord __neuronDim;
    // active neuron dimension
    NeuronCoord __activeNeuronDim;

    // 7):
    // dropout factor
    float __dropOut = 0.5;

    // 8):
    Layer* __prevLayer = nullptr;
    Layer* __nextLayer = nullptr;
};

#endif
