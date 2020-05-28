#ifndef FCLAYER_H
#define FCLAYER_H

#include "Layer.h"

class Neuron;
class DataInterface;

/***********************************************************************************
 *
 * Construct specific layer
 *
 *     NOTE: 
 *          Multithreading will be applied inside batch, namely on sample level
 *
 ***********************************************************************************/

class ConstructLayer: public Layer 
{
public:
    ConstructLayer();
    ConstructLayer(LayerType t); // for input layer
    ConstructLayer(LayerType t, int n_neurons); // for fc layer and output layer
    ConstructLayer(LayerType t, int n_kernels, std::pair<size_t, size_t> d); // for cnn layer
    ~ConstructLayer();

    // external interfaces
    virtual void Init(); // overall init
    virtual void InitNeurons();
    virtual void InitWeightsAndBias();
    virtual void Connect(Layer* prev=nullptr, Layer* next=nullptr);

    // interface
    virtual void EpochInit();
    virtual void BatchInit();

    virtual void SetNumberOfNeuronsFC(size_t n);
    virtual void SetNumberOfKernelsCNN(size_t n);
    virtual void SetKernelSizeCNN(std::pair<size_t, size_t> s);
    void InitNeuronsCNN();
    void InitNeuronsFC();
    void InitNeuronsInputLayer();
    void InitFilters();

    // hyper parameters
    virtual void SetLearningRate(double);
    virtual void SetRegularizationMethod(Regularization);
    virtual void SetRegularizationParameter(double);
    virtual void EnableDropOut();
    virtual void DisableDropOut();
    // setters
    virtual void SetPoolingMethod(PoolingMethod);
    virtual void SetCNNStride(int);
    virtual void SetDropOutFactor(float);
    virtual void SetPrevLayer(Layer *);
    virtual void SetNextLayer(Layer *);
    virtual void SetCostFuncType(CostFuncType t);

    //
    virtual void ForwardPropagateForSample(int sample_index);
    virtual void BackwardPropagateForBatch();
    virtual void ComputeCostInOutputLayerForCurrentSample(int sample_index);

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

    //after each batch process, we update weights and bias
    virtual void ProcessBatch();
    virtual void PostProcessBatch();

    // interal interfaces
    virtual void ProcessSample();

    // extract a value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesA();
    void UpdateImagesA(int sample_index);
    // extract z value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesZ();
    void UpdateImagesZ(int sample_index);
    // extract delta value from neurons and re-organize these values in matrix form, only for current batch
    // !!::NOTE::!!  during backpropagation, delta need to be calculated over all batch samples,
    // !!::NOTE::!!  b/c Cost function is a sum of all batch samples
    // !!::NOTE::!!  this is not the same with A & Z images, they are calculated for each sample
    virtual std::vector<Images>& GetImagesDelta();
    void UpdateImagesDelta();
    // Fill A matrix for input layer
    void FillDataToInputLayerA();

    // get active neuron flags
    //virtual std::vector<std::vector<std::vector<NeuronCoord>>>& GetActiveNeuronFlags();
    virtual void UpdateCoordsForActiveNeuronFC(); // re-assign coordinates to active neurons; for drop-out 
    virtual void UpdateActiveWeightsAndBias(); // update active weights and bias matrix for active neurons/weight matrix

    // assign weights and bias to neurons
    virtual void AssignWeightsAndBiasToNeurons();

    // drop out
    virtual void DropOut();

    // update active flag FC; heler function for FC;
    void __UpdateActiveFlagFC();
    void __UpdateActiveFlagCNN();
 
    // update original weights and bias from active weights and bias
    virtual void TransferValueFromActiveToOriginal_WB();
    virtual void TransferValueFromOriginalToActive_WB();

    // helpers
    virtual void UpdateImageForCurrentTrainingSample();
    virtual void ClearImage();
    virtual NeuronCoord GetActiveNeuronDimension();
    virtual void Print();
    virtual void PassDataInterface(DataInterface *data_interface);
    virtual void ClearUsedSampleForInputLayer_obsolete();

    // getters
    virtual PoolingMethod & GetPoolingMethod();
    virtual int GetCNNStride();
    virtual std::vector<Matrix>* GetWeightMatrix();
    virtual std::vector<Matrix>* GetBiasVector();
    virtual LayerType GetType();
    virtual float GetDropOutFactor();
    virtual std::vector<Filter2D>& GetActiveFlag();
    virtual std::pair<size_t, size_t> GetOutputImageSize(); // used for setup layer
    virtual std::pair<size_t, size_t> GetOutputImageSizeCNN(); // used for setup layer
    virtual std::pair<size_t, size_t> GetOutputImageSizeFC(); // used for setup layer
    virtual int GetNumberOfNeurons();
    virtual int GetNumberOfNeuronsFC();
    virtual int GetBatchSize();
    virtual CostFuncType GetCostFuncType();
    virtual DataInterface * GetDataInterface();

private:
    // 1):
    LayerType __type = LayerType::fullyConnected;

    // for setup fc layer
    size_t __n_neurons_fc = 0;
    // for setup cnn layer
    size_t __n_kernels_cnn = 0;
    std::pair<size_t, size_t> __kernelDim;
    // for setup cnn layer
    // during setup, output image size should be already decided
    std::pair<size_t, size_t> __outputImageSizeCNN;

    // 2):
    // this 3D matrix keeps all neurons in current layer
    std::vector<Pixel2D<Neuron*>> __neurons; // for MLPs, neurons will be in vertical vector form 
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
    // for cnn layer, each matrix in this vector corresponds to a kernel
    // for fc layer, each matrix in this vector corresponds to a row of the original w&b, for easier to assign them to neurons
    std::vector<Matrix> __weightMatrixActive;
    std::vector<Matrix> __biasVectorActive;

    // 4):
    // this 3D matrix filters out active neurons for FC, active weight matrix element for CNN
    std::vector<Filter2D> __activeFlag;

    // 5):
    // neuron images, Images is for sample
    // vector<Images> is for batch
    // size should = batch size
    std::vector<Images> __imageA;
    std::vector<Images> __imageZ;
    std::vector<Images> __imageDelta;
    //std::vector<Images> __imageSigmaPrime;
    std::vector<double> __outputLayerCost;

    // 6):
    // total neuron dimension
    NeuronCoord __neuronDim;
    // active neuron dimension
    NeuronCoord __activeNeuronDim;

    // 7):
    bool __use_drop_out = false;
    // dropout factor
    float __dropOut = 0.5;
    // stride, for now x-y share the same stride
    // for now, only use 1. Other values will be implemented with future improvements
    int __cnnStride = 1; 
    PoolingMethod __poolingMethod = PoolingMethod::Max;
    CostFuncType __cost_func_type = CostFuncType::cross_entropy;
 
    // 8):
    Layer* __prevLayer = nullptr;
    Layer* __nextLayer = nullptr;

    // 9):
    // an interface for processing input image
    DataInterface *__p_data_interface = nullptr;
};

#endif
