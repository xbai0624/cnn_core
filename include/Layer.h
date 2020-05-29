/*
 * a abstract base class for layer design, includes necessary data structures
 * layer interface class
 */

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cstddef> // size_t
#include <utility>
//#include <ostream>

#include "Matrix.h"

// necessary data structures for layer/neuron class
enum class PoolingMethod 
{
    Max,
    Average
};


enum class LayerType 
{
    fullyConnected,
    cnn,
    pooling,
    input, // input and output layer also require special treatment
    output
};


enum class Regularization 
{
    L2,
    L1
};

enum class CostFuncType
{
    cross_entropy,  // sigmoid
    log_likelihood, // for softmax
    quadratic_sum   // not practical, for test only
};


// a 2d matrix for mask out neurons
struct Filter2D
{
    std::vector<std::vector<bool>> __filter; 

    Filter2D()
    {}

    Filter2D(size_t i, size_t j)
    {
        __filter.resize(i, std::vector<bool>(j, true));
    }

    std::pair<size_t, size_t> Dimension()
    {
        if(__filter.size() == 0) return std::pair<size_t, size_t>(0, 0);
        return std::pair<size_t, size_t>(__filter.size(), __filter[0].size());
    }

    std::vector<bool>& operator[](size_t i)
    {
        return __filter[i];
    }
};
std::ostream & operator<<(std::ostream &, const Filter2D &t);

// a 2d structure for organize neurons
template <class T>
struct Pixel2D
{
    std::vector<std::vector<T>> __plane;
    Pixel2D()
    {}

    Pixel2D(size_t i, size_t j)
    {
        __plane.resize(i, std::vector<T>(j, 0));
    }

    std::pair<size_t, size_t> Dimension()
    {
        if(__plane.size() == 0) return std::pair<size_t, size_t>(0, 0);
        return std::pair<size_t, size_t>(__plane.size(), __plane[0].size());
    }

    std::vector<T> & operator[](size_t i)
    {
        return __plane[i];
    }


};


struct NeuronCoord 
{
    // Record neuron (pixel) position in current layer
    // if fully connected layer, neurons will be in vector form, only i will be used
    // if cnn/pooling layer, (i, j) is position for one kernel output, k is for kernel index
    //     (i, j): i for row, j for collum, like bellow
    //     *  ******* j *******
    //     *  *****************
    //     i  *****************
    //     *  *****************
    //     *  *****************
    // this way it agrees with matrix: vector<vector<float>>
    size_t i, j, k;
    NeuronCoord(): 
	i(0), j(0), k(0) 
    {}

    NeuronCoord(size_t _i, size_t _j, size_t _k)
	: i(_i), j(_j), k(_k) 
    {}
};
std::ostream & operator<<(std::ostream&, const NeuronCoord &c);


// a struct holding layer output image of one single training sample
// so each batch will have a vector of this (Images) struct
struct Images
{
    std::vector<Matrix> OutputImageFromKernel; // this vector is for different cnn kernels 
    // this vector saves: <Kernel_0, Kernel_1, Kernel_2, ...... , Kernel_n>

    Images()
    {}

    size_t GetNumberOfKernels()
    {
        // for cnn layer, this returns number of kernels
        // for mlp layer, this should return 1
        return OutputImageFromKernel.size();
    }
};


// ref: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9

// layer class
//**************************************************************************************
//                                      design 
// each layer has two sets of w&b matrix, neuron matrix; one set is the "original copy"
// the other set is the "active copy" (for drop out, etc).
// Before processing every batch, the BatchInit() function will update the "active copy"
// according to "active-neuron-flag matrix".
//
// The weights and bias will be updated by batch unit ("active copy"), 
// and the "orignal copy" will be also updated according to the "flag matrix".
//
//**************************************************************************************

class DataInterface;

class Layer 
{
public:
    Layer();
    virtual ~Layer();

    // external interfaces
    virtual void Init()=0; // overall init
    virtual void EpochInit()=0; // init before each epoch
    virtual void Connect(Layer *prev=nullptr, Layer *next=nullptr) = 0;

    // propagation
    virtual void ForwardPropagateForSample(int) = 0;
    virtual void BackwardPropagateForSample(int) = 0;
    virtual void UpdateWeightsAndBias() = 0;
    virtual void ComputeCostInOutputLayerForCurrentSample(int) = 0;

    // setup hyper parameters
    virtual void SetLearningRate(double) = 0;
    virtual void SetRegularizationMethod(Regularization) = 0;
    virtual void SetRegularizationParameter(double) = 0;

    //after each batch process, we update weights and bias
    virtual void ProcessBatch()=0;
    virtual void PostProcessBatch()=0;

    // internal interfaces
    virtual void BatchInit()=0;
    virtual void InitFilters()=0;
    virtual void ProcessSample()=0;
    // a helper
    virtual void SetNumberOfNeuronsFC(size_t n) = 0;
    virtual void SetNumberOfKernelsCNN(size_t n) = 0;
    virtual void SetKernelSizeCNN(std::pair<size_t, size_t> s) = 0;
    virtual void InitNeurons()=0;
    virtual void InitWeightsAndBias()=0;

    // extract a value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesActiveA()=0;
    virtual std::vector<Images>& GetImagesFullA()=0;
    // extract z value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesActiveZ()=0;
    virtual std::vector<Images>& GetImagesFullZ()=0;
    // extract delta value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesActiveDelta()=0;
    virtual std::vector<Images>& GetImagesFullDelta()=0;

    // get active neuron flags
    //virtual std::vector<std::vector<std::vector<NeuronCoord>>>& GetActiveNeuronFlags()=0;
    virtual void UpdateCoordsForActiveNeuronFC()=0; // re-assign coordinates to active neurons

    virtual void UpdateActiveWeightsAndBias()=0; // update active weights and bias matrix for active neurons/weight matrix

    // assign weights and bias to neurons
    virtual void AssignWeightsAndBiasToNeurons()=0;

    // drop out
    virtual void DropOut()=0;
    virtual void EnableDropOut() = 0;
    virtual void DisableDropOut() = 0;

    // update original weights and bias from active weights and bias
    virtual void TransferValueFromActiveToOriginal_WB()=0;
    virtual void TransferValueFromOriginalToActive_WB()=0;

    // helpers
    virtual void UpdateImageForCurrentTrainingSample()=0;
    virtual void ClearImage()=0;
    virtual NeuronCoord GetActiveNeuronDimension()=0;
    virtual void Print() = 0;
    virtual void PassDataInterface(DataInterface *data_interface) = 0;
    virtual void FillDataToInputLayerA() = 0;
    virtual void ClearUsedSampleForInputLayer_obsolete() = 0;

    // setters
    virtual void SetPoolingMethod(PoolingMethod)=0;
    virtual void SetCNNStride(int)=0;
    virtual void SetDropOutFactor(float)=0;
    virtual void SetPrevLayer(Layer *) = 0; // pass pointer by reference
    virtual void SetNextLayer(Layer *) = 0; // pass pointer by reference
    virtual void SetCostFuncType(CostFuncType t) = 0;

    // getters
    virtual PoolingMethod & GetPoolingMethod()=0;
    virtual int GetCNNStride()=0;
    virtual std::vector<Matrix>* GetWeightMatrix()=0;
    virtual std::vector<Matrix>* GetBiasVector()=0;
    virtual LayerType GetType()=0;
    virtual float GetDropOutFactor()=0;
    virtual std::vector<Filter2D>& GetActiveFlag()=0;
    virtual std::pair<size_t, size_t> GetOutputImageSize() = 0; // used for setup layer
    virtual int GetNumberOfNeurons() = 0;
    virtual int GetNumberOfNeuronsFC() = 0;
    virtual int GetID(){return __layerID;};
    virtual int GetBatchSize() = 0;
    virtual CostFuncType GetCostFuncType() = 0;
    virtual DataInterface * GetDataInterface() = 0;
    virtual Layer* GetNextLayer() = 0;
    virtual Layer* GetPrevLayer() = 0;

private:
    // reserved section
    static int __layerCount;
    int __layerID = 0;
};

#endif
