/*
 * a abstract base class for layer design, includes necessary data structures
 * layer interface class
 */

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cstddef> // size_t
#include <utility>

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
    pooling
};


enum class Regularization 
{
    L2,
    L1
};


// a 2d matrix for mask out neurons
struct Filter2D
{
    std::vector<std::vector<bool>> __filter; 

    Filter2D()
    {}
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
	i(-1), j(-1), k(-1) 
    {}

    NeuronCoord(size_t _i, size_t _j, size_t _k)
	: i(_i), j(_j), k(_k) 
    {}
};


// a struct holding layer output image of one single training sample
// so each batch will have a vector of this struct
struct Images
{
    std::vector<Matrix> SampleOutputImage;

    Images()
    {}

    size_t GetNumberOfKernels()
    {
        // for cnn layer, this returns number of kernels
        // for mlp layer, this should return 1
        return SampleOutputImage.size();
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
class Layer 
{
public:
    Layer();
    virtual ~Layer();

    // external interfaces
    virtual void Init()=0; // overall init
    virtual void EpochInit()=0; // init before each epoch
    virtual void ForwardPropagate() = 0;
    virtual void BackwardPropagate() = 0;
    virtual void UpdateWeightsAndBias() = 0;

    //after each batch process, we update weights and bias
    virtual void ProcessBatch()=0;
    virtual void PostProcessBatch()=0;

    // interal interfaces
    virtual void BatchInit()=0;
    virtual void ProcessSample()=0;
    // a helper
    virtual void InitNeurons()=0;

    // extract a value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesA()=0;
    // extract z value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesZ()=0;
    // extract delta value from neurons and re-organize these values in matrix form, only for current training sample
    virtual std::vector<Images>& GetImagesDelta()=0;

    // get active neuron flags
    //virtual std::vector<std::vector<std::vector<NeuronCoord>>>& GetActiveNeuronFlags()=0;
    virtual void UpdateCoordsForActiveNeuronFC()=0; // re-assign coordinates to active neurons

    virtual void UpdateActiveWeightsAndBias()=0; // update active weights and bias matrix for active neurons/weight matrix

    // assign weights and bias to neurons
    virtual void AssignWeightsAndBiasToNeurons()=0;

    // drop out
    virtual void DropOut()=0;

    // update original weights and bias from active weights and bias
    virtual void TransferValueFromActiveToOriginal_WB()=0;

    // helpers
    virtual void UpdateImageForCurrentTrainingSample()=0;
    virtual void ClearImage()=0;
    virtual NeuronCoord GetActiveNeuronDimension()=0;

    // setters
    virtual void SetPoolingMethod(PoolingMethod)=0;
    virtual void SetCNNStride(int)=0;
    virtual void SetDropOutFactor(float)=0;
    // getters
    virtual PoolingMethod & GetPoolingMethod()=0;
    virtual int GetCNNStride()=0;
    virtual std::vector<Matrix>* GetWeightMatrix()=0;
    virtual std::vector<double>* GetBiasVector()=0;
    virtual LayerType GetType()=0;
    virtual float GetDropOutFactor()=0;
    virtual std::vector<std::vector<std::vector<bool>>>& GetActiveFlag()=0;

private:
    // reserved section
};

#endif
