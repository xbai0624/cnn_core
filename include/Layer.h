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
#include <cassert>
//#include <iostream>
#include "Matrix.h"

// actuation function type used in neuron class: a =\sigma(z)
//     for neuron
enum class ActuationFuncType 
{
    Sigmoid,
    SoftMax,
    Tanh,
    Relu,
    Undefined
};


// necessary data structures for layer/neuron class
enum class PoolingMethod 
{
    Max,
    Average,
    Undefined
};


enum class LayerType 
{
    fullyConnected,
    cnn,
    pooling,
    input, // 1) input and output layer also require special treatment; input layer has no neurons, it is a transfer layer forwarding data to its followers
    output,
    Undefined
};
std::ostream & operator<<(std::ostream&, const LayerType &);

enum class LayerDimension
{
    _1D,
    _2D,
    Undefined
};


enum class Regularization 
{
    L2,
    L1,
    Undefined
};

enum class CostFuncType
{
    cross_entropy,  // sigmoid
    log_likelihood, // for softmax
    quadratic_sum,  // not practical, for test only
    Undefined
};


// a 2d matrix for mask out neurons
struct Filter2D
{
    std::vector<std::vector<bool>> __filter; 

    Filter2D()
    {
        __filter.clear();
    }

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

//#include <iostream>

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

    Images ReshapeKernels(size_t I, size_t J) const
    {
        // reshape matrix from different kernels
	assert(OutputImageFromKernel.size() > 0);
	Images res;
        for(auto &i: OutputImageFromKernel)	
	{
	    auto dim = i.Dimension();
	    assert(dim.first * dim.second == I * J);
	    Matrix tmp = i.Reshape(I, J);
	    res.OutputImageFromKernel.push_back(tmp);
	}
	return res;
    }

    Images Vectorization() const
    {
        // combine outputs from all kernels into a one-collum matrix
	// this is for feedforward from 2D layer to 1D layer 
	//     (2D layer can have multiple kernels, 1D layer should only have 1 "kernel");
	assert(OutputImageFromKernel.size() > 0);
        auto dim = OutputImageFromKernel[0].Dimension();
	
	Images tmp = ReshapeKernels(dim.first*dim.second, 1);
	Matrix large_vector = Matrix::ConcatenateMatrixByI(tmp.OutputImageFromKernel);

	tmp.OutputImageFromKernel.clear();
	tmp.OutputImageFromKernel.push_back(large_vector);
	return tmp;
    }

    Images Tensorization(size_t I, size_t J) const
    {
        // this is the inverse operation for vectorization
	// used for backward propagation, 1D layer to 2D layer
	//     [I J] is the dimension of tensorized kernel matrix
        assert(OutputImageFromKernel.size() == 1); // make sure this is from 1D layer (only one kernel)
	Matrix tmp = OutputImageFromKernel[0];
	auto dim = tmp.Dimension();
	assert(dim.second == 1); // make sure it is a one-collumn matrix
	size_t unit_quantity = I * J; //  number of elements in each tensorized kernel
	assert(dim.first%unit_quantity == 0);

	int nKernels = dim.first/unit_quantity;
	Images Ret;
	for(size_t i=0;i<(size_t)nKernels;i++)
	{
	    Matrix _tmp = tmp.GetSection(i*unit_quantity, (i+1)*unit_quantity, 0, 1);
	    Matrix _t = _tmp.Reshape(I, J);
	    Ret.OutputImageFromKernel.push_back(_t);
	}
	//std::cout<<"before return: "<<&Ret<<std::endl;
	return Ret;
    }
};
std::ostream & operator<<(std::ostream &, const Images & );


// A list summarizing all parameters used for constructing layers
class DataInterface; // declaratioin for DataInterface class
struct LayerParameterList
{
    // summary for layer parameters
    //     use this struct to pass parameters to each layers
    //     to avoid forget setting some parameters for layers

    LayerType _gLayerType;              // for all layers
    LayerDimension _gLayerDimension;    // for all layers

    DataInterface * _pDataInterface;    // for all layers

    size_t _nNeuronsFC;                 // for fully connected layers
    size_t _nKernels;                   // for cnn and pooling layers
    std::pair<size_t, size_t> _gDimKernel; // kernel dimension for cnn and pooling layers

    float _gLearningRate;               // for non-input layers

    bool _gUseDropout;                  // for middle layers (non-input, non-output)
    float _gDropoutFactor;              // for middle layers

    Regularization _gRegularization;              // for non-input layers
    float _gRegularizationParameter;    // for non-input layers

    ActuationFuncType _gActuationFuncType;

    // default constructor
    LayerParameterList():
	_gLayerType(LayerType::Undefined), _gLayerDimension(LayerDimension::Undefined), 
	_pDataInterface(nullptr), _nNeuronsFC(0), _nKernels(0), _gDimKernel(std::pair<size_t, size_t>(0, 0)),
	_gLearningRate(0), _gUseDropout(false), _gDropoutFactor(0), _gRegularization(Regularization::Undefined),
	_gRegularizationParameter(0), _gActuationFuncType(ActuationFuncType::Sigmoid)
    {
    }

    LayerParameterList(LayerType layer_type, LayerDimension layer_dimension, DataInterface *data_interface, 
	    size_t n_neurons, size_t n_kernels, std::pair<size_t, size_t> dimension_kernel, float learning_rate,
	    bool use_dropout, float dropout_factor, Regularization regu, float regu_parameter, ActuationFuncType neuron_act_f_type):
	_gLayerType(layer_type), _gLayerDimension(layer_dimension), 
	_pDataInterface(data_interface), _nNeuronsFC(n_neurons), _nKernels(n_kernels), _gDimKernel(dimension_kernel),
	_gLearningRate(learning_rate), _gUseDropout(use_dropout), _gDropoutFactor(dropout_factor), _gRegularization(regu),
	_gRegularizationParameter(regu_parameter), _gActuationFuncType(neuron_act_f_type)
    {
    }
};


// batch vs epoch see:
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

    // extract a value from neurons and re-organize these values in matrix form,
    virtual std::vector<Images>& GetImagesActiveA()=0;
    virtual std::vector<Images>& GetImagesFullA()=0;
    // extract z value from neurons and re-organize these values in matrix form,
    virtual std::vector<Images>& GetImagesActiveZ()=0;
    virtual std::vector<Images>& GetImagesFullZ()=0;
    // extract sigma^\prime value from neurons and re-organize these values in matrix form,
    virtual std::vector<Images>& GetImagesActiveSigmaPrime()=0;
    virtual std::vector<Images>& GetImagesFullSigmaPrime()=0;
    // extract delta value from neurons and re-organize these values in matrix form,
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
    virtual void FillBatchDataToInputLayerA() = 0;
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
    virtual std::vector<Matrix>* GetWeightMatrixOriginal()=0;
    virtual std::vector<Matrix>* GetBiasVectorOriginal()=0;
    virtual std::vector<Images> & GetWeightGradients() = 0;
    virtual std::vector<Images> & GetBiasGradients() = 0;
    virtual LayerType GetType()=0;
    virtual LayerDimension GetLayerDimension()=0;
    virtual float GetDropOutFactor()=0;
    virtual std::vector<Filter2D>& GetActiveFlag()=0;
    virtual std::pair<size_t, size_t> GetOutputImageSize() = 0; // used for setup layer
    virtual int GetNumberOfNeurons() = 0;
    virtual int GetNumberOfNeuronsFC() = 0;
    virtual size_t GetNumberOfKernelsCNN() = 0;
    virtual std::pair<size_t, size_t> GetKernelDimensionCNN() = 0;
    virtual int GetID(){return __layerID;};
    virtual int GetBatchSize() = 0;
    virtual CostFuncType GetCostFuncType() = 0;
    virtual DataInterface * GetDataInterface() = 0;
    virtual Layer* GetNextLayer() = 0;
    virtual Layer* GetPrevLayer() = 0;

    // result check
    virtual void SaveAccuracyAndCostForBatch() = 0;
    virtual std::vector<float> &GetAccuracyForBatches() = 0;
    virtual std::vector<float> &GetCostForBatches() = 0;

private:
    // reserved section
    static int __layerCount;
    int __layerID = 0;
};

#endif
