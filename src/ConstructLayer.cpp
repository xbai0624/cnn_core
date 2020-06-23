#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric> // std::iota
#include <cassert>
#include <cmath>

#include "Matrix.h"
#include "ConstructLayer.h"
#include "Neuron.h"
#include "Tools.h"
#include "DataInterface.h"

using namespace std;

static double SGN(double x)
{
    if( x == 0) return 0;
    return x>0?1:-1;
}

ConstructLayer::ConstructLayer()
{
    // place holder
}

ConstructLayer::ConstructLayer(LayerParameterList p_list)
{
    // common parameters
    __type = p_list._gLayerType;

    // drop out
    if(p_list._gUseDropout)
    {
        EnableDropOut();
	__dropOutBranches = p_list._gdropoutBranches;
        SetDropOutFactor(p_list._gDropoutFactor);
    }
    else
    {
        DisableDropOut();
    }

    SetLearningRate(p_list._gLearningRate);
    __layerDimension = p_list._gLayerDimension;
    __p_data_interface = p_list._pDataInterface;
    __regularizationMethod = p_list._gRegularization;
    __regularizationParameter = p_list._gRegularizationParameter;
    __neuron_actuation_func_type = p_list._gActuationFuncType;
    __trainingType = p_list._gTrainingType;

    // parameters dependent on layer type
    if(__type == LayerType::input)
    {
        __layerDimension = p_list._gLayerDimension;
    }
    else if(__type == LayerType::fullyConnected)
    {
        SetNumberOfNeuronsFC(p_list._nNeuronsFC);
    }
    else if(__type == LayerType::cnn)
    {
        SetNumberOfKernelsCNN(p_list._nKernels);
	SetKernelSizeCNN(p_list._gDimKernel);
    }
    else if(__type == LayerType::pooling)
    {
        SetNumberOfKernelsCNN(p_list._nKernels);
	SetKernelSizeCNN(p_list._gDimKernel);
    }
    else if(__type == LayerType::output)
    {
        SetNumberOfNeuronsFC(p_list._nNeuronsFC);
    }
    else
    {
	std::cout<<__func__<<" Error: trying to construct unsupported layer."
	    <<std::endl;
	exit(0);
    }
}

ConstructLayer::ConstructLayer(LayerType t, LayerDimension layer_dimension)
{
    // for input layer
    __type = t;
    __layerDimension = layer_dimension;
}

ConstructLayer::ConstructLayer(LayerType t, int n_neurons)
{
    // for fc layer
    __type = t;
    SetNumberOfNeuronsFC((size_t)n_neurons);
}

ConstructLayer::ConstructLayer(LayerType t, int n_kernels, std::pair<size_t, size_t> d)
{
    // for cnn and pooling layer
    __type = t;
    SetNumberOfKernelsCNN((size_t)n_kernels);
    // kernel size
    SetKernelSizeCNN(d);
}

ConstructLayer::~ConstructLayer()
{
    // place holder
}

void ConstructLayer::Init()
{
    if(__type == LayerType::input) 
    {
	InitNeurons();
	InitFilters(); // input layer need to init filters, make all neurons active
    }
    else if(__type == LayerType::output) // output layer, reserved
    {
	InitNeurons();
	InitWeightsAndBias();
    }
    else  // other layer
    {
	InitNeurons();
	InitWeightsAndBias();
    }

    // setup drop out pool
    InitFilters();
    if(__use_drop_out)
    {
	SetupDropOutFilterPool();
    }
}

void ConstructLayer::EpochInit()
{
}

void ConstructLayer::Connect(Layer *prev, Layer *next)
{
    __prevLayer = prev;
    __nextLayer = next;
}

void ConstructLayer::PassDataInterface(DataInterface *data_interface)
{
    __p_data_interface = data_interface;
}

void ConstructLayer::ProcessBatch()
{
}

void ConstructLayer::PostProcessBatch()
{
}

void ConstructLayer::BatchInit()
{
    // init image dimension before each batch starts
    // the necessity of this function is originated from drop out in fc layer
    // fc layer drop out changes the dimension of the images

    // init drop out filters; reset all filters to true
    //InitFilters();

    if(__use_drop_out)  // drop out
    {
        // use different drop-out branches in a rolling sequence
        __dropOutBranchIndex++;
	__dropOutBranchIndex = __dropOutBranchIndex%__dropOutBranches;
	DropOut();
    }

    // only fc layer; cnn layer image won't change
    if(__type == LayerType::fullyConnected || __type == LayerType::output) 
    {
	UpdateCoordsForActiveNeuronFC();
    }

    // prepare weights and bias
    UpdateActiveWeightsAndBias();

    // assign active weights and bias to neurons
    AssignWeightsAndBiasToNeurons();

    // clear training information from last batch for this layer
    int batch_size = __p_data_interface->GetBatchSize();
    __imageA.resize(batch_size);
    __imageZ.resize(batch_size);
    __imageSigmaPrime.resize(batch_size);
    __imageDelta.resize(batch_size);

    __imageAFull.resize(batch_size);
    __imageZFull.resize(batch_size);
    __imageSigmaPrimeFull.resize(batch_size);
    __imageDeltaFull.resize(batch_size);
    __outputLayerCost.resize(batch_size);

    __wGradient.clear(); // these two needs to be cleared, not resized
    __bGradient.clear(); // these two needs to be cleared, not resized

    // clear training information from last batch for neurons inside this layer
    for(auto &pixel_2d: __neurons)
    {
        auto dim = pixel_2d.Dimension();
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
	        pixel_2d[i][j]->ClearPreviousBatch();
	    }
	}
    }
}

void ConstructLayer::ProcessSample()
{
}

void  ConstructLayer::SetNumberOfNeuronsFC(size_t n)
{
    if(__type == LayerType::fullyConnected || __type == LayerType::output) {
	__n_neurons_fc = n;
    }
    else {
	std::cout<<"Error: needs to set layer type before setting number of neurons."
	    <<std::endl;
	exit(0);
    }
}

void  ConstructLayer::SetNumberOfKernelsCNN(size_t n)
{
    if(__type != LayerType::cnn && __type != LayerType::pooling)
    {
	std::cout<<"Error: needs to set layer type before setting number of kernels."
	    <<std::endl;
	exit(0);
    }
    __n_kernels_cnn = n;
}

void  ConstructLayer::SetKernelSizeCNN(std::pair<size_t, size_t> s)
{
    // pooling layer also borrowed this function
    if(__type != LayerType::cnn && __type != LayerType::pooling)
    {
	std::cout<<__func__<<" Error: needs to set layer type before setting kernel size."
	    <<std::endl;
	exit(0);
    }

    __kernelDim.first = s.first;
    __kernelDim.second = s.second;
}

void  ConstructLayer::SetPrevLayer(Layer* layer)
{
    __prevLayer = layer;
} 

void  ConstructLayer::SetNextLayer(Layer* layer)
{
    __nextLayer = layer;
} 

void ConstructLayer::InitNeurons()
{
    if(__type == LayerType::cnn)
	InitNeuronsCNN();
    else if(__type == LayerType::pooling)
        InitNeuronsPooling();
    else if(__type == LayerType::input)
	InitNeuronsInputLayer();
    else if(__type == LayerType::fullyConnected)
	InitNeuronsFC();
    else if(__type == LayerType::output)
        InitNeuronsFC();
    else
    {
	std::cout<<"Error: Init neurons, unrecognized layer type."<<std::endl;
	exit(0);
    }
    //std::cout<<"Debug: Layer:"<<GetID()<<" init neruons done."<<std::endl;

    // after initializing all neurons, setup neuron dimension information
    // setup total neuron dimension
    size_t k = __neurons.size();
    size_t i = 0;
    size_t j = 0;
    if(k > 0)
    {
	i = __neurons[0].Dimension().first;
	j = __neurons[0].Dimension().second;

	__neuronDim.k = k;
	__neuronDim.i = i;
	__neuronDim.j = j;
    }

    // setup neuron layer information
    for(size_t kk=0;kk<k;kk++)
    {
	for(size_t ii=0;ii<i;ii++)
	    for(size_t jj=0;jj<j;jj++)
	    {
		__neurons[kk][ii][jj]->SetLayer(dynamic_cast<Layer*>(this));
		//__neurons[kk][ii][jj]->SetPreviousLayer(__prevLayer); // obsolete
		//__neurons[kk][ii][jj]->SetNextLayer(__nextLayer);     // obsolete
	    }
    }
}

void ConstructLayer::InitNeuronsCNN()
{
    // clear
    __neurons.clear();

    // get output image size of previous layer (input image size for this layer)
    assert(__prevLayer != nullptr);
    auto size_prev_layer = __prevLayer->GetOutputImageSize();
    //std::cout<<__func__<<" prev layer output image size: "<<size_prev_layer<<std::endl;

    int n_row = size_prev_layer.first;
    int n_col = size_prev_layer.second;

    assert(__cnnStride >= 1);
    // deduct output image dimension
    int x_size = (int)n_row - (int)__kernelDim.first + 1;
    int y_size = (int)n_col - (int)__kernelDim.second + 1;
    if(x_size <= 0) x_size = 1; // small images will be complemented by padding
    if(y_size <= 0) y_size = 1; // so it is safe to set size >= 1
    __outputImageSizeCNN.first = x_size;
    __outputImageSizeCNN.second = y_size;

    if(__cnnStride > 1)
    {
	__outputImageSizeCNN.first = (int)__outputImageSizeCNN.first / (int) __cnnStride + 1;
	__outputImageSizeCNN.second = (int)__outputImageSizeCNN.second / (int) __cnnStride + 1;
    }
    assert(__outputImageSizeCNN.first >=1);
    assert(__outputImageSizeCNN.second >=1);

    for(size_t k=0;k<__n_kernels_cnn;k++)
    {
	Pixel2D<Neuron*> image(__outputImageSizeCNN.first, __outputImageSizeCNN.second);
	for(size_t i=0;i<__outputImageSizeCNN.first;i++)
	{
	    for(size_t j=0;j<__outputImageSizeCNN.second;j++)
	    {
		Neuron *n = new Neuron();
		n->SetCoord(i, j, k); // set neuron coord
		n->SetActuationFuncType(__neuron_actuation_func_type);
		image[i][j] = n;
	    }
	}
	__neurons.push_back(image);
    }
}

void ConstructLayer::InitNeuronsPooling()
{
    // clear
    __neurons.clear();

    // get output image size of previous layer (input image size for this layer)
    assert(__prevLayer != nullptr);
    auto size_prev_layer = __prevLayer->GetOutputImageSize();
    //std::cout<<__func__<<" prev layer output image size: "<<size_prev_layer<<std::endl;

    int n_row = size_prev_layer.first;
    int n_col = size_prev_layer.second;

    assert(__prevLayer->GetNumberOfKernelsCNN() == __n_kernels_cnn);

    // deduct output image dimension
    //     !!! NOTE !!! pooling layer is different with cnn layer
    //     !!!          pooling layer kernel should have no coverage overlap on the input image
    int x_size = (int)n_row / (int)__kernelDim.first;
    int y_size = (int)n_col / (int)__kernelDim.second;
    if(x_size <= 0) x_size = 1; // small images will be complemented by padding
    if(y_size <= 0) y_size = 1; // so it is safe to set size >= 1

    if((int)n_row%(int)__kernelDim.first != 0) x_size += 1; // residue will be complemented by padding
    if((int)n_col%(int)__kernelDim.second != 0) y_size += 1; // residue will be complemented by padding

    // pooling layer borrowed variabe '__outputImageSizeCNN', to reduce memory
    __outputImageSizeCNN.first = x_size;
    __outputImageSizeCNN.second = y_size;
    assert(__outputImageSizeCNN.first >=1);
    assert(__outputImageSizeCNN.second >=1);

    for(size_t k=0;k<__n_kernels_cnn;k++)
    {
	Pixel2D<Neuron*> image(__outputImageSizeCNN.first, __outputImageSizeCNN.second);
	for(size_t i=0;i<__outputImageSizeCNN.first;i++)
	{
	    for(size_t j=0;j<__outputImageSizeCNN.second;j++)
	    {
		Neuron *n = new Neuron();
		n->SetCoord(i, j, k); // set neuron coord
		n->SetActuationFuncType(__neuron_actuation_func_type);
		image[i][j] = n;
	    }
	}
	__neurons.push_back(image);
    }
}

void ConstructLayer::InitNeuronsFC()
{
    __neurons.clear();
    assert(__n_neurons_fc >= 1);
    Pixel2D<Neuron*> image(__n_neurons_fc, 1);
    for(size_t i=0;i<__n_neurons_fc;i++)
    {
	Neuron *n = new Neuron();
	n->SetCoord(i, 0, 0);
	n->SetActuationFuncType(__neuron_actuation_func_type);
	image[i][0] = n;
    }
    __neurons.push_back(image);
}


void ConstructLayer::InitNeuronsInputLayer()
{
    __neurons.clear();

    if(__p_data_interface == nullptr)
    {
        std::cout<<__func__<<" Error: must implement/pass DataInterface class before initializing neurons for input layer"
	         <<std::endl;
        exit(0);
    }

    assert(__layerDimension != LayerDimension::Undefined);

    auto dim = __p_data_interface->GetDataDimension();

    // 1):
    // for 1D (fc) input layer; 1D input layer needs fake neurons, which will be used by
    //     its following layer to setup weight matrix dimension
    if(__layerDimension == LayerDimension::_1D)
    {
	assert(dim.second == 1); // make sure matrix transformation has been done
	//cout<<"Info::input layer dimension: "<<dim<<endl;
	Pixel2D<Neuron*> image(dim.first, dim.second);
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		Neuron *n = new Neuron();
		// n->SetCoord(i, j, k); // input layer neuron does not need coord
		n->SetActuationFuncType(__neuron_actuation_func_type);
		image[i][0] = n;
	    }
	}
	__neurons.push_back(image);
    }

    // 2):
    // for 2D input layer, 2D input layer does not need fake neurons, instead
    //     its following layers will directly get 'A' images from 2D input layer
    else if(__layerDimension == LayerDimension::_2D)
    {
        // 2D layers neurons number should equal to dim.first * dim.second
	size_t total_neurons = dim.first * dim.second;
	Pixel2D<Neuron*> image(total_neurons, 1);
	for(size_t i=0;i<total_neurons;i++)
	{
	    for(size_t j=0;j<1;j++)
	    {
		Neuron *n = new Neuron();
		// n->SetCoord(i, j, k); // input layer neuron does not need coord
		image[i][0] = n;
	    }
	}
	__neurons.push_back(image);
    }
}



void ConstructLayer::InitFilters()
{
    // clear previous filter
    __activeFlag.clear();

    // init filter 2d matrix, fill true value to all elements
    if(__type == LayerType::input)
    {
	if(__layerDimension == LayerDimension::_1D)
	{
	    assert(__neurons.size() == 1); // only one kernel
	    auto dim = __neurons[0].Dimension();
	    Filter2D f(dim.first, dim.second); // default to true
	    __activeFlag.push_back(f);
	}
	else if(__layerDimension == LayerDimension::_2D)
	{
	    // 2D input layer need a fake filter with dimension (number_of_fake_neurons, 1), similar like 1D
	    // this is just in case one directly connect a 2D input layer to a 1D fc layer
	    //
	    // since filters in 2D middle layers are not used, so it is safe to code like 1D case
	    assert(__neurons.size() == 1); // only one kernel
	    auto dim = __neurons[0].Dimension(); // for 2D layer, the fake neurons has also been initialized
	    Filter2D f(dim.first, dim.second); // default to true
	    __activeFlag.push_back(f);
	}
    }
    else if(__type == LayerType::output)
    {
	assert(__neurons.size() == 1); // only one kernel
	auto dim = __neurons[0].Dimension();
	assert(dim.second == 1); // only one collum
	Filter2D f(dim.first, dim.second); // default to true
	__activeFlag.push_back(f);
    }
    else if(__type == LayerType::fullyConnected)
    {
	assert(__neurons.size() == 1);
	auto dim = __neurons[0].Dimension();
	Filter2D f(dim.first, dim.second); // default to true
	__activeFlag.push_back(f);
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling)
    {
	size_t nKernels = __weightMatrix.size();
	for(size_t i=0;i<nKernels;i++)
	{
	    auto dim = __weightMatrix[i].Dimension();
	    Filter2D f(dim.first, dim.second); // default to true
	    __activeFlag.push_back(f);
	}
    }
}

void  ConstructLayer::InitWeightsAndBias()
{
    //std::cout<<"Layer id: "<<GetID()<<", W initialization."<<endl;
    // init weights and bias
    // no need to init active w&b, they will be filled in Batch starting phase
    // clear everything
    __weightMatrix.clear();
    __biasVector.clear();

    if(__trainingType == TrainingType::ResumeTraining)
    {
        LoadTrainedWeightsAndBias();
        return;
    }

    if(__p_data_interface == nullptr)
    {
        std::cout<<__func__<<" error: layer Need DataInterface already set."
	         <<std::endl;
        exit(0);
    }
    //int batch_size = __p_data_interface->GetBatchSize();
    //int N = __p_data_interface->GetNumberOfBatches();

    // total_entries is used to randomly initialize weight matrix, 
    // using a Gaussian distribution, wieth mean=0, sigma = 1/sqrt(total_entries)
    //double total_entries = (double)batch_size * (double)N; // use this one, it is better (incorrect)
    //double total_entries = (double)batch_size;          // this one is not as good as the above one (incorrect)

    if(__type == LayerType::fullyConnected || __type == LayerType::output) // output layer is also a fully connected layer
    {
	int n_prev = 0;
	int n_curr = GetNumberOfNeurons();
	if(__prevLayer == nullptr) // input layer
	{
	    std::cout<<"ERROR WARNING: layer"<<GetID()
		<<" has no prev layer, default 10 neruons for prev layer was used."
		<<std::endl;
	    n_prev = 10;
	}
	else 
	    n_prev = __prevLayer->GetNumberOfNeurons();

        // weight initialization
	Matrix w(n_curr, n_prev);
	if(GetNeuronActuationFuncType() != ActuationFuncType::Relu)
	{
	    // for non-ReLu actuation functions, use Xavier initialization
	    // random with a normal distribution with sigma = 1/sqrt(number of fan-in neurons)
	    w.RandomGaus(0., 1./sqrt((double)n_prev)); // (0, sqrt(n_neuron)) normal distribution

	    //w.RandomGaus(0., 1./sqrt((double)total_entries)); // (0, sqrt(n_neuron)) normal distribution
	}
	else
	{
	    // for ReLu actuation functions, use Kaiming He method: https://arxiv.org/pdf/1502.01852.pdf
	    // step 1): initialize the matrix with a standard normal distribution
	    w.RandomGaus(0., 1.);

	    // step 2): then hadamard the weight matrix with a number sqrt(2/n)
	    //          where n is the fan-in neurons (number of neurons in previous layer)
	    w = w*sqrt(2./(double)n_prev);
	}
	__weightMatrix.push_back(w);

        // bias initialization
	Matrix b(n_curr, 1);
	if(GetNeuronActuationFuncType() != ActuationFuncType::Relu)
	{
	    // for non-ReLu neurons, use Xavier initialization, initialize it to 0
	    b = b * 0.;
	    //b.RandomGaus(0., 1.); // (0, 1) normal distribution
	}
	else
	{
	    // for ReLu neurons, use Kaiming He method, initialize bias to 0
            b = b*0.;
	}
	__biasVector.push_back(b);
    }
    else if(__type == LayerType::cnn)
    {
	//auto norm_factor = __p_data_interface->GetBatchSize();
	for(size_t i=0;i<__n_kernels_cnn;i++)
	{
	    Matrix w(__kernelDim);
	    // weight matrix initialization
	    if(GetNeuronActuationFuncType() != ActuationFuncType::Relu)
	    {
	        // for non-ReLu neurons, use Xavier initialization
		int fan_in_neurons = __prevLayer->GetNumberOfNeurons();
		w.RandomGaus(0., 1./sqrt((double)fan_in_neurons));
		//w.RandomGaus(0., 1./sqrt((double)total_entries));
	    }
	    else
	    {
	        // for ReLu neurons, use Kaiming He method
		w.RandomGaus(0., 1.);
		int fan_in_neurons = __prevLayer->GetNumberOfNeurons();
		w = w * sqrt(2./ (double) fan_in_neurons);
	    }
	    __weightMatrix.push_back(w);

	    Matrix b(1, 1);
	    // bias matrix initialization
	    if(GetNeuronActuationFuncType() != ActuationFuncType::Relu)
	    {
	        // for non-ReLu neurons, use Xavier initialization
		// Xavier also require b to be initialized with 0
		b = b * 0.;
		//b.RandomGaus(0., 1.); // (0, 1) normal distribution
	    }
	    else 
	    {
	        // for ReLu neurons, use Kaiming He method, set bias to 0.
	        b = b * 0.;
	    }
	    __biasVector.push_back(b);
	}
    }
    else if(__type == LayerType::pooling)
    {
        for(size_t i=0;i<__n_kernels_cnn;i++) // pooling used cnn symbol
	{
	    Matrix w(__kernelDim, 1); // pooling layer weight and bias matrix are not used, so set them to 1
	    __weightMatrix.push_back(w);

	    Matrix b(1, 1, 0);
	    __biasVector.push_back(b);
	}
    }
    else {
	std::cout<<"Error: need layer type info before initing w&b."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::ForwardPropagateForSample(int sample_index)
{
    // forward propagation: 
    //     ---) compute Z, A, A'(Z) for this layer
    //          these works are done neuron by neuron (neuron level)

    // layers needs to finish all z's first 
    // this special treatment is for softmax actuation functions in output layer
    // b/c in softmax, one need all Z's already calculated before updating A and sigma^\prime
    // step 1) finish z
    for(size_t k=0;k<__neurons.size();k++)
    {
	auto dim = __neurons[k].Dimension();
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		//cout<<"coord (i, j, k): ("<<i<<", "<<j<<", "<<k<<")"<<endl;
		if(!__neurons[k][i][j]->IsActive()) continue;
		__neurons[k][i][j] -> UpdateZ(sample_index);
	    }
	}
    }
    UpdateImagesZ(sample_index);
    // step 2) then  finish a and sigma^\prime
    for(size_t k=0;k<__neurons.size();k++)
    {
	auto dim = __neurons[k].Dimension();
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		//cout<<"coord (i, j, k): ("<<i<<", "<<j<<", "<<k<<")"<<endl;
		if(!__neurons[k][i][j]->IsActive()) continue;
		__neurons[k][i][j] -> UpdateA(sample_index);
		__neurons[k][i][j] -> UpdateSigmaPrime(sample_index); // sigma^\prime is not needed in layer, but needed in neuron
	    }
	}
    }
    UpdateImagesA(sample_index);

    // after finished A and Z matrix, get Sigma^\prime matrix
    UpdateImagesSigmaPrime(sample_index);
}

void ConstructLayer::BackwardPropagateForSample(int sample_index)
{
    // backward propagation:
    //     ---) compute delta for this layer
    //     ---) only after all samples in this batch finished forward propagation, one can do this backward propagation

    if(__type == LayerType::output)
    {
	ComputeCostInOutputLayerForCurrentSample(sample_index);
    } 
    else 
    {
	for(size_t k=0;k<__neurons.size();k++)
	{
	    auto dim = __neurons[k].Dimension();
	    for(size_t i=0;i<dim.first;i++)  {
		for(size_t j=0;j<dim.second;j++) {
		    if(__neurons[k][i][j]->IsActive())
		    {
			__neurons[k][i][j] -> BackwardPropagateForSample(sample_index);
			/*
			   if(__type == LayerType::fullyConnected)
			   {
			   cout<<__func__<<" fc layer neuron coord: "<<__neurons[k][i][j]->GetCoord()<<endl;
			   }
			   else if(__type == LayerType::cnn)
			   {
			   cout<<__func__<<" cnn layer neuron coord: "<<__neurons[k][i][j]->GetCoord()<<endl;
			   }
			   */
		    }
		}
	    }
	}
	// when propagation for this layer is done, update the Delta matrices
	UpdateImagesDelta(sample_index);
    }
}

//#include <mutex>
//mutex mtx;

// cost functions ---------- cross entropy 
static double cross_entropy(Matrix &A, Matrix &Y)
{
    auto dim = A.Dimension();
    assert(dim == Y.Dimension());
    assert(dim.second == 1);

    double res = 0.;
    for(size_t i=0;i<dim.first;i++)
    {
	if((double)A[i][0] <= 1e-10 || (double)A[i][0] >= 1-1e-10) res += 0;
	else
	    //res += Y[i][0] * log(A[i][0]) + (1. - Y[i][0]) * log(1. - A[i][0]);
	    res += (double)Y[i][0] * log((double)A[i][0]); 
    }

    return res; // no minus symbol here, 
                // need to add a - sign when computing the cost for this batch
}

// cost functions ---------- log likelihood
static double log_likelihood(Matrix &, Matrix &)
{
    // this one works for softmax layer
    // details to be implemented

    // Y should be one-hot vector
    return 0;	
}

// cost functions ---------- quadratic sum
static double quadratic_sum(Matrix &A, Matrix &Y)
{
    // this one is only for research test, not used in reality
    auto dim = A.Dimension();
    assert(dim == Y.Dimension());
    assert(dim.second == 1);

    double res = 0.;
    for(size_t i=0;i<dim.first;i++)
    {
	res += (A[i][0] - Y[i][0]) *  (A[i][0] - Y[i][0]);
    }

    return res; // no minus symbol here, 
                // need to add a - sign when computing the cost for this batch
}

void ConstructLayer::ComputeCostInOutputLayerForCurrentSample(int sample_index)
{
    if(__type != LayerType::output)
    {
        cout<<"Error: ComputeCostInOutputLayerForCurrentSample() only works for output layer."
	    <<endl;
        exit(0);
    }

    // --- 2) compute the cost function C(a_i, y_i)
    Images sample_image = __imageA[sample_index]; // output layer drop out is not used for sure, so use __imageA is OK.
    assert(sample_image.GetNumberOfKernels() == 1); // output layer must be a fully connected layer, so one kernel
    // now get a_i
    Matrix sample_A = sample_image.OutputImageFromKernel[0];
    assert(sample_A.Dimension().second == 1); // must be a collum matrix
    //cout<<sample_A<<endl;

    //assert(sample_number >= 1); // obsolete
    //Matrix sample_label = (__p_data_interface->GetCurrentBatchLabel())[sample_number-1];
    Matrix sample_label = (__p_data_interface->GetCurrentBatchLabel())[sample_index];
    assert(sample_label.Dimension()  == sample_A.Dimension());

    double cost = 0.;
    if(__cost_func_type == CostFuncType::cross_entropy)
    {
	cost = cross_entropy(sample_A, sample_label);
    }
    else if(__cost_func_type == CostFuncType::log_likelihood)
    {
	cost = log_likelihood(sample_A, sample_label);
    }
    else if(__cost_func_type == CostFuncType::quadratic_sum)
    {
	cost = quadratic_sum(sample_A, sample_label);
    }
    else {
	cout<<"Error: cost function only supports cross_entropy, loglikelihood, quadratic_sum"
	    <<endl;
	exit(0);
    }
    // push cost for current sample to memory
    //__outputLayerCost.push_back(cost);
    __outputLayerCost[sample_index]= cost;
    //cout<<"sample cost: "<<cost<<endl;
    //getchar();


    // --- 3) then calculate delta: delta = delta(a_i, y_i) for this sample
    // -------- please note: this function only calculate \delta for current sample, 
    // ----------- when doing back propagation for hidden layers, only delta with the same sample_index should be updated 
    // ----------- the overall delta for this batch should be an average of all deltas in this batch
    // ----------- this is due to the characteristic of Cost function, which is an averaged sum over all samples

    Matrix delta = Matrix(sample_A.Dimension());
    if(__cost_func_type == CostFuncType::cross_entropy)
    {
	delta = sample_A - sample_label; // softmax and sigmoid all have this form
    }
    else if(__cost_func_type == CostFuncType::log_likelihood)
    {
	delta = sample_A - sample_label;
    }
    else if(__cost_func_type == CostFuncType::quadratic_sum)
    {
	delta = sample_A - sample_label; // this is a place holder, not used
    }
    else {
	cout<<"Error: cost function only supports cross_entropy, loglikelihood, quadratic_sum"
	    <<endl;
	exit(0);
    }
    // push cost for current sample to memory
    Images images_delta_from_current_sample;
    images_delta_from_current_sample.OutputImageFromKernel.push_back(delta); // only one kernel in fc layer
    __imageDelta[sample_index]=images_delta_from_current_sample;
    __imageDeltaFull[sample_index]=images_delta_from_current_sample; // in output layer, dropout is not used for sure
}

std::vector<Images>& ConstructLayer::GetImagesActiveA()
{
    return __imageA;
}

std::vector<Images>& ConstructLayer::GetImagesFullA()
{
    return __imageAFull;
}

std::vector<Images>& ConstructLayer::GetImagesActiveSigmaPrime()
{
    return __imageSigmaPrime;
}

std::vector<Images>& ConstructLayer::GetImagesFullSigmaPrime()
{
    return __imageSigmaPrimeFull;
}

void ConstructLayer::FillBatchDataToInputLayerA()
{
    // if this layer is input layer, then fill the 'a' matrix directly with input image data
    std::vector<Matrix> & input_data = __p_data_interface->GetCurrentBatchData();
    //cout<<">>>: "<<input_data.size()<<" samples in current batch from data interface"<<endl;

    // input data whitening
    if(__input_data_batch_whitening)
	Matrix::BatchNormalization(input_data);

    // first clear the previous batch
    __imageA.clear(); // input layer dropout is not used
    __imageAFull.clear();

    // load all batch data to memory, this should be faster
    for(auto &i: input_data)
    {
	Images image_a; // one image

	// input layer only has one kernel
	image_a.OutputImageFromKernel.push_back(i);

	// push this image to images of this batch
	__imageA.push_back(image_a);
	__imageAFull.push_back(image_a);
    }

    cout<<">>>: "<<__imageA.size()<<" samples in current batch."<<endl;
}

void ConstructLayer::ClearUsedSampleForInputLayer_obsolete()
{
    // this function not needed anymore
    if(__type != LayerType::input)
    {
        cout<<"Error: Clear used sample for input layer only works for input layer..."<<endl;
	exit(0);
    }
    if(__imageA.size() <= 0)
    {
        cout<<"Error: ClearUsedSampleForInputLayer(): __imageA already empty."<<endl;
	exit(0);
    }

    __imageA.pop_back();
}

static Matrix filterMatrix(Matrix &A, Filter2D &F)
{
    // this function takes out all active elements from A according to F
    //       and return them in another matrix
    //       the filter info is given in matrix F
    auto dimA = A.Dimension();
    auto dimF = F.Dimension();
    assert(dimA == dimF);

    vector<vector<double>> R;
    for(size_t i=0;i<dimA.first;i++)
    {
	vector<double> _tmp_row;
	for(size_t j=0;j<dimA.second;j++)
	{
	    if(F[i][j]==1)
	    {
		_tmp_row.push_back(A[i][j]);
	    }
	    else if(F[i][j] != 0)
	    {
		std::cout<<"Error: filter matrix element value must be 0 or 1"<<std::endl;
		exit(0);
	    }
	}
	if(_tmp_row.size() > 0)
	    R.push_back(_tmp_row);
    }
    Matrix Ret(R);
    return Ret;
}

void ConstructLayer::UpdateImagesA(int sample_id)
{
    // __imageA shold only store images from all active neurons, the matching info can be achieved from filter matrix
    //    drop out only happens on batch level
    //    so imageA will clear after each batch is done
    //    **** on batch level, the filter matrix stays the same, so no need to worry the change of filter matrix inside a batch
    //size_t l = __imageA.size();
    //cout<<" >>> image in lyaer id: "<<GetID()<<" size: "<<l<<endl;

    // extract the A matrices from neurons for current traning sample
    Images sample_image_A;
    Images sample_image_A_full;

    if(__type == LayerType::fullyConnected || __type == LayerType::output) 
    {
	// for fully connected layer; output layer is also a fully connected layer
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix A( __neuronDim.i, __neuronDim.j, 0);
	    //cout<<"A before filling"<<endl<<A<<endl;
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto a_vector = __neurons[k][i][j]->GetAVector();
		    if(__neurons[k][i][j]->IsActive())
		    {  // make sure no over extract
			//cout<<a_vector.size()<<"......"<<l<<endl;
			//assert(a_vector.size() - 1 == l); // obsolete
			//A[i][j] = a_vector.back();        // obsolete
			A[i][j] = a_vector[sample_id];
		    }
		    else
		    {
		        // if neuron is inactive, set it to 0
			A[i][j] = 0;
		    }
		}
	    }
	    // save full image
	    sample_image_A_full.OutputImageFromKernel.push_back(A);
	    //cout<<"A after filling: "<<endl<<A<<endl;

	    // only save active elements
	    Matrix R = filterMatrix(A, __activeFlag[k]);
	    sample_image_A.OutputImageFromKernel.push_back(R);

	    assert(R.Dimension().first  == __activeNeuronDim.i);
	    assert(R.Dimension().second == __activeNeuronDim.j);
	}
	//__imageA.push_back(sample_image_A); // obsolete
	__imageA[sample_id] = sample_image_A;
	__imageAFull[sample_id] = sample_image_A_full;
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling ) // for cnn layer and pooling layer
    {
	// for cnn, drop out happens on kernels (weight matrix)
	// so the neurons are all active
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix A( __neuronDim.i, __neuronDim.j);
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto a_vector = __neurons[k][i][j]->GetAVector();
		    if(__neurons[k][i][j]->IsActive())
		    {  // make sure no over extract
			//assert(a_vector.size() - 1 == l); // obsolete
			//A[i][j] = a_vector.back();        // obsolete
			A[i][j] = a_vector[sample_id];
		    }
		    else
		    {
			A[i][j] = 0;
		    }
		}
	    }
	    sample_image_A.OutputImageFromKernel.push_back(A); // no need to filter
	    sample_image_A_full.OutputImageFromKernel.push_back(A); // no need to filter
	}
	//__imageA.push_back(sample_image_A);
	__imageA[sample_id] = sample_image_A;
	__imageAFull[sample_id] = sample_image_A;
    }
    else // reserved for other layer types
    {
    }
}


void ConstructLayer::UpdateImagesSigmaPrime(int sample_id)
{
    // __imageSigmaPrime shold only store images from all active neurons, the matching info can be achieved from filter matrix
    // __imageSigmaPrimeFull stores images from all neurons (active + inactive), with inactive set to 0.
    //    **** on batch level, the filter matrix stays the same, so no need to worry the change of filter matrix inside a batch

    // extract the sigma^\prime matrices from neurons for current traning sample
    Images sample_image_SigmaPrime;
    Images sample_image_SigmaPrime_full;

    if(__type == LayerType::fullyConnected || __type == LayerType::output) 
    {
	// for fully connected layer; output layer is also a fully connected layer
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix SigP( __neuronDim.i, __neuronDim.j, 0);
	    //cout<<"A before filling"<<endl<<A<<endl;
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto s_vector = __neurons[k][i][j]->GetSigmaPrimeVector();
		    if(__neurons[k][i][j]->IsActive())
		    {
			SigP[i][j] = s_vector[sample_id];
		    }
		    else
		    {
		        // if neuron is inactive, set it to 0
			SigP[i][j] = 0;
		    }
		}
	    }
	    // save full image
	    sample_image_SigmaPrime_full.OutputImageFromKernel.push_back(SigP);
	    //cout<<"A after filling: "<<endl<<A<<endl;

	    // only save active elements
	    Matrix R = filterMatrix(SigP, __activeFlag[k]);
	    sample_image_SigmaPrime.OutputImageFromKernel.push_back(R);

	    assert(R.Dimension().first  == __activeNeuronDim.i);
	    assert(R.Dimension().second == __activeNeuronDim.j);
	}
	//__imageA.push_back(sample_image_A); // obsolete
	__imageSigmaPrime[sample_id] = sample_image_SigmaPrime;
	__imageSigmaPrimeFull[sample_id] = sample_image_SigmaPrime_full;
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling ) // for cnn layer and pooling layer
    {
	// for cnn, drop out happens on kernels (weight matrix)
	// so the neurons are all active
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix SigP( __neuronDim.i, __neuronDim.j);
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto s_vector = __neurons[k][i][j]->GetSigmaPrimeVector();
		    if(__neurons[k][i][j]->IsActive())
		    { 
			SigP[i][j] = s_vector[sample_id];
		    }
		    else
		    {
			SigP[i][j] = 0;
		    }
		}
	    }
	    sample_image_SigmaPrime.OutputImageFromKernel.push_back(SigP); // no need to filter
	    sample_image_SigmaPrime_full.OutputImageFromKernel.push_back(SigP); // no need to filter
	}
	//__imageA.push_back(sample_image_A);
	__imageSigmaPrime[sample_id] = sample_image_SigmaPrime;
	__imageSigmaPrimeFull[sample_id] = sample_image_SigmaPrime;
    }
    else // reserved for other layer types
    {
    }
}


std::vector<Images>& ConstructLayer::GetImagesActiveZ()
{
    return __imageZ;
}

std::vector<Images>& ConstructLayer::GetImagesFullZ()
{
    return __imageZFull;
}

void ConstructLayer::UpdateImagesZ(int sample_id)
{
    // __imageZ shold only store images from all active neurons, the matching info can be achieved from filter matrix
    //    drop out only happens on batch level
    //    so imageZ will clear after each batch is done
    //    **** on batch level, the filter matrix stays the same, so no need to worry the change of filter matrix on batch level

    // extract the A matrices from neurons for current traning sample
    Images sample_image_Z;
    Images sample_image_Z_full;

    if(__type == LayerType::fullyConnected || __type == LayerType::output) // for fully connected layer; output layer is also a fully connected layer
    {
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix Z( __neuronDim.i, __neuronDim.j, 0);
	    //cout<<"Z before filling"<<endl<<Z<<endl;
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto z_vector = __neurons[k][i][j]->GetZVector();
		    if(__neurons[k][i][j]->IsActive())
		    {  // make sure no over extract
			//assert(z_vector.size() - 1 == l); // obsolete
			//Z[i][j] = z_vector.back();        // obsolete
			Z[i][j] = z_vector[sample_id];
		    }
		    else
		    {
		        // if neuron is inactive, set it to 0
			Z[i][j] = 0;
		    }
		}
	    }
	    //cout<<"Z after filing: "<<endl<<Z<<endl;
	    // save full image
	    sample_image_Z_full.OutputImageFromKernel.push_back(Z);
	    // only save active elements
	    Matrix R = filterMatrix(Z, __activeFlag[k]);
	    sample_image_Z.OutputImageFromKernel.push_back(R);

	    assert(R.Dimension().first  == __activeNeuronDim.i);
	    assert(R.Dimension().second == __activeNeuronDim.j);
	}
	//__imageZ.push_back(sample_image_Z);
	__imageZ[sample_id] = sample_image_Z;
	__imageZFull[sample_id] = sample_image_Z_full;
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling) // for cnn layer and pooling
    {
	// for cnn, drop out happens on kernels (weight matrix)
	// so the neurons are all active
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix Z( __neuronDim.i, __neuronDim.j);
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto z_vector = __neurons[k][i][j]->GetZVector();
		    if(__neurons[k][i][j]->IsActive())
		    {  // make sure no over extract
			//assert(z_vector.size() - 1 == l); // obsolete
			//Z[i][j] = z_vector.back();        // obsolete
			Z[i][j] = z_vector[sample_id];
		    }
		    else
		    {
			Z[i][j] = 0;
		    }
		}
	    }
	    sample_image_Z.OutputImageFromKernel.push_back(Z); // no need to filter
	    sample_image_Z_full.OutputImageFromKernel.push_back(Z); // no need to filter
	}
	//__imageZ.push_back(sample_image_Z);
	__imageZ[sample_id] = sample_image_Z;
	__imageZFull[sample_id] = sample_image_Z_full;
    }
    else // for other layer types
    {
        // reserved for future types of layers
    }
}


std::vector<Images>& ConstructLayer::GetImagesActiveDelta()
{
    return __imageDelta;
}

std::vector<Images>& ConstructLayer::GetImagesFullDelta()
{
    return __imageDeltaFull;
}


void ConstructLayer::UpdateImagesDelta(int sample_index)
{
    // __imageDelta shold only store images from all active neurons, the matching info can be achieved from filter matrix
    //    drop out only happens on batch level
    //    so imageDelta will clear after each batch is done
    //    **** on batch level, the filter matrix stays the same, so no need to worry the change of filter matrix on batch level
    //
    //    The above comment is copied from UpdateImagesZ(); just to refresh memory, no use in this function
    //

    // extract the Delta matrices from neurons for current traning sample
    Images sample_image_delta;
    Images sample_image_delta_full;

    // !!! Note:: !!! delta for output layer needs different method, and is done ComputeCostInOutputLayerForCurrentSample(int) function
    //                no need to do it here
    if(__type == LayerType::fullyConnected) // for fully connected layer, 
    {
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix Delta( __neuronDim.i, __neuronDim.j, 0);
	    //cout<<"sample index: "<<sample_index<<endl;
	    //cout<<"before filling delta"<<endl<<Delta<<endl;
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto delta_vector = __neurons[k][i][j]->GetDeltaVector();
		    if(__neurons[k][i][j]->IsActive())
		    {  // make sure no over extract
			//assert(delta_vector.size() - 1 == l); // obsolete
			//Delta[i][j] = delta_vector.back();    // obsolete
			Delta[i][j] = delta_vector[sample_index];    // obsolete
		    }
		    else
		    {
		        // if neuron is inactive, set it to 0
			Delta[i][j] = 0;
		    }
		}
	    }
	    // save full delta image
	    sample_image_delta_full.OutputImageFromKernel.push_back(Delta);
	    //cout<<"full image delta: "<<endl<<Delta<<endl;
	    //getchar();
	    // only save active elements
	    Matrix R = filterMatrix(Delta, __activeFlag[k]);
	    sample_image_delta.OutputImageFromKernel.push_back(R);

	    assert(R.Dimension().first  == __activeNeuronDim.i);
	    assert(R.Dimension().second == __activeNeuronDim.j);
	}
	//__imageDelta.push_back(sample_image_delta);
	__imageDelta[sample_index] = sample_image_delta;
	__imageDeltaFull[sample_index] = sample_image_delta_full;
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling) // for cnn layer and pooling layer
    {
	// for cnn, drop out happens on kernels (weight matrix)
	// so the neurons are all active
	for(size_t k=0;k<__neuronDim.k;k++) // kernel
	{
	    Matrix Delta( __neuronDim.i, __neuronDim.j);
	    for(size_t i=0;i<__neuronDim.i;i++){
		for(size_t j=0;j<__neuronDim.j;j++)
		{
		    auto delta_vector = __neurons[k][i][j]->GetDeltaVector();
		    if(__neurons[k][i][j]->IsActive())
		    {  // make sure no over extract
			//assert(delta_vector.size() - 1 == l);
			//Delta[i][j] = delta_vector.back();
			Delta[i][j] = delta_vector[sample_index];
		    }
		    else
		    {
			Delta[i][j] = 0;
		    }
		}
	    }
	    sample_image_delta.OutputImageFromKernel.push_back(Delta); // no need to filter
	    sample_image_delta_full.OutputImageFromKernel.push_back(Delta); 
	}
	//__imageDelta.push_back(sample_image_delta); // obsolete
	__imageDelta[sample_index] = sample_image_delta;
	__imageDeltaFull[sample_index] = sample_image_delta_full;
    }
    else // reserved for other layer types
    {
    }
}

void ConstructLayer::UpdateCoordsForActiveNeuronFC()
{
    // enable/disable neuron; and
    // update coords for active neuron
    assert(__neurons.size() == 1); 
    auto dim = __neurons[0].Dimension();
    //cout<<"Neuron dimension in FC layer: "<<dim<<endl;
    assert(dim.second == 1);

    // get filter
    assert(__activeFlag.size() == 1);

    size_t active_i = 0;
    for(size_t i=0;i<dim.first;i++)
    {
	// first reset coords back
	//__neurons[0][i][0]->SetCoord(0, i, 0); //SetCoord(i, j, k), please note the sequence of the coordinates
	__neurons[0][i][0]->SetCoord(i, 0, 0);   //SetCoord(i, j, k), 

	// then set coord according to filter mask
	if(!__activeFlag[0][i][0]){
	    __neurons[0][i][0]->Disable();
	    continue;
	}
	__neurons[0][i][0]->Enable();
	//__neurons[0][i][0]->SetCoord(0, active_i, 0);
	__neurons[0][i][0]->SetCoord(active_i, 0, 0); // SetCoord(i, j, k)
	active_i++;
    }
}

void ConstructLayer::UpdateActiveWeightsAndBias()
{
    // clear active weights and bias from previous batch
    __weightMatrixActive.clear();
    __biasVectorActive.clear();

    // update active weights and bias for this batch
    TransferValueFromOriginalToActive_WB();
}

void ConstructLayer::AssignWeightsAndBiasToNeurons()
{
    // pass active weights and bias pointers to neurons
    if(__type == LayerType::fullyConnected || __type == LayerType::output) // output is also a fully connected layer
    {
	// assert(__weightMatrixActive.size() == 1); // should be equal to number of active neurons
	assert(__neurons.size() == 1);
	auto dim = __neurons[0].Dimension();
	assert(dim.second == 1);

	size_t active_i = 0;
	for(size_t i=0;i<dim.first;i++)
	{
	    if(!__neurons[0][i][0]->IsActive())
		continue;
	    __neurons[0][i][0] -> PassWeightPointer(&__weightMatrixActive[active_i]);
	    __neurons[0][i][0] -> PassBiasPointer(&__biasVectorActive[active_i]);

	    active_i++;
	}
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling)
    {
	size_t nKernel = __weightMatrixActive.size();
	assert(__neurons.size() == nKernel); // layers must match
	auto dim = __neurons[0].Dimension(); // image size (pixel)
	for(size_t k=0;k<nKernel;k++)
	{
	    for(size_t i=0;i<dim.first;i++)
	    {
		for(size_t j=0;j<dim.second;j++)
		{
		    __neurons[k][i][j] -> PassWeightPointer(&__weightMatrixActive[k]);
		    __neurons[k][i][j] -> PassBiasPointer(&__biasVectorActive[k]);
		}
	    }
	}
    }
    else 
    {
	std::cout<<"Error: assign w&b pointers, unrecognized layer type."
	    <<std::endl;
	exit(0);
    }
}

void ConstructLayer::DropOut()
{
    if(__type == LayerType::fullyConnected)
	__UpdateActiveFlagFC();
    else if(__type == LayerType::cnn || __type == LayerType::pooling)
	__UpdateActiveFlagCNN();
    else 
    {
	std::cout<<"Error: drop out, un-recongnized layer type."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::EnableDropOut()
{
    //cout<<"debug layer id: "<<GetID()<<", enabling drop out."<<endl;
    __use_drop_out = true;
}

void ConstructLayer::DisableDropOut()
{
    //cout<<"debug layer id: "<<GetID()<<", disabling drop out."<<endl;
    __use_drop_out = false;
}

void ConstructLayer::SetDropOutBranches(int total_number_of_drop_out_branches)
{
    // when using drop out, set how many drop out branches one wants to have (maximum)
    // one cannot have inifinite number of drop out branches, otherwise
    // each drop out branch (w&b) will only get trained once (one batch).
    // The network will never get trained enough.
    __dropOutBranches = total_number_of_drop_out_branches;
}

void ConstructLayer::SetupDropOutFilterPool()
{
    //std::cout<<"INFO: setting up drop-out filter pools."<<std::endl;
    if(__activeFlag.size() <= 0)
    {
        std::cout<<__func__<<" Error: Filters must be initialized before setting up filter pool."
	         <<std::endl;
	exit(0);
    }
    assert(__activeFlag[0].Dimension().first > 0);
    assert(__dropOutBranches >= 1);
    assert(__dropOut > 0.); // one cannot drop all neurons/matrix_elements

    // 1) generate drop-out pool for each filter
    std::vector<std::vector<Filter2D>> _p;
    for(auto &i: __activeFlag)
    {
        std::vector<Filter2D> tmp = i.GenerateCompleteDropOutSet(__dropOutBranches, __dropOut);
	_p.push_back(tmp);
    }

    // 2) fill accordingly to the drop-out pool
    for(int drop_id=0;drop_id<__dropOutBranches;drop_id++)
    {
        std::vector<Filter2D> tmp;
	for(auto &i: _p)
	{
	    tmp.push_back(i[drop_id]);
	}
	__activeFlagPool.push_back(tmp);
    }
}

void ConstructLayer::__UpdateActiveFlagFC()
{
    assert((int)__activeFlagPool.size() == __dropOutBranches);
    __activeFlag = __activeFlagPool[__dropOutBranchIndex];
}

void ConstructLayer::__UpdateActiveFlagCNN()
{
    assert((int)__activeFlagPool.size() == __dropOutBranches);
    __activeFlag = __activeFlagPool[__dropOutBranchIndex];
}


void ConstructLayer::__UpdateActiveFlagFC_Obsolete() // this function has been made obsolete
{
    // for drop out
    // randomly mask out a few neurons
    assert(__activeFlag.size() == 1);
    auto dim = __activeFlag[0].Dimension();
    assert(dim.second == 1);
    // number of neurons to make inactive
    int n_dead = (int)dim.first * __dropOut;

    // generate a filter vector
    std::vector<bool> tmp(dim.first, true);
    for(int i=0;i<n_dead;i++)
	tmp[i] = false;
    TOOLS::Shuffle(tmp);

    // mask out filters
    for(size_t i=0;i<dim.first;i++)
    {
	__activeFlag[0][i][0] = tmp[i];
    }
}

void ConstructLayer::__UpdateActiveFlagCNN_Obsolete() // this function has been made obsolete
{
    // for drop out
    // randomly mask out a few elements of weight matrix
    size_t nKernel = __weightMatrix.size();
    assert(nKernel > 0);
    auto dim = __weightMatrix[0].Dimension();
    assert(dim.first >= 1);
    assert(dim.second >= 1);
    // total matrix elements
    int nTotal = (int)dim.first * (int)dim.second;
    // number of elements to mask out
    int nDead = nTotal * __dropOut;

    // setup a random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row(0, (int)dim.first-1);
    std::uniform_int_distribution<> col(0, (int)dim.second-1);

    auto not_inside = [&](std::vector<std::pair<int, int>> &v, std::pair<int, int>&p) -> bool
    {
	for(auto &i: v)
	    if (i == p) return false;
	return true;
    };

    for(size_t k=0;k<nKernel;k++)
    {
	std::vector<std::pair<int, int>> rand_elements;
	while(rand_elements.size() < (size_t)nDead)
	{
	    int r = row(gen);
	    int c = col(gen);
	    std::pair<int, int> p(r, c);
	    if(not_inside(rand_elements, p)) 
		rand_elements.emplace_back(r, c);
	}

	for(auto &i: rand_elements)
	{
	    __activeFlag[k][i.first][i.second] = false;
	}
    }
}


void ConstructLayer::TransferValueFromActiveToOriginal_WB()
{
    // original WB <= filter => active WB

    size_t nKernels = __weightMatrix.size();
    if(__biasVector.size() != nKernels || __activeFlag.size() != nKernels) 
    {
	std::cout<<"Error: number of kernels not match in filtering"<<std::endl;
	exit(0);
    }

    // for cnn layer
    // cnn layer drop out algorithm only make the filtered element be zero, dimension wont change
    auto active_to_original_cnn = [&](Matrix &original_M, Filter2D &filter_M, Matrix &active_M) 
    {
	auto dim = original_M.Dimension();
	if( dim != filter_M.Dimension())
	{
	    std::cout<<"Error: filter M & original M dimension not match."<<std::endl;
	    exit(0);
	}

	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		// transfer value from active matrix to original matrix
		if(filter_M[i][j]) original_M[i][j] = active_M[i][j];
	    }
	}
    };

    // for fully connected layer
    // fc layer matrix dimension will change, because drop out will delete neurons
    // since one neuron holds one row of the original w&b matrix, 
    // and collum number still equals the number of active neurons of previous layer
    auto active_to_original_fc = [&](Filter2D &filter_M, Filter2D &filter_M_prev_layer) 
    {
	auto dim = __weightMatrix[0].Dimension();
	assert(dim.first = filter_M.Dimension().first);

	if(filter_M.Dimension().second != 1 || filter_M_prev_layer.Dimension().second != 1)
	{
	    std::cout<<"Error: filter matrix fc layer should be a 1-collumn matrix."<<std::endl;
	    exit(0);
	}

	size_t active_i = 0;
	for(size_t i=0;i<dim.first;i++)
	{
	    if(!filter_M[i][0]) continue;

	    // get weights for current active neuron
	    Matrix act_row = __weightMatrixActive[active_i];
	    assert(act_row.Dimension().first == 1);

	    // get bias for current active neuron
	    Matrix act_b = __biasVectorActive[active_i];
	    assert(act_b.Dimension().first == 1);
	    assert(act_b.Dimension().second == 1);

	    // weight
	    size_t active_j = 0;
	    for(size_t j=0;j<dim.second;j++)
	    {
		if(!filter_M_prev_layer[j][0])
		    continue;
		__weightMatrix[0][i][j] = act_row[active_i][active_j];
		active_j++;
	    }
	    // bias
	    __biasVector[0][i][0] = act_b[0][0];

	    active_i++;
	    assert(act_row.Dimension().second == active_j);
	}
	assert(__weightMatrixActive.size() == active_i);
    };

    // start transfer
    for(size_t i=0;i<nKernels;i++)
    {
	if(__type == LayerType::cnn || __type == LayerType::pooling) {
	    active_to_original_cnn(__weightMatrix[i], __activeFlag[i], __weightMatrixActive[i]);
	}
	else if(__type == LayerType::fullyConnected || __type == LayerType::output)
	{
	    // for fc layer, nKernels = 1
	    assert(nKernels == 1);
	    auto filter_prev_layer = __prevLayer->GetActiveFlag();
	    assert(filter_prev_layer.size() == 1);
	    active_to_original_fc(__activeFlag[i], filter_prev_layer[0]);
	}
    }
}


void ConstructLayer::TransferValueFromOriginalToActive_WB()
{
    size_t nKernel = __weightMatrix.size();
    assert(nKernel == __biasVector.size());
    if(__type == LayerType::cnn || __type == LayerType::pooling) 
	assert(nKernel == __activeFlag.size());

    // a lambda funtion for cnn layer mapping original w&b matrix to active w&b matrix
    auto map_matrix_cnn = [&](Matrix &ori_M, Filter2D &filter_M)
    {
	auto dim = ori_M.Dimension();
	assert(dim == filter_M.Dimension());

	// weight matrix
	Matrix tmp(dim);
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++){
		if(filter_M[i][j]){
		    tmp[i][j] = ori_M[i][j];
		}
		else{
		    tmp[i][j] = 0.;
		}
	    }
	}
	__weightMatrixActive.push_back(tmp);
    };

    // a lambda funtion for fc layer mapping original w&b matrix to active w&b matrix
    //   to comply with Neuron design, each active row of original w&b matrix will be 
    //   taken out and then form a matrix, then filled to active w&b matrix
    //   so active w&b matrix will be composed of dimension (1, N) matrix, and then each of them will be passed to neurons
    auto map_matrix_fc = [&](Matrix &ori_M, Filter2D &filter_M)
    {
	auto dim = ori_M.Dimension();

	assert(dim.first == filter_M.Dimension().first);
	assert(filter_M.Dimension().second == 1);
	assert(__biasVector.size() == 1);

	for(size_t i=0;i<dim.first;i++)
	{
	    if(!filter_M[i][0]) continue;

	    // weight
	    std::vector<double> act_row;
	    for(size_t j=0;j<dim.second;j++)
	    {
		if(__prevLayer != nullptr &&   // current layer is not input layer
			(__prevLayer->GetType() == LayerType::fullyConnected ) // only fc layer no need for other type layers (output layer is labeled as 'fc'; input layer has no w&b matrix)
		  )
		{
		    auto filter_prev = __prevLayer->GetActiveFlag();
		    assert(filter_prev.size() == 1);
		    if(!filter_prev[0][j][0])
			continue;
		}
		act_row.push_back(ori_M[i][j]);
	    }
	    std::vector<std::vector<double>> act_M;
	    act_M.push_back(act_row);
	    Matrix tmp(act_M);
	    __weightMatrixActive.push_back(tmp);

	    // bias
	    std::vector<double> act_bias;
	    act_bias.push_back(__biasVector[0][i][0]);
	    std::vector<std::vector<double>> act_b;
	    act_b.push_back(act_bias);
	    Matrix tmp_b(act_b);
	    __biasVectorActive.push_back(tmp_b);
	}
    };

    // start mapping
    if(__type == LayerType::cnn || __type == LayerType::pooling)
    {
	for(size_t k=0; k<nKernel;k++)
	{
	    map_matrix_cnn(__weightMatrix[k], __activeFlag[k]);
	}
	// for cnn, drop out won't change threshold
	__biasVectorActive = __biasVector; 
    }
    else if (__type == LayerType::fullyConnected || __type == LayerType::output)
    {
	map_matrix_fc(__weightMatrix[0], __activeFlag[0]);
    }

    // update active neuron coord after drop out
    if(__type == LayerType::cnn || __type == LayerType::pooling)
    {
	__activeNeuronDim = __neuronDim; // drop out not happening on neurons, so dimension stays same
    }
    else if(__type == LayerType::fullyConnected || __type == LayerType::output)
    {
	__activeNeuronDim = __neuronDim; // update active neuron dimension
	size_t active_neurons = __weightMatrixActive.size();
	__activeNeuronDim.i = active_neurons;
    }

    //std::cout<<"Debug: Layer:"<<GetID()<<" TransferValueFromOriginalToActiveWB() done."<<std::endl;
} 

void ConstructLayer::UpdateImageForCurrentTrainingSample()
{
    // loop for all neurons
    // not used
    // instead implemented in Network class
}

void ConstructLayer::ClearImage()
{
    __imageA.clear();
    __imageZ.clear();
    __imageSigmaPrime.clear();
    __imageDelta.clear();

    __imageAFull.clear();
    __imageZFull.clear();
    __imageSigmaPrimeFull.clear();
    __imageDeltaFull.clear();

    __wGradient.clear();
    __bGradient.clear();

}

NeuronCoord ConstructLayer::GetActiveNeuronDimension()
{
    return __activeNeuronDim;
}

void ConstructLayer::SetDropOutFactor(double f)
{
    __dropOut = f;
}

void ConstructLayer::SetCostFuncType(CostFuncType t)
{
    __cost_func_type = t;
}

std::vector<Matrix>* ConstructLayer::GetWeightMatrix()
{
    return &__weightMatrixActive;
}

std::vector<Matrix>* ConstructLayer::GetBiasVector()
{
    return &__biasVectorActive;
}

std::vector<Matrix>* ConstructLayer::GetWeightMatrixOriginal()
{
    return &__weightMatrix;
}

std::vector<Matrix>* ConstructLayer::GetBiasVectorOriginal()
{
    return &__biasVector;
}

std::vector<Images> & ConstructLayer::GetWeightGradients()
{
    return __wGradient;
}

std::vector<Images> & ConstructLayer::GetBiasGradients()
{
    return __bGradient;
}


LayerType ConstructLayer::GetType()
{
    return __type;
}

LayerDimension ConstructLayer::GetLayerDimension()
{
    return __layerDimension;
}

double ConstructLayer::GetDropOutFactor()
{
    return __dropOut;
}

std::vector<Filter2D>& ConstructLayer::GetActiveFlag()
{
    return __activeFlag;
}

int ConstructLayer::GetBatchSize()
{
    if( __p_data_interface == nullptr)
    {
        std::cout<<"Error: ConstructLayer::GetBatchSize() data interface is nullptr."
	         <<endl;
        exit(0);
    }

    int batch_size = __p_data_interface -> GetBatchSize();
    if(batch_size <= 0)
    {
        std::cout<<"Error: ConstructLayer::GetBatchSize() batch size is 0, seems data interface is not implemented."
	         <<endl;
        exit(0);
    }
    return batch_size;
}

CostFuncType ConstructLayer::GetCostFuncType()
{
    if(__type != LayerType::output)
    {
        cout<<"Error: ConstructLayer::GetCostFuncType() only works for output layer"
	    <<endl;
	exit(0);
    }
    return __cost_func_type;
}

ActuationFuncType ConstructLayer::GetNeuronActuationFuncType()
{
    return __neuron_actuation_func_type;
}

DataInterface * ConstructLayer::GetDataInterface()
{
    if(__p_data_interface == nullptr)
    {
        cout<<"Error: ConstructLayer::GetDataInterface() data interface is nullptr."
	    <<endl;
	exit(0);
    }
    return __p_data_interface;
}

Layer* ConstructLayer::GetNextLayer()
{
    if(__nextLayer == nullptr)
    {
        std::cout<<"Error: ConstructLayer::GetNextLayer(): __nextLayer is not setup"
	         <<std::endl;
	exit(0);
    }
    return __nextLayer;
}


Layer* ConstructLayer::GetPrevLayer()
{
    if(__prevLayer == nullptr)
    {
        std::cout<<"Error: ConstructLayer::GetNextLayer(): __prevLayer is not setup"
	         <<std::endl;
	exit(0);
    }

    return __prevLayer;
}


void ConstructLayer::UpdateWeightsAndBias()
{
    // after finishing one training batch, update weights and bias

    // 1) first update w&b graidents
    UpdateWeightsAndBiasGradients();

    // 2) then update weights and bias
    auto layerType = this->GetType();

    if(layerType == LayerType::fullyConnected || layerType == LayerType::output)  // output is also a fc layer
    {
	UpdateWeightsAndBiasFC();
    }
    else if(layerType == LayerType::cnn) 
    {
	UpdateWeightsAndBiasCNN();
    }
    else if(layerType == LayerType::pooling) 
    {
	UpdateWeightsAndBiasPooling();
    }
    else 
    {
	std::cout<<__func__<<" Error: update weights and bias, unsupported layer type."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::UpdateWeightsAndBiasGradients()
{
    // after finishing one training batch, update weights and bias gradients
    auto layerType = this->GetType();

    if(layerType == LayerType::fullyConnected || layerType == LayerType::output)  // output is also a fully connected layer
    {
	UpdateWeightsAndBiasGradientsFC();
    }
    else if(layerType == LayerType::cnn) 
    {
	UpdateWeightsAndBiasGradientsCNN();
    }
    else if(layerType == LayerType::pooling) 
    {
	UpdateWeightsAndBiasGradientsPooling();
    }
    else 
    {
	std::cout<<__func__<<" Error: update weights and bias gradients, unsupported layer type."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::UpdateWeightsAndBiasGradientsFC()
{
    //cout<<__func__<<" started."<<endl;
    // after finishing one batch, update weights and bias, for FC layer
    // get output from previous layer for current training sample
    vector<Images> a_images;
    if(/*__prevLayer->GetType() != LayerType::input && */__prevLayer->GetType() != LayerType::fullyConnected) // input layer has no W&B
    {
        // if previous layer is not an input/fullyConnected layer, then Vectorization operation is needed
	//      this is for cnn->fc or pooling->fc
        auto _tmp = __prevLayer->GetImagesFullA(); // 'a' images from previous layer
	for(auto &i: _tmp)
	    a_images.push_back(i.Vectorization());
    }
    else
    {
        // if previous layer is input/fc, no other operation is needed
	a_images = __prevLayer->GetImagesFullA(); // 'a' images from previous layer
    }
    auto d_images = this->GetImagesFullDelta(); // delta images from current layer
    // NOTE: 'a' and 'delta' include value correpsonds to disabled neurons
    //       it's just these values have been set to 0 when updating __imagesA, __imagesDelta
    //       see Functions UpdateImagesA() and UpdateImagesDelta()
    //       since dC/dw_ij = a^{l-1}_j * delta^l_k, so if a = 0 or delta = 0, then dC/dw = 0, namely no change on w for inactive neurons
    if(a_images.size() != d_images.size()) {
	std::cout<<__func__<<" Error: batch size not equal..."<<std::endl;
	exit(0);
    }
    if(a_images.back().OutputImageFromKernel.size() != 1 ) {
	std::cout<<__func__<<" Error: updating weights and bias gradients for FC, dimension not match."<<std::endl;
	std::cout<<"                  kernel size: "<<a_images.back().OutputImageFromKernel.size()<<std::endl;
	exit(0);
    }

    // so directly work on __weightMatrix, not __weightMatrixActive
    // check layer type
    if(__weightMatrix.size() != 1) {
	std::cout<<__func__<<" Error: more than 1 weight matrix exist in fully connected layer."<<std::endl;
	exit(0);
    }

    //cout<<"sample size: "<<d_images.size()<<endl;
    //for(auto &i: d_images)
    //{
    //    cout<<i.OutputImageFromKernel[0].Dimension()<<endl;
    //    cout<<i.OutputImageFromKernel[0]<<endl;
    //}
    // loop for samples
    for(size_t i=0;i<a_images.size();i++)
    {
	Matrix a_matrix = a_images[i].OutputImageFromKernel[0]; // 'a' image from previous layer
	Matrix d_matrix = d_images[i].OutputImageFromKernel[0]; // 'd' image from current layer
	//cout<<"Layer id: "<<GetID()<<endl;
	//cout<<"a matrix: "<<endl<<a_matrix<<endl;
	//cout<<"delta matrix: "<<endl<<d_matrix<<endl;

	auto d1 = (__weightMatrix[0]).Dimension(), d2 = a_matrix.Dimension(), d3 = d_matrix.Dimension();
	if(d1.first != d3.first || d1.second != d2.first)
	{
	    std::cout<<__func__<<"Error: updating weights and bias gradients for FC, dimension not match."<<std::endl;
	    std::cout<<"          current layer 'w' matrix dimension: "<<d1<<std::endl;
	    std::cout<<"          previous layer 'a' matrix dimension: "<<d2<<std::endl;
	    std::cout<<"          current layer 'delta' matrix dimension: "<<d3<<std::endl;
	    exit(0);
	}

	//cout<<"sample_id: "<<i<<endl;
	Matrix a_T = a_matrix.Transpose();
	Matrix dw = d_matrix * a_T;

        //cout<<a_T.Dimension()<<endl;
	//cout<<"a :"<<endl<<a_T<<endl;
	//cout<<d_matrix.Dimension()<<endl;
	//cout<<"d :"<<endl<<d_matrix<<endl;
	//cout<<"dw: "<<endl<<dw<<endl;
	//cout<<"dw matrix."<<endl;
	//getchar();

	// push bias gradient for current training sample
	Images tmp_w_gradients;
	tmp_w_gradients.OutputImageFromKernel.push_back(dw);
	__wGradient.push_back(tmp_w_gradients); // push weight gradient for current training sample

	Images tmp_b_gradients;
	tmp_b_gradients.OutputImageFromKernel.push_back(d_matrix);
	__bGradient.push_back(tmp_b_gradients); // bias gradient equals delta
    }

    //cout<<"INFO: "<<__func__<<" finished."<<endl;
} 

/* ***************************************************************
// not used. Using Matrix hadamard multiply operation instead
static void maskMatrix(Matrix& M, Matrix& F)
{
    // set all inactive elements to 0
    auto dimM = M.Dimension();
    auto dimF = F.Dimension();

    assert(dimM == dimF);

    for(size_t i=0;i<dimF.first;i++)
    {
        for(size_t j=0;j<dimF.second;j++)
	{
	    if(F[i][j] == 0) M[i][j] = 0;
	}
    }
}
*/

void ConstructLayer::UpdateWeightsAndBiasFC()
{
    //cout<<__func__<<" started..."<<endl;
    // after finishing one batch, update weights and bias, for FC layer
    size_t M = __imageDeltaFull.size(); // batch size
    if( M != __wGradient.size() ) {
	std::cout<<__func__<<" Error: update FC weights, batch size not match."<<std::endl;
	std::cout<<"           weight gradient batch size: "<<__wGradient.size()<<endl;
	std::cout<<"           delta image batch size: "<<__imageDeltaFull.size()<<endl;
	exit(0);
    }
    if( __wGradient[0].OutputImageFromKernel.size() != 1 || __bGradient[0].OutputImageFromKernel.size()!=1) 
    {
	std::cout<<__func__<<" Error: update FC weiths, more than 1 w gradient matrix found."
	    <<std::endl;
	exit(0);
    }

    // gradient descent
    if(__weightMatrix.size() != __neuronDim.k) {
	std::cout<<__func__<<" Error: update FC layer weights, more than 1 weight matrix found."<<std::endl;
	std::cout<<"__weightMatrix size: "<<__weightMatrix.size()<<std::endl;
	std::cout<<"neuron dimension: "<<__neuronDim<<std::endl;
	exit(0);
    }
    Matrix dw((__weightMatrix[0]).Dimension(), 0); 
    for(size_t i=0;i<M;i++){ // sum x (batches)
	dw  = dw + __wGradient[i].OutputImageFromKernel[0];
	//cout<<__wGradient[i].OutputImageFromKernel[0]<<endl;
	//cout<<i<<" before sum"<<endl;
	//getchar();
    }

    if(__weights_optimizer == WeightsOptimizer::SGD)
    {
        // standard SGD 
	dw = dw * double(__learningRate/(double)M); // over batch 
    }
    else if(__weights_optimizer == WeightsOptimizer::Adam)
    {
        // Adam optimizer
	dw = dw * (1./ (double)M );

	//Matrix tmp = dw * __learningRate;
	//cout<<"Before adam: "<<endl<<dw<<endl;
        dw = AdamOptimizer(dw, 0);
	//cout<<"after adam: "<<endl<<dw<<endl;
	//getchar();
    }
    else
    {
        std::cout<<__func__<<" Error: unsupported optimization method."<<std::endl;
	exit(0);
    }
    //cout<<"learning rate: "<<double(__learningRate/(double)M)<<endl;

    // Get filter Matrix for masking Regularization item
    assert(__activeFlag.size() == 1);
    auto convertFilterToMatrix = [&](Filter2D & F) -> Matrix
    {
        auto dim = F.Dimension();
	Matrix M(dim, 0);
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
	        if(F[i][j]) M[i][j] = 1;
	    }
	}
	return M;
    };
    Matrix currentLayerFilter = convertFilterToMatrix(__activeFlag[0]);
    //cout<<currentLayerFilter.Dimension()<<endl;
    Matrix prevLayerFilter(1, dw.Dimension().second, 1);
    //cout<<"before: "<<endl<<prevLayerFilter<<endl;

    if(__prevLayer->GetType() == LayerType::input) // only fc layer need this, other layers won't affect
    {
	auto prevLayerFilters = __prevLayer->GetActiveFlag();
	assert(prevLayerFilters.size() == 1);
	assert(prevLayerFilters[0].Dimension().first == (size_t)__prevLayer->GetNumberOfNeurons());

	Matrix filter = convertFilterToMatrix(prevLayerFilters[0]);
	prevLayerFilter = filter.Transpose();
	//cout<<"after: "<<endl<<prevLayerFilter<<endl;
    }

    // following is two flag matrix (dim[n,1] x dim[1, m], with bool elements) product
    // all the elements of the product will be either 0 or 1
    // this product can be used to mask out disabled elements in weight matrix regularization
    Matrix F = currentLayerFilter * prevLayerFilter;
    //cout<<F<<endl;
    //cout<<prevLayerFilter<<endl;
    //cout<<dw<<endl;
    //getchar();

    assert(F.Dimension() == (__weightMatrix[0]).Dimension());
    // Regularization
    //double f_regularization = 0.;
    if(__regularizationMethod == Regularization::L2) 
    {
        // obsolete
	//f_regularization = 1 - __learningRate * __regularizationParameter / M;
	//(__weightMatrix[0]) = (__weightMatrix[0]) * f_regularization - dw;
	
	// new
	// hadamard multiply with F to maks out all inactive elements
	Matrix regularization_M = (__weightMatrix[0]^F) *(__learningRate * __regularizationParameter/((double)M)); 
	Matrix total_correction_M = regularization_M + dw; // dw already include learning rate during generation
	__weightMatrix[0] = __weightMatrix[0] - total_correction_M;
    } 
    else if(__regularizationMethod == Regularization::L1) 
    {
	Matrix _t = (__weightMatrix[0]); // make a copy of weight matrix
	_t(&SGN); // get the sign for each element in weight matrix
	_t = _t^F; // hadamard operation to mask out all inactive elements
	_t = _t * (__learningRate*__regularizationParameter/(double)M); // L1 regularization part
	(__weightMatrix[0]) = (__weightMatrix[0]) -  _t; // apply L1 regularization to weight matrix
	(__weightMatrix[0]) = (__weightMatrix[0]) - dw; // apply gradient decsent part
    } 
    else 
    {
	std::cout<<__func__<<" Error: update FC weights, unsupported regularization."<<std::endl;
	exit(0);
    }

    //cout<<"reached hrere..."<<endl;

    // bias
    Matrix db(__biasVector[0].Dimension());
    for(size_t i=0;i<M;i++){
	db = db + __bGradient[i].OutputImageFromKernel[0];
    }
    db = db / (double)M;
    db = db * __learningRate;
    __biasVector[0] = __biasVector[0] - db;

    //cout<<__func__<<" ended.."<<endl;
}

void ConstructLayer::UpdateWeightsAndBiasGradientsCNN()
{
    // after finishing one batch, update weights and bias gradient, (CNN layer)
    // cnn layer is different with FC layer. For FC layer, different 
    // neurons have different weights, however, for CNN layer, different
    // neurons in one image share the same weights and bias (kernel).
    // So one need to loop for kernels

    // get 'a' matrix from previous layer for current training sample
    //         !!! cnn->fc connection is allowed, however, 
    //         !!1 fc->cnn connection is not implemented for the moment
    //         !!! so no vectorization or tensorization is needed here
    auto aVec = __prevLayer -> GetImagesFullA();
    // get 'delta' matrix for current layer
    auto deltaVec = this -> GetImagesFullDelta();

    // get kernel number
    size_t nKernel = __weightMatrixActive.size();

    // loop for batch
    for(size_t nbatch = 0;nbatch<aVec.size();nbatch++)
    {
	Images &a_image =  aVec[nbatch];
	std::vector<Matrix> &a_matrix = a_image.OutputImageFromKernel;
	Images &d_image = deltaVec[nbatch];
	std::vector<Matrix> &d_matrix = d_image.OutputImageFromKernel;
	if(d_matrix.size() != nKernel) {
	    std::cout<<"Error: updateing cnn w gradients, number of kernels not match."<<std::endl;
	    exit(0);
	}

	// tmp image for saving weight and bias gradients
	Images tmp_w_gradients;
	Images tmp_b_gradients;

	// loop for kernel
	for(size_t k = 0;k<nKernel;k++)
	{
	    // get 'delta' matrix for current kernel
	    Matrix &delta = d_matrix[k];

	    // update current kernel
	    auto dimKernel = __weightMatrixActive[k].Dimension();

	    // weight gradient
	    Matrix dw(dimKernel);

	    for(size_t i=0;i<dimKernel.first;i++)
	    {
		for(size_t j=0;j<dimKernel.second;j++)
		{
		    // It is a convolution value with 'a_previous_layer' as the input image and 'delta_this_layer' as the "kernel"
		    // step 1) (i, j) is the w(i, j) of current layer weight matrix -> need to find the corresponding coord in 'a_previous_layer'
		    //size_t i_a_prev = i*__cnnStride; // now __cnnStride is set to 1, and is not adjustable
		    //size_t j_a_prev = j*__cnnStride; // need to derive a detailed expression for __cnnStride > 1
		    // step 2) gradient descent part
		    double _tmp = 0;
		    for(auto &_ap: a_matrix){
			_tmp += Matrix::GetCorrelationValue(_ap, delta, i, j);
		    }
		    dw[i][j] = _tmp;
		}
	    }
	    tmp_w_gradients.OutputImageFromKernel.push_back(dw); // push weight gradient for current training sample

	    // update bias gradient
	    Matrix db(1, 1);
	    auto dim = delta.Dimension();
	    double b_gradient = delta.SumInSection(0, dim.first, 0, dim.second);
	    db[0][0] = b_gradient;
	    tmp_b_gradients.OutputImageFromKernel.push_back(db);
	}
	__wGradient.push_back(tmp_w_gradients);
	__bGradient.push_back(tmp_b_gradients);
    }
}

void ConstructLayer::UpdateWeightsAndBiasCNN()
{
    // after finishing one training batch, update weights and bias gradient, (CNN layer)
    // cnn layer is different with FC layer. For FC layer, different 
    // neurons have different weights, however, for CNN layer, different
    // neurons in one image share the same weights and bias.
    // so when you are
    // using this function in layer class, you need to loop over every 
    // neuron for FC layer, but you should not loop over neruons in 
    // the same image for CNN layer, you should just use only one neuron,
    // any neuron would be fine, instead you need to loop over images. --- This is old design

    // after finishing one batch, update weights and bias, CNN layer
    size_t M = __imageDeltaFull.size(); // batch size
    if( M != __wGradient.size() ) {
	std::cout<<"Error: update FC weights, batch size not match."<<std::endl;
	exit(0);
    }

    // in case use drop out, need to filter out all inactive elements
    // Get filter Matrix from Filter2D structure
    assert(__activeFlag.size() == __weightMatrix.size());
    auto convertFilterToMatrix = [&](Filter2D & F) -> Matrix
    {
	auto dim = F.Dimension();
	Matrix _M(dim, 0);
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		if(F[i][j]) _M[i][j] = 1;
	    }
	}
	return _M;
    };

    // loop for kernel
    size_t nKernel = __weightMatrix.size();
    for(size_t k=0;k<nKernel;k++)
    {
        // 1) get gradient
	// gradient descent
	Matrix dw(__weightMatrix[k].Dimension());
	// loop for batch
	for(size_t i=0;i<M;i++){ 
	    dw  = dw + __wGradient[i].OutputImageFromKernel[k];
	}
	if(__weights_optimizer == WeightsOptimizer::SGD)
	{
	    // stochastic gradient descent
	    dw = dw * double(__learningRate/(double)M); // gradients average over batch size
	}
	else if(__weights_optimizer == WeightsOptimizer::Adam)
	{
	    // adam optimize
	    dw = dw * (1./(double)M);
	    dw = AdamOptimizer(dw, k);
	}
	else
	{
	    std::cout<<__func__<<" Error: unsupported weights optimizer."<<std::endl;
	    exit(0);
	}

        // Hadamard F to mask out all inactive elements
	Matrix F = convertFilterToMatrix(__activeFlag[k]);
	assert(F.Dimension() == (__weightMatrix[k]).Dimension());
	dw = dw^F; // for safe reason, mask out all inactive elements

	// regularization part
	//double f_regularization = 0;
	if(__regularizationMethod == Regularization::L2)
	{
	    // obsolete
	    //f_regularization = 1 - __learningRate * __regularizationParameter / (double)M;
	    //(__weightMatrix[k]) = (__weightMatrix[k])*f_regularization;
	    //(__weightMatrix[k]) = (__weightMatrix[k]) - dw;

	    Matrix regularization_M = (__weightMatrix[k]^F) * (__learningRate * __regularizationParameter/((double)M));
	    Matrix total_correction_M = regularization_M + dw; // dw already have learing rate multiplied
	    __weightMatrix[k] = __weightMatrix[k] - total_correction_M;
	}
	else if(__regularizationMethod == Regularization::L1)
	{
	    Matrix tmp = (__weightMatrixActive[k]);
	    tmp(&SGN);
	    tmp = tmp^F; // hadamard operation to mask out all inactive elements
	    tmp = tmp * (__learningRate*__regularizationParameter/(double)M);
	    //(*__w) = (*__w) - tmp - dw;
	    (__weightMatrix[k]) = (__weightMatrix[k]) - dw;
	    (__weightMatrix[k]) = (__weightMatrix[k]) - tmp;
	}
	else {
	    std::cout<<"Error: update CNN weights, unsupported regularizaton method."<<std::endl;
	    exit(0);
	}

	// update bias
	Matrix db(1, 1);
	// loop for batch
	for(size_t i=0;i<M;i++){
	    db = db + __bGradient[i].OutputImageFromKernel[k];
	}
	db = db * double(__learningRate / (double)M);
	__biasVector[k] = __biasVector[k] - db;
    }
}

void ConstructLayer::UpdateWeightsAndBiasGradientsPooling()
{
    // after finishing one training sample, update weights and bias gradient, (pooling layer)
    // pooling layer weight elements always equal to 1, bias always equal to 0
    // no need to update
    return; 
}

void ConstructLayer::UpdateWeightsAndBiasPooling()
{
    // after finishing one batch, update weights and bias , (pooling layer)
    // pooling layer weight elements always equal to 1, bias always equal to 0
    // no need to update
    return; 
}

static double __AdamSquareRoot(double v)
{
    return sqrt((double)v);
}

Matrix ConstructLayer::AdamOptimizer(const Matrix &dw, int kernel_index)
{
    //cout<<"----------------------------------------------------------"<<endl;
    // Adam optimizer, refer to: https://arxiv.org/abs/1412.6980
    // sanity check
    assert(Momentum_1st_order.size() == Momentum_2nd_order.size());
    if(Momentum_1st_order.size() == 0)
    {
        // initiaize with 0's
	if(__type == LayerType::fullyConnected || __type == LayerType::output || __type == LayerType::input)
	{
	    assert(kernel_index == 0);
	    Momentum_1st_order.resize(1, Matrix(dw.Dimension(), 0));
	    Momentum_2nd_order.resize(1, Matrix(dw.Dimension(), 0));
	}
	else if(__type == LayerType::cnn || __type == LayerType::pooling)
	{
	    assert(__n_kernels_cnn > 0);
	    Momentum_1st_order.resize(__n_kernels_cnn, Matrix(dw.Dimension(), 0));
	    Momentum_2nd_order.resize(__n_kernels_cnn, Matrix(dw.Dimension(), 0));
	}
    }

    // g_t
    Matrix g_t = dw;
    //cout<<"g_t: "<<endl<<g_t<<endl;

    // these parameters are from paper: https://arxiv.org/abs/1412.6980
    //     according to the ref: learning rate is best to be 0.001
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    // get m_t
    Matrix m_t_1 = Momentum_1st_order[kernel_index];
    //cout<<"m_t_1: "<<endl<<m_t_1<<endl;
    //cout<<" 1 - beta1: "<<1.-beta1<<endl;
    Matrix m_tmp = g_t * (1. - beta1);
    //cout<<"m_tmp: "<<endl<<m_tmp<<endl;
    Matrix m_t = (m_t_1 * beta1) + m_tmp;
    //Matrix debug_tmp = m_t_1 * beta1;
    //cout<<"m_t_1*beta1: "<<endl<<(debug_tmp)<<endl;
    //cout<<"m_t: = (m_t_1 * beta1) + g_t * (1 - beta1): "<<endl<<m_t<<endl;
    // save m_t
    Momentum_1st_order[kernel_index] = m_t; // update 1st order momentum
    // get v_t
    Matrix v_t_1 = Momentum_2nd_order[kernel_index];
    //cout<<"v_t_1: "<<endl<<v_t_1<<endl;
    m_tmp = (g_t^g_t) * (1. - beta2);
    //cout<<"g_t^2 * (1. - beta2(0.999))"<<endl<<m_tmp<<endl;
    Matrix v_t = (v_t_1 *beta2) + m_tmp;
    //debug_tmp = v_t_1 *beta2;
    //cout<<"v_t_1 * beta2:"<<endl<<debug_tmp<<endl;
    //cout<<"vt = v_t_1*beta2 + g_t^2*(1-beta2): "<<endl<<v_t<<endl;
    // save v_t
    Momentum_2nd_order[kernel_index] = v_t; // update 2nd order momentum

    // get bias corrected m_t and v-t
    __beta1_to_power_t *= beta1;
    __beta2_to_power_t *= beta2;
    Matrix bias_corrected_1st_order_moment = m_t / (1. - __beta1_to_power_t);
    //cout<<"bias corrected m_t: /"<<(1. - __beta1_to_power_t)<<endl<<bias_corrected_1st_order_moment<<endl;
    Matrix bias_corrected_2nd_order_moment = v_t / (1. - __beta2_to_power_t);
    //cout<<"bias corrected v_t: /"<<(1. - __beta2_to_power_t)<<endl<<bias_corrected_2nd_order_moment<<endl;

    // get d_theta_t 
    Matrix m_epsilon(g_t.Dimension(), epsilon);
    bias_corrected_2nd_order_moment(__AdamSquareRoot);
    //cout<<"sqrt(^v_t): "<<endl<<bias_corrected_2nd_order_moment<<endl;
    Matrix square_root_of_vt = bias_corrected_2nd_order_moment;
    Matrix denominator = square_root_of_vt + m_epsilon;
    //cout<<"m_epsilon: "<<endl<<m_epsilon<<endl;
    //cout<<"sqrt(^v_t): + epsilon"<<endl<<denominator<<endl;
    Matrix ratio = bias_corrected_1st_order_moment / denominator;
    //cout<<"^m_t/(sqrt(^v_t) + epsilon):"<<endl<<ratio<<endl;

    // according to the reference, alpha best to be 0.001, however, 
    // based on my test, it is too slow, alpha = 0.06 works best
    double alpha = __learningRate;
    Matrix dw_Adam = ratio * alpha;
    //Matrix dw_Adam = ratio * __learningRate;
    //cout<<"learning rate: "<<__learningRate<<endl;
    //cout<<"dw_Adam: "<<endl<<dw_Adam<<endl;
    //getchar();

    return dw_Adam;
}

void ConstructLayer::SetLearningRate(double l)
{
    // set up learning rate
    __learningRate = l;
}

void ConstructLayer::SetRegularizationMethod(Regularization r)
{
    // set L1 or L2 regularization
    __regularizationMethod = r;
}

void ConstructLayer::SetRegularizationParameter(double p)
{
    // set hyper parameter lambda
    __regularizationParameter = p;
}

void ConstructLayer::SetPoolingMethod(PoolingMethod m)
{
    __poolingMethod = m;
}

void ConstructLayer::SetCNNStride(int s)
{
    __cnnStride = s;
}

PoolingMethod & ConstructLayer::GetPoolingMethod()
{
    return __poolingMethod;
}

int ConstructLayer::GetCNNStride()
{
    return __cnnStride;
}

std::pair<size_t, size_t> ConstructLayer::GetOutputImageSize()
{
    if(__type == LayerType::fullyConnected)
	return GetOutputImageSizeFC();
    else if(__type == LayerType::cnn)
	return GetOutputImageSizeCNN();
    else if(__type == LayerType::input)
        return GetOutputImageSizeInputLayer();
    else
	return GetOutputImageSizeCNN();
}

std::pair<size_t, size_t> ConstructLayer::GetOutputImageSizeCNN()
{
    // used for setup cnn layer
    return __outputImageSizeCNN;
}

std::pair<size_t, size_t> ConstructLayer::GetOutputImageSizeFC()
{
    // used for setup cnn layer
    return std::pair<size_t, size_t>(__n_neurons_fc, 1);
}

std::pair<size_t, size_t> ConstructLayer::GetOutputImageSizeInputLayer()
{
    // used for setup first cnn layer
    // directly get dimension from DataInterface 
    auto dim = __p_data_interface->GetDataDimension();

    // directly return image dimension
    return dim;
}


int ConstructLayer::GetNumberOfNeurons()
{
    if(__type == LayerType::fullyConnected || __type == LayerType::output ) // output layer is also a fully connected layer
    {
	// used for setup fc layer
	return __n_neurons_fc;
    }
    else if(__type == LayerType::input)
    {
	auto dim = __neurons[0].Dimension();
	return static_cast<int>((dim.first * dim.second));
    }
    else if(__type == LayerType::cnn || __type == LayerType::pooling)
    {
	auto dim = GetOutputImageSize();
	return static_cast<int>((dim.first * dim.second * __n_kernels_cnn));
    }
    else
    {
	std::cout<<"Warning: GetNumberOfNeurons only work for fc and input layer."<<std::endl;
	return 0;
    }
}

int ConstructLayer::GetNumberOfNeuronsFC()
{
    // used for setup fc layer
    return __n_neurons_fc;
}

size_t ConstructLayer::GetNumberOfKernelsCNN()
{
    // note: both cnn layer and pooling layer use this function
    return __n_kernels_cnn;
}

std::pair<size_t, size_t> ConstructLayer::GetKernelDimensionCNN()
{
    // note: both cnn layer and pooling layer use this function; to save memory
    return __kernelDim;
}


void ConstructLayer::SaveAccuracyAndCostForBatch()
{
    // accuracy
    // get labels for current batch
    std::vector<Matrix> & batch_labels =  __p_data_interface -> GetCurrentBatchLabel();
    assert(__imageA.size() == batch_labels.size());

    double correct_prediction = 0.;
    for(size_t i=0;i<batch_labels.size();i++)
    {
        Matrix & label_sample = batch_labels[i];
	Matrix & output_sample = __imageA[i].OutputImageFromKernel[0]; // output layer is 1D

	auto dim = output_sample.Dimension();
	assert(dim == label_sample.Dimension());
	assert(dim.second == 1);

        std::pair<size_t, size_t> max_coord;
	double max = output_sample.MaxInSection(0, dim.first, 0, dim.second, max_coord);

        double confidence_threshold = 0.8;
	if (max > confidence_threshold)
	{
	    Matrix tmp(dim, 0);
	    tmp[max_coord.first][max_coord.second] = 1.;

	    if(tmp == label_sample)
	        correct_prediction += 1.0;
	}

    }

    double accuracy_for_this_batch = correct_prediction / (int)batch_labels.size();
    __accuracyForBatches.push_back(accuracy_for_this_batch);

    // cost
    double cost = 0.;
    for(auto &i: __outputLayerCost)
        cost += i;
    cost /= (double)__outputLayerCost.size();

    __lossForBatches.push_back(-cost);

    std::cout<<"............ accuracy for batch training: "<<accuracy_for_this_batch<<endl;
    std::cout<<"............    losss for batch training: "<<-cost<<endl;
}

std::vector<double> & ConstructLayer::GetAccuracyForBatches()
{
    return __accuracyForBatches;
}


std::vector<double> & ConstructLayer::GetCostForBatches()
{
    return __lossForBatches;
}

void ConstructLayer::SaveTrainedWeightsAndBias()
{
    int layer_id = GetID();
    LayerType layer_type = GetType();
    LayerDimension layer_dimension = GetLayerDimension();

    // parse file name
    ostringstream oss;
    oss<<"weights_and_bias_trained/LayerID=";
    oss<<layer_id;
    oss<<layer_type;
    oss<<layer_dimension;
    oss<<"_weights_and_bias.txt";

    // save weights and bias
    std::fstream ff(oss.str(), std::ios::out);
    std::cout<<__func__<<"(): Saving trained weights and bias from layer: "<<layer_id<<" to files."<<std::endl;
    if(!ff.is_open())
    {
        std::cout<<__func__<<" Error: cannot open file: "<<oss.str()<<" to save weights and bias."
	         <<std::endl;
	std::cout<<"           please make sure you have 'weights_and_bias_trained' folder exsit."
	         <<std::endl;
	exit(0);
    }
    for(size_t i=0;i<__weightMatrix.size();i++)
    {
        ff<<"weight "<<i<<":"<<std::endl;
        //cout<<"weight "<<i<<":"<<std::endl;
	ff<<__weightMatrix[i]<<std::endl;
	//cout<<__weightMatrix[i]<<endl;
	ff<<"bias "<<i<<":"<<std::endl;
	//cout<<"bias: "<<i<<":"<<std::endl;
	ff<<__biasVector[i]<<std::endl;
	//cout<<__biasVector[i]<<endl;
    }
    ff.close();
}

void ConstructLayer::LoadTrainedWeightsAndBias()
{
    int layer_id = GetID();
    LayerType layer_type = GetType();
    LayerDimension layer_dimension = GetLayerDimension();

    // parse file name
    ostringstream oss;
    oss<<"weights_and_bias_trained/LayerID=";
    oss<<layer_id;
    oss<<layer_type;
    oss<<layer_dimension;
    oss<<"_weights_and_bias.txt";

    // save weights and bias
    std::fstream ff(oss.str(), std::ios::in);
    std::cout<<__func__<<"(): Loading trained weights and bias for layer: "<<layer_id<<" from files."<<std::endl;

    if(!ff.is_open())
    {
	std::cout<<__func__<<" Error: cannot open file: "<<oss.str()<<" to load weights and bias."
	    <<std::endl;
	std::cout<<"           please make sure you have files exsit."
	    <<std::endl;
	exit(0);
    }

    // a helper
    auto parseLine = [&](std::string line) -> std::vector<double>
    {
        std::vector<double> tmp;
        istringstream iss(line);
	double v;
	while(iss>>v)
	{
	    tmp.push_back(v);
	}
	return tmp;
    };

    std::string line;
    std::vector<std::vector<double>> vv;
    while(getline(ff, line))
    {
        std::vector<double> v_line;
        if(line.find("weight") != std::string::npos)
	{
	    if(vv.size() > 0) // last read is bias matrix
	    {
	        Matrix M(vv);
		__biasVector.push_back(M);
		vv.clear();
	    }
	} 
	else if(line.find("bias") != std::string::npos)
	{
	    if(vv.size() > 0) // last read is weight matrix
	    {
	        Matrix M(vv);
		__weightMatrix.push_back(M);
		vv.clear();
	    }
	}
	else if(line.size() > 0)
	{
	    v_line = parseLine(line);
	    vv.push_back(v_line);
	}
    }

    // save last bias matrix
    if(vv.size() > 0)
    {
        Matrix M(vv);
	__biasVector.push_back(M);
	vv.clear();
    }

    assert(__weightMatrix.size() == __biasVector.size());
    if(__type == LayerType::fullyConnected || __type == LayerType::output)
    {
	assert(__weightMatrix.size() == 1);
	auto dim = __weightMatrix[0].Dimension();
	//cout<<"fc: load file dim:"<<dim<<endl;
	//cout<<"fc neurons: "<<__n_neurons_fc<<endl;
	assert(dim.first == __n_neurons_fc);
    }
    if(__type == LayerType::cnn || __type == LayerType::pooling)
    {
        assert(__weightMatrix.size() == __n_kernels_cnn);
	auto dim = __weightMatrix[0].Dimension();
	//cout<<"cnn: load file dim: "<<dim<<endl;
	//cout<<"cnn kernel dim: "<<__kernelDim<<endl;
	assert(dim == __kernelDim);
    }
}

void ConstructLayer::Print()
{
    // layer id
    std::cout<<"----------- Layer ID: "<<GetID()<<" ------------"<<std::endl;
    // layer type
    if(__type == LayerType::cnn )std::cout<<"layer type: cnn"<<std::endl;
    else if(__type == LayerType::fullyConnected) std::cout<<"layer type: FC"<<std::endl;
    // drop out factor
    std::cout<<"drop out factor: "<<__dropOut<<std::endl;
    std::cout<<"use drop out: "<<__use_drop_out<<std::endl;
    if(__type == LayerType::fullyConnected)
	std::cout<<"number of fc neurons: "<<__n_neurons_fc<<std::endl;
    if(__type == LayerType::cnn)
    {
	std::cout<<"number of cnn kernels: "<<__n_kernels_cnn<<std::endl;
	std::cout<<"kernel  dimension: "<<__kernelDim<<std::endl;
    }
    // weight matrix
    std::cout<<" --- w&b "<<std::endl;
    for(size_t i=0;i<__weightMatrix.size();i++)
    {
	std::cout<<"weight matrix : "<<i<<std::endl;
	std::cout<<__weightMatrix[i]<<std::endl;
	std::cout<<"bias matrix: "<<i<<std::endl;
	std::cout<<__biasVector[i]<<std::endl;
    }
    // drop out filter
    std::cout<<" --- active flag matrix "<<std::endl;
    for(size_t i=0;i<__activeFlag.size();i++)
    {
	std::cout<<"active flag : "<<i<<std::endl;
	std::cout<<__activeFlag[i]<<std::endl;
    }

    // active weight matrix
    std::cout<<" --- active w&b "<<std::endl;
    for(size_t i=0;i<__weightMatrixActive.size();i++)
    {
	std::cout<<"active weight matrix : "<<i<<std::endl;
	std::cout<<__weightMatrixActive[i]<<std::endl;
	std::cout<<"active bias matrix: "<<i<<std::endl;
	std::cout<<__biasVectorActive[i]<<std::endl;
    }

    std::cout<<std::endl<<"==================================="<<std::endl;
    std::cout<<" --- neuron information: "<<std::endl;
    for(size_t ii=0;ii<__neurons.size();ii++)
    {
	auto __neuron_dimension = __neurons[ii].Dimension();
	std::cout<<"Neruon Matrix Dimension: "<<__neuron_dimension<<std::endl;

	for(size_t i=0;i<__neuron_dimension.first;i++)
	{
	    for(size_t j=0;j<__neuron_dimension.second;j++)
	    {
		std::cout<<"coord:  ("<<i<<", "<<j<<")"<<std::endl;
		__neurons[ii][i][j]->Print();
	    }
	}
    }
}
