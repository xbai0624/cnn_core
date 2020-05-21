#include <iostream>
#include <numeric> // std::iota
#include <cassert>
#include <cmath>

#include "Matrix.h"
#include "ConstructLayer.h"
#include "Neuron.h"
#include "Tools.h"
#include "DataInterface.h"

using namespace std;

static float SGN(float x)
{
    if( x == 0) return 0;
    return x>0?1:-1;
}

ConstructLayer::ConstructLayer()
{
    // place holder
}

ConstructLayer::ConstructLayer(LayerType t)
{
    __type = t;
}

ConstructLayer::ConstructLayer(LayerType t, int n_neurons)
{
    __type = t;
    SetNumberOfNeuronsFC((size_t)n_neurons);
}

ConstructLayer::ConstructLayer(LayerType t, int n_kernels, std::pair<size_t, size_t> d)
{
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
	__p_data_interface = new DataInterface();
	ImplementInputLayerA();
	InitNeurons();
	InitFilters(); // input layer need to init filters, make all neurons active
    }
    else  // other layer
    {
	InitNeurons();
	InitWeightsAndBias();
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
    InitFilters();

    if(__use_drop_out)  // drop out
	DropOut();

    // only fc layer; cnn layer image won't change
    if(__type == LayerType::fullyConnected) 
    {
	UpdateCoordsForActiveNeuronFC();
    }

    // prepare weights and bias
    UpdateActiveWeightsAndBias();

    // assign active weights and bias to neurons
    AssignWeightsAndBiasToNeurons();

    // clear training information from last batch
    __imageA.clear();
    __imageZ.clear();
    __imageDelta.clear();
}

void ConstructLayer::ProcessSample()
{
}

void  ConstructLayer::SetNumberOfNeuronsFC(size_t n)
{
    if(__type != LayerType::fullyConnected) {
	std::cout<<"Error: needs to set layer type before setting number of neurons."
	    <<std::endl;
	exit(0);
    }
    __n_neurons_fc = n;
}

void  ConstructLayer::SetNumberOfKernelsCNN(size_t n)
{
    if(__type != LayerType::cnn)
    {
	std::cout<<"Error: needs to set layer type before setting number of kernels."
	    <<std::endl;
	exit(0);
    }
    __n_kernels_cnn = n;
}

void  ConstructLayer::SetKernelSizeCNN(std::pair<size_t, size_t> s)
{
    if(__type != LayerType::cnn)
    {
	std::cout<<"Error: needs to set layer type before setting kernel size."
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
    else if(__type == LayerType::input)
	InitNeuronsInputLayer();
    else if(__type == LayerType::fullyConnected)
	InitNeuronsFC();
    else
    {
	std::cout<<"Error: Init neurons, unrecognized layer type."<<std::endl;
	exit(0);
    }
    std::cout<<"Debug: Layer:"<<GetID()<<" init neruons done."<<std::endl;

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
		__neurons[kk][ii][jj]->SetPreviousLayer(__prevLayer);
		__neurons[kk][ii][jj]->SetNextLayer(__nextLayer);
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
    //std::cout<<"prev layer output image size: "<<size_prev_layer<<std::endl;

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
		image[i][j] = n;
	    }
	}
	__neurons.push_back(image);
    }
}

void ConstructLayer::InitNeuronsFC()
{
    __neurons.clear();
    Pixel2D<Neuron*> image(__n_neurons_fc, 1);
    for(size_t i=0;i<__n_neurons_fc;i++)
    {
	Neuron *n = new Neuron();
	image[i][0] = n;
    }
    __neurons.push_back(image);
}


void ConstructLayer::InitNeuronsInputLayer()
{
    __neurons.clear();
    if(__imageA.size() <= 0)
    {
        std::cout<<"Error: must initialize 'A' matrix before initializing neurons for input layer"
	         <<std::endl;
        exit(0);
    }

    Matrix tmp = __imageA[0].OutputImageFromKernel[0]; // first sample
    auto dim = tmp.Dimension();
    //cout<<"Info::input layer dimension: "<<dim<<endl;
    Pixel2D<Neuron*> image(dim.first, dim.second);
    for(size_t i=0;i<dim.first;i++)
    {
	for(size_t j=0;j<dim.second;j++)
	{
	    Neuron *n = new Neuron();
	    image[i][0] = n;
	}
    }
    __neurons.push_back(image);
}



void ConstructLayer::InitFilters()
{
    // clear previous filter
    __activeFlag.clear();

    // init filter 2d matrix, fill true value to all elements
    if(__type == LayerType::input)
    {
	assert(__neurons.size() == 1);
	auto dim = __neurons[0].Dimension();
	Filter2D f(dim.first, dim.second);
	__activeFlag.push_back(f);
    }
    if(__type == LayerType::fullyConnected)
    {
	assert(__neurons.size() == 1);
	auto dim = __neurons[0].Dimension();
	Filter2D f(dim.first, dim.second);
	__activeFlag.push_back(f);
    }
    else if(__type == LayerType::cnn)
    {
	size_t nKernels = __weightMatrix.size();
	for(size_t i=0;i<nKernels;i++)
	{
	    auto dim = __weightMatrix[i].Dimension();
	    Filter2D f(dim.first, dim.second);
	    __activeFlag.push_back(f);
	}
    }
}

void  ConstructLayer::InitWeightsAndBias()
{
    // init weights and bias
    // no need to init active w&b, they will be filled in Batch starting phase
    // clear everything
    __weightMatrix.clear();
    __biasVector.clear();

    if(__type == LayerType::fullyConnected)
    {
	int n_prev = 0;
	int n_curr = GetNumberOfNeurons();
	if(__prevLayer == nullptr) // input layer
	{
	    std::cout<<"INFO: layer"<<GetID()
		<<" has no prev layer, default 10 neruons for prev layer was used."
		<<std::endl;
	    n_prev = 10;
	}
	else n_prev = __prevLayer->GetNumberOfNeurons();

	Matrix w(n_curr, n_prev);
	w.RandomGaus(0., 1./sqrt((float)n_curr)); // (0, sqrt(n_neuron)) normal distribution
	__weightMatrix.push_back(w);

	Matrix b(n_curr, 1);
	b.RandomGaus(0., 1.); // (0, 1) normal distribution
	__biasVector.push_back(b);
    }
    else if(__type == LayerType::cnn)
    {
	auto o_d = GetOutputImageSizeCNN();
	float n_curr = (float)o_d.first * (float)o_d.second;
	for(size_t i=0;i<__n_kernels_cnn;i++)
	{
	    Matrix w(__kernelDim);
	    w.RandomGaus(0., 1./sqrt(n_curr));
	    __weightMatrix.push_back(w);

	    Matrix b(1, 1);
	    b.RandomGaus(0., 1.); // (0, 1) normal distribution
	    __biasVector.push_back(b);
	}
    }
    else if(__type == LayerType::pooling)
    {
    }
    else {
	std::cout<<"Error: need layer type info before initing w&b."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::ForwardPropagate()
{
    // forward propagation: 
    //     ---) compute Z, A, A'(Z) for this layer
    //          these works are done neuron by neuron (neuron level)

    //cout<<"total neuron dimension: "<<__neuronDim<<endl; 
    for(size_t k=0;k<__neurons.size();k++)
    {
	auto dim = __neurons[k].Dimension();
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		//cout<<"coord (i, j, k): ("<<i<<", "<<j<<", "<<k<<")"<<endl;
		if(!__neurons[k][i][j]->IsActive()) continue;
		__neurons[k][i][j] -> UpdateZ();
		__neurons[k][i][j] -> UpdateA();
		__neurons[k][i][j] -> UpdateSigmaPrime();
	    }
	}
    }
}

void ConstructLayer::BackwardPropagate()
{
    // backward propagation:
    //     ---) compute delta for this layer

    for(size_t k=0;k<__neurons.size();k++)
    {
	auto dim = __neurons[k].Dimension();
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		if(__neurons[k][i][j]->IsActive()){
		    __neurons[k][i][j] -> UpdateDelta();
		}
	    }
	}
    }
}

std::vector<Images>& ConstructLayer::GetImagesA()
{
    if(__type == LayerType::fullyConnected || __type == LayerType::cnn)
        UpdateImagesA(); // append results from last sample

    return __imageA;
}


void ConstructLayer::ImplementInputLayerA()
{
    // if this layer is input layer, then fill the 'a' matrix directly with input image data

    auto input_data = __p_data_interface->GetNewBatch();

    // load all batch data to memory, this should be faster
    for(auto &i: input_data)
    {
	Images image_a; // one image

	// input layer only has one kernel
	image_a.OutputImageFromKernel.push_back(i);

	// push this image to images of this batch
	__imageA.push_back(image_a);
    }
}

void ConstructLayer::UpdateImagesA()
{
    size_t l = __imageA.size();

    if(__type == LayerType::fullyConnected)
    {
    }
    else if(__type == LayerType::cnn)
    {
    }
    else
    {
    }
}

std::vector<Images>& ConstructLayer::GetImagesZ()
{
    if(__type == LayerType::fullyConnected || __type == LayerType::cnn)
        UpdateImagesZ(); // append results from last sample
 
    return __imageZ;
}

void ConstructLayer::UpdateImagesZ()
{
}


std::vector<Images>& ConstructLayer::GetImagesDelta()
{
    if(__type == LayerType::fullyConnected || __type == LayerType::cnn)
        UpdateImagesDelta(); // append results from last sample
 
    return __imageDelta;
}

void ConstructLayer::UpdateImagesDelta()
{
}

void ConstructLayer::UpdateCoordsForActiveNeuronFC()
{
    // enable/disable neuron; and
    // update coords for active neuron
    assert(__neurons.size() == 1); 
    auto dim = __neurons[0].Dimension();
    assert(dim.second == 1);

    // get filter
    assert(__activeFlag.size() == 1);

    size_t active_i = 0;
    for(size_t i=0;i<dim.first;i++)
    {
	// first reset coords back
	__neurons[0][i][0]->SetCoord(0, i, 0);

	// then set coord according to filter mask
	if(!__activeFlag[0][i][0]){
	    __neurons[0][i][0]->Disable();
	    continue;
	}
	__neurons[0][i][0]->Enable();
	__neurons[0][i][0]->SetCoord(0, active_i, 0);
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
    if(__type == LayerType::fullyConnected)
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
    else if(__type == LayerType::cnn)
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
    else if(__type == LayerType::cnn)
	__UpdateActiveFlagCNN();
    else 
    {
	std::cout<<"Error: drop out, un-recongnized layer type."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::EnableDropOut()
{
    __use_drop_out = true;
}

void ConstructLayer::DisableDropOut()
{
    __use_drop_out = false;
}

void ConstructLayer::__UpdateActiveFlagFC()
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
    Shuffle(tmp);

    // mask out filters
    for(size_t i=0;i<dim.first;i++)
    {
	__activeFlag[0][i][0] = tmp[i];
    }
}

void ConstructLayer::__UpdateActiveFlagCNN()
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
	if(__type == LayerType::cnn) {
	    active_to_original_cnn(__weightMatrix[i], __activeFlag[i], __weightMatrixActive[i]);
	}
	else if(__type == LayerType::fullyConnected)
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
    if(__type == LayerType::cnn) 
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
	    std::vector<float> act_row;
	    for(size_t j=0;j<dim.second;j++)
	    {
		if(__prevLayer != nullptr) // current layer is not input layer
		{
		    auto filter_prev = __prevLayer->GetActiveFlag();
		    assert(filter_prev.size() == 1);
		    if(!filter_prev[0][j][0])
			continue;
		}
		act_row.push_back(ori_M[i][j]);
	    }
	    std::vector<std::vector<float>> act_M;
	    act_M.push_back(act_row);
	    Matrix tmp(act_M);
	    __weightMatrixActive.push_back(tmp);

	    // bias
	    std::vector<float> act_bias;
	    act_bias.push_back(__biasVector[0][i][0]);
	    std::vector<std::vector<float>> act_b;
	    act_b.push_back(act_bias);
	    Matrix tmp_b(act_b);
	    __biasVectorActive.push_back(tmp_b);
	}
    };

    // start mapping
    if(__type == LayerType::cnn)
    {
	for(size_t k=0; k<nKernel;k++)
	{
	    map_matrix_cnn(__weightMatrix[k], __activeFlag[k]);
	}
	// for cnn, drop out won't change threshold
	__biasVectorActive = __biasVector; 
    }
    else if (__type == LayerType::fullyConnected)
    {
	map_matrix_fc(__weightMatrix[0], __activeFlag[0]);
    }

    std::cout<<"Debug: Layer:"<<GetID()<<" TransferValueFromOriginalToActiveWB() done."<<std::endl;
} 

void ConstructLayer::UpdateImageForCurrentTrainingSample()
{
    // loop for all neurons

}

void ConstructLayer::ClearImage()
{
    __imageA.clear();
    __imageZ.clear();
    __imageDelta.clear();
}

NeuronCoord ConstructLayer::GetActiveNeuronDimension()
{
    return __activeNeuronDim;
}

void ConstructLayer::SetDropOutFactor(float f)
{
    __dropOut = f;
}

std::vector<Matrix>* ConstructLayer::GetWeightMatrix()
{
    return &__weightMatrixActive;
}

std::vector<Matrix>* ConstructLayer::GetBiasVector()
{
    return &__biasVectorActive;
}

LayerType ConstructLayer::GetType()
{
    return __type;
}

float ConstructLayer::GetDropOutFactor()
{
    return __dropOut;
}

std::vector<Filter2D>& ConstructLayer::GetActiveFlag()
{
    return __activeFlag;
}

void ConstructLayer::UpdateWeightsAndBias()
{
    // after finishing one training sample, update weights and bias
    auto layerType = this->GetType();

    if(layerType == LayerType::fullyConnected) 
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
	std::cout<<"Error: update weights and bias, unsupported layer type."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::UpdateWeightsAndBiasGradients()
{
    // after finishing one training sample, update weights and bias
    auto layerType = this->GetType();

    if(layerType == LayerType::fullyConnected) 
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
	std::cout<<"Error: update weights and bias gradients, unsupported layer type."<<std::endl;
	exit(0);
    }
}

void ConstructLayer::UpdateWeightsAndBiasGradientsFC()
{
    // after finishing one batch, update weights and bias, for FC layer
    // get output from previous layer for current training sample
    auto a_images = __prevLayer->GetImagesA(); // a images from previous layer
    auto d_images = this->GetImagesDelta(); // delta images from current layer
    if(a_images.size() != d_images.size()) {
	std::cout<<"Error: batch size not equal..."<<std::endl;
	exit(0);
    }
    if(a_images.back().OutputImageFromKernel.size() != 1 ) {
	std::cout<<"Error: updating weights and bias gradients for FC, dimension not match."<<std::endl;
	exit(0);
    }

    // check layer type
    if(__weightMatrixActive.size() != 1) {
	std::cout<<"Error: more than 1 weight matrix exist in fully connected layer."<<std::endl;
	exit(0);
    }

    // loop for batch
    for(size_t i=0;i<a_images.size();i++)
    {
	Matrix a_matrix = a_images[i].OutputImageFromKernel[0]; // 'a' image from previous layer
	Matrix d_matrix = d_images[i].OutputImageFromKernel[0]; // 'd' image from current layer

	auto d1 = (__weightMatrixActive[0]).Dimension(), d2 = a_matrix.Dimension(), d3 = d_matrix.Dimension();
	if(d1.first != d3.first || d1.second != d2.first)
	{
	    std::cout<<"Error: updating weights and bias gradients for FC, dimension not match."<<std::endl;
	    exit(0);
	}

	Matrix a_T = a_matrix.Transpose();
	Matrix dw = d_matrix * a_T;

	// push bias gradient for current training sample
	Images tmp_w_gradients;
	tmp_w_gradients.OutputImageFromKernel.push_back(dw);
	__wGradient.push_back(tmp_w_gradients); // push weight gradient for current training sample

	Images tmp_b_gradients;
	tmp_b_gradients.OutputImageFromKernel.push_back(d_matrix);
	__bGradient.push_back(tmp_b_gradients); // bias gradient equals delta
    }
} 

void ConstructLayer::UpdateWeightsAndBiasFC()
{
    // after finishing one batch, update weights and bias, for FC layer
    // get output from previous layer for current training sample
    size_t M = __imageDelta.size(); // batch size
    if( M != __wGradient.size() ) {
	std::cout<<"Error: update FC weights, batch size not match."<<std::endl;
	exit(0);
    }
    if( __wGradient[0].OutputImageFromKernel.size() != 1 || __bGradient[0].OutputImageFromKernel.size()!=1) 
    {
	std::cout<<"Error: update FC weiths, more than 1 w gradient matrix found."
	    <<std::endl;
	exit(0);
    }

    // gradient descent
    if(__weightMatrixActive.size() != 1) {
	std::cout<<"Error: update FC layer weights, more than 1 weight matrix found."<<std::endl;
	exit(0);
    }
    Matrix dw((__weightMatrixActive[0]).Dimension()); 
    for(size_t i=0;i<M;i++){ // sum x (batches)
	dw  = dw + __wGradient[i].OutputImageFromKernel[0];
    }
    dw = dw * float(__learningRate/(double)M); // over batch 

    // Regularization
    double f_regularization = 0.;
    if(__regularizationMethod == Regularization::L2) 
    {
	f_regularization = 1 - __learningRate * __regularizationParameter / M;
	(__weightMatrixActive[0]) = (__weightMatrixActive[0]) * f_regularization - dw;
    } 
    else if(__regularizationMethod == Regularization::L1) 
    {
	Matrix _t = (__weightMatrixActive[0]); // make a copy of weight matrix
	_t(&SGN); // get the sign for each element in weight matrix
	_t = _t * (__learningRate*__regularizationParameter/(double)M); // L1 regularization part
	(__weightMatrixActive[0]) = (__weightMatrixActive[0]) -  _t; // apply L1 regularization to weight matrix
	(__weightMatrixActive[0]) = (__weightMatrixActive[0]) - dw; // apply gradient decsent part
    } 
    else 
    {
	std::cout<<"Error: update FC weights, unsupported regularization."<<std::endl;
	exit(0);
    }

    // bias
    Matrix db(__biasVector[0].Dimension());
    for(size_t i=0;i<M;i++){
	db = db + __bGradient[i].OutputImageFromKernel[0];
    }
    db = db / (double)M;
    db = db * __learningRate;
    __biasVector[0] = __biasVector[0] - db;
}

void ConstructLayer::UpdateWeightsAndBiasGradientsCNN()
{
    // after finishing one batch, update weights and bias gradient, (CNN layer)
    // cnn layer is different with FC layer. For FC layer, different 
    // neurons have different weights, however, for CNN layer, different
    // neurons in one image share the same weights and bias (kernel).
    // So one need to loop for kernels

    // get 'a' matrix from previous layer for current training sample
    auto aVec = __prevLayer -> GetImagesA();
    // get 'delta' matrix for current layer
    auto deltaVec = this -> GetImagesDelta();

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
		    // gradient descent part
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
    // after finishing one training sample, update weights and bias gradient, (CNN layer)
    // cnn layer is different with FC layer. For FC layer, different 
    // neurons have different weights, however, for CNN layer, different
    // neurons in one image share the same weights and bias.
    // so when you are
    // using this function in layer class, you need to loop over every 
    // neuron for FC layer, but you should not loop over neruons in 
    // the same image for CNN layer, you should just use only one neuron,
    // anyone would be fine, instead you need to loop over images.

    // after finishing one batch, update weights and bias, CNN layer
    size_t M = __imageDelta.size(); // batch size
    if( M != __wGradient.size() ) {
	std::cout<<"Error: update FC weights, batch size not match."<<std::endl;
	exit(0);
    }

    // loop for kernel
    size_t nKernel = __weightMatrixActive.size();
    for(size_t k=0;k<nKernel;k++)
    {
	// gradient descent
	Matrix dw(__weightMatrixActive[k].Dimension());
	// loop for batch
	for(size_t i=0;i<M;i++){ 
	    dw  = dw + __wGradient[i].OutputImageFromKernel[k];
	}
	dw = dw * float(__learningRate/(double)M); // gradients average over batch size

	// regularization part
	double f_regularization = 0;
	if(__regularizationMethod == Regularization::L2){
	    f_regularization = 1 - __learningRate * __regularizationParameter / (float)M;
	    (__weightMatrixActive[k]) = (__weightMatrixActive[k])*f_regularization;
	    (__weightMatrixActive[k]) = (__weightMatrixActive[k]) - dw;
	}
	else if(__regularizationMethod == Regularization::L1){
	    Matrix tmp = (__weightMatrixActive[k]);
	    tmp(&SGN);
	    tmp = tmp * (__learningRate*__regularizationParameter/(float)M);
	    //(*__w) = (*__w) - tmp - dw;
	    (__weightMatrixActive[k]) = (__weightMatrixActive[k]) - dw;
	    (__weightMatrixActive[k]) = (__weightMatrixActive[k]) - tmp;
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
	db = db * float(__learningRate / (float)M);
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

int ConstructLayer::GetNumberOfNeurons()
{
    if(__type == LayerType::fullyConnected){
	// used for setup fc layer
	return __n_neurons_fc;
    }
    else if(__type == LayerType::input)
    {
        auto dim = __neurons[0].Dimension();
	return static_cast<int>((dim.first * dim.second));
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

