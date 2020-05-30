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
    InitFilters();

    if(__use_drop_out)  // drop out
	DropOut();

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
    __imageDelta.resize(batch_size);

    __imageAFull.resize(batch_size);
    __imageZFull.resize(batch_size);
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
    else if(__type == LayerType::output)
        InitNeuronsFC();
    else
    {
	std::cout<<"Error: Init neurons, unrecognized layer type."<<std::endl;
	exit(0);
    }
    std::cout<<"Debug: Layer:"<<GetID()<<" init neruons done."<<std::endl;

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
		//__neurons[kk][ii][jj]->SetLayer(this);
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
    assert(__n_neurons_fc >= 1);
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

    if(__p_data_interface == nullptr)
    {
        std::cout<<"Error: must implement/pass DataInterface class before initializing neurons for input layer"
	         <<std::endl;
        exit(0);
    }

    auto dim = __p_data_interface->GetDataDimension();
    assert(dim.second == 1); // make sure matrix transformation has been done
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
	assert(__neurons.size() == 1); // only one kernel
	auto dim = __neurons[0].Dimension();
	Filter2D f(dim.first, dim.second);
	__activeFlag.push_back(f);
    }
    else if(__type == LayerType::output)
    {
	assert(__neurons.size() == 1); // only one kernel
	auto dim = __neurons[0].Dimension();
	assert(dim.second == 1); // only one collum
	Filter2D f(dim.first, dim.second);
	__activeFlag.push_back(f);
    }
    else if(__type == LayerType::fullyConnected)
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

void ConstructLayer::ForwardPropagateForSample(int sample_index)
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
		__neurons[k][i][j] -> ForwardPropagateForSample(sample_index);
	    }
	}
    }

    // when propagation for this layer is done, update the A, Z matrices (extract value from neurons and update them to layer)
    UpdateImagesA(sample_index);
    UpdateImagesZ(sample_index);
}

void ConstructLayer::BackwardPropagateForSample(int sample_index)
{
    // backward propagation:
    //     ---) compute delta for this layer
    //     ---) only after all samples in this batch finished forward propagation, one can do this backward propagation

    for(size_t k=0;k<__neurons.size();k++)
    {
	auto dim = __neurons[k].Dimension();
	for(size_t i=0;i<dim.first;i++)
	{
	    for(size_t j=0;j<dim.second;j++)
	    {
		if(__neurons[k][i][j]->IsActive()){
		    __neurons[k][i][j] -> BackwardPropagateForSample(sample_index);
		}
	    }
	}
    }

    // when propagation for this layer is done, update the Delta matrices
    UpdateImagesDelta(sample_index);
}


// cost functions ---------- cross entropy 
static double cross_entropy(Matrix &A, Matrix &Y)
{
    auto dim = A.Dimension();
    assert(dim == Y.Dimension());
    assert(dim.second == 1);

    double res = 0.;
    for(size_t i=0;i<dim.first;i++)
    {
	res += Y[i][0] * log(A[i][0]) + (1. - Y[i][0]) * log(1. - A[i][0]);
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
    // for output layer
    // propagation(backward and forward) are integrated in this function

    // --- 1) first do forward propagation, calculate z and a
    ForwardPropagateForSample(sample_index);

    // --- 2) then compute the cost function C(a_i, y_i)
    //      if you want a softmax layer, then the softmax should also be done here
    //size_t sample_number = __imageA.size(); // obsolete

    //Images sample_image = __imageA.back();
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


    // --- 3) then calculate delta: delta = delta(a_i, y_i) for this sample
    // -------------- please note: this function only calculate \delta for current sample, which cannot be used for backpropagation
    // ------------------ the \delta used for backpropagation should be a sum of all \deltas in this batch
    //
    // ------------------ so: on batch level, you need to forward propagate n_batch_size times
    // ----------------------- but only back propagate 1 time
    //
    // ----------------------- in other words, backpropagation only happens when all forwardpropagation finished !!! !!! ******
    // ----------------------- this is because THE COST_FUNCTION is defined as a sum over all samples in one batch !!!!! ******

    Matrix delta = Matrix(sample_A.Dimension());
    if(__cost_func_type == CostFuncType::cross_entropy)
    {
	delta = sample_A - sample_label;
    }
    else if(__cost_func_type == CostFuncType::log_likelihood)
    {
	delta = sample_A - sample_label;
    }
    else if(__cost_func_type == CostFuncType::quadratic_sum)
    {
	delta = sample_A - sample_label;
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


void ConstructLayer::FillDataToInputLayerA()
{
    // if this layer is input layer, then fill the 'a' matrix directly with input image data
    auto input_data = __p_data_interface->GetCurrentBatchData();

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

    vector<vector<float>> R;
    for(size_t i=0;i<dimA.first;i++)
    {
	vector<float> _tmp_row;
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
    else if(__type == LayerType::cnn) // for cnn layer
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
    else // for other layer types
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

    if(__type == LayerType::fullyConnected) // for fully connected layer
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
    else if(__type == LayerType::cnn) // for cnn layer
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
    //    so imageZ will clear after each batch is done
    //    **** on batch level, the filter matrix stays the same, so no need to worry the change of filter matrix on batch level
    //
    //    The above comment is copied from UpdateImagesZ(); just to refresh your memory, no use in this function
    //

    // extract the A matrices from neurons for current traning sample
    Images sample_image_delta;
    Images sample_image_delta_full;

    if(__type == LayerType::fullyConnected) // for fully connected layer
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
    else if(__type == LayerType::cnn) // for cnn layer
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
    else // for other layer types
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
		if(__prevLayer != nullptr &&   // current layer is not input layer
			(__prevLayer->GetType() == LayerType::input || __prevLayer->GetType() == LayerType::fullyConnected ) // only fc layer and input layer no need for other type layers
		  )
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
    else if (__type == LayerType::fullyConnected || __type == LayerType::output)
    {
	map_matrix_fc(__weightMatrix[0], __activeFlag[0]);
    }

    // update active neuron coord after drop out
    if(__type == LayerType::cnn)
    {
	__activeNeuronDim = __neuronDim; // drop out not happening on neurons, so dimension stays same
    }
    else if(__type == LayerType::fullyConnected || __type == LayerType::output)
    {
	__activeNeuronDim = __neuronDim; // update active neuron dimension
	size_t active_neurons = __weightMatrixActive.size();
	__activeNeuronDim.i = active_neurons;
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

    __imageAFull.clear();
    __imageZFull.clear();
    __imageDeltaFull.clear();

}

NeuronCoord ConstructLayer::GetActiveNeuronDimension()
{
    return __activeNeuronDim;
}

void ConstructLayer::SetDropOutFactor(float f)
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
    auto a_images = __prevLayer->GetImagesFullA(); // a images from previous layer
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
    dw = dw * float(__learningRate/(double)M); // over batch 
    //cout<<"learning rate: "<<float(__learningRate/(double)M)<<endl;

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

    if(__prevLayer->GetType() == LayerType::input) // only fc layer, other layers won't affect
    {
	auto prevLayerFilters = __prevLayer->GetActiveFlag();
	assert(prevLayerFilter.size() == 1);

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
	Matrix regularization_M = (__weightMatrix[0]^F) *(__learningRate * __regularizationParameter/((float)M)); 
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
