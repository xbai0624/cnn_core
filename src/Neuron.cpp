#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>

#include "Layer.h"
#include "Neuron.h"
#include "Matrix.h"
#include "DataInterface.h"

using namespace std;

long Neuron::__neuron_Count = 0;

Neuron::Neuron()
{
    // place holder
    __neuron_id = __neuron_Count;
    __neuron_Count++;
}

Neuron::~Neuron()
{
    // place holder
}

void Neuron::PassWeightPointer(Matrix *m)
{
    // pass weight matrix pointer
    // matrix dimension is set in Layer class
    __w = m;
}

void Neuron::PassBiasPointer(Matrix *_b)
{
    // pass bias pointer
    __b = _b;
}

void Neuron::ForwardPropagateForSample(int sample_index)
{
    UpdateZ(sample_index);
    UpdateA(sample_index);
    UpdateSigmaPrime(sample_index);
}

void Neuron::BackwardPropagateForSample(int sample_index)
{
    UpdateDelta(sample_index);
}


double Neuron::__sigmoid(double z)
{
    return 1/(1 + exp(-z));
}

double Neuron::__tanh(double z)
{
    double _t = exp(2.0*z);
    return (_t - 1.)/(_t +1.);
}

double Neuron::__relu(double z)
{
    // return max(0, z);
    if(z < 0.) return 0;
    return z;
}

void Neuron::Disable()
{
    __active_status = false;
}

void Neuron::Enable()
{
    __active_status = true;
}

bool Neuron::IsActive()
{
    return __active_status;
}

void Neuron::Reset()
{
    if(__layer == nullptr)
    {
        std::cout<<"Error: Neuron::Reset(): the layer info has not been set for this neuron."
	         <<endl;
        exit(0);
    }
    int batch_size = __layer->GetBatchSize();
    // after updateing weights and biase, 
    // namely after one training (one sample or one batch depends on user)
    // reset neurons for next computation
    __a.resize(batch_size); // reset a
    __delta.resize(batch_size); // reset delta
    __z.resize(batch_size); // reset z
    __sigmaPrime.resize(batch_size);

    //__wGradient.clear(); // reset weight gradient
    //__bGradient.clear(); // reset bias gradient
}

void Neuron::SetLayer(Layer* l)
{
    // set layer that this neuron currently belongs to
    __layer = l;
}

//void Neuron::SetPreviousLayer(Layer* l)
//{
    // set neuron's prevous layer
    //__previousLayer = l;
//}

//void Neuron::SetNextLayer(Layer* l)
//{
    // set neurons's next layer
    //__nextLayer = l;
//}

void Neuron::SetActuationFuncType(ActuationFuncType t)
{
    // set neurons's actuation function type
    __funcType = t;
}

void Neuron::UpdateZ(int sample_index)
{
    // update z for current layer
    if(__layer->GetType() == LayerType::fullyConnected || __layer->GetType() == LayerType::output)
    {
	//std::cout<<"fully connected layer update z"<<std::endl;
	UpdateZFC(sample_index);
    } 
    else if(__layer->GetType() == LayerType::cnn)
    {
	//std::cout<<"cnn layer update z"<<std::endl;
	UpdateZCNN(sample_index);
    } 
    else if(__layer->GetType() == LayerType::pooling)
    {
	//std::cout<<"pooling layer update z"<<std::endl;
	UpdateZPooling(sample_index);
    } 
    else 
    {
	std::cout<<"Error: unsupported layer type."<<std::endl;
	exit(0);
    }
}

void Neuron::UpdateZFC(int sample_index)
{
    // for fully connected layer, matrix reform are done by Layer class
    // currently layer and its previous layer are fully connected
    // weight matrix dimension will be: (1, M)
    Layer* __previousLayer = __layer->GetPrevLayer();
    auto _t = __previousLayer -> GetImagesA();
    //cout<<"batch size: "<<_t.size()<<endl;
    //cout<<"kernel number: "<<_t[0].GetNumberOfKernels()<<endl;
    //cout<<" image dimension: "<<_t[0].OutputImageFromKernel[0].Dimension()<<endl;
    if(_t.size() < 1 || _t.size() < (size_t)sample_index) 
    {
	std::cout<<"Error: previous layer has not 'A' image."<<std::endl;
	exit(0);
    }
    //Images &images = _t.back(); // get images for current sample
    Images &images = _t[sample_index]; // get images for current sample
    if(images.OutputImageFromKernel.size() != 1) 
    {
	std::cout<<"Eroor: layer type not match, expecting FC layer, FC layer should only have 1 kernel."<<std::endl;
	exit(0);
    }

    Matrix &image = images.OutputImageFromKernel[0]; // FC layer has only one "kernel" (equivalent kernel)
    //cout<<"neuron id: "<<__neuron_id<<", image A debug:"<<endl;
    //cout<<image<<endl;


    //cout<<"weight matrix dimension: "<<(*__w).Dimension()<<endl;
    //cout<<(*__w)<<endl;
    Matrix res = (*__w) * image;
    //cout<<"res matrix: "<<endl;
    //cout<<res<<endl;
    auto dim = res.Dimension();
    if(dim.first != 1 || dim.second != 1) 
    {
	std::cout<<"Error: wrong dimension, expecting 1D matrix."<<std::endl;
    }
    double z = res[0][0];
    z = z + (*__b)[0][0];

    //__z.push_back(z);
    __z[sample_index] = z;
}

void Neuron::UpdateZCNN(int sample_index)
{
    // cnn layer
    // every single output image needs input from all input images
    Layer *__previousLayer = __layer->GetPrevLayer();
    auto inputImage = __previousLayer->GetImagesA();
    auto w_dim = __w->Dimension();
    int stride = __layer->GetCNNStride();

    size_t i_start = __coord.i*stride;
    size_t j_start = __coord.j*stride;
    size_t i_end = i_start + w_dim.first;
    size_t j_end = j_start + w_dim.second;

    //auto &current_sample_image = (inputImage.back()).OutputImageFromKernel;
    auto &current_sample_image = (inputImage[sample_index]).OutputImageFromKernel;
    auto image_dim = current_sample_image[0].Dimension();
    if(i_end > image_dim.first || j_end > image_dim.second)
    {
	std::cout<<"Error: cnn z update: matrix dimension not match, probably due to wrong padding."<<std::endl;
	exit(0);
    }

    // compute z
    double res = 0;
    for(auto &m: current_sample_image)
    {
	for(size_t i=0;i<w_dim.first;i++){
	    for(size_t j=0;j<w_dim.second;j++){
		res += m[i+i_start][j+j_start] * (*__w)[i][j];
	    }
	}
    }
    double z = res + (*__b)[0][0];
    //__z.push_back(z);
    __z[sample_index] = z;
}

void Neuron::UpdateZPooling(int sample_index)
{
    // pooling layer
    // should be with cnn layer, just kernel matrix all elements=1, bias = 0;
    Layer* __previousLayer = __layer->GetPrevLayer();
    auto inputImage = __previousLayer->GetImagesA();
    //if(inputImage.back().OutputImageFromKernel.size() < __coord.k)
    if(inputImage[sample_index].OutputImageFromKernel.size() < __coord.k)
    {
	// output image for current sample
	// pooling layer is different with cnn layer
	// in pooling layer, kernel and 'A' images has a 1-to-1 mapping relationship
	// for pooling layer, number of kernels (previous layer)  = number of kernels (current layer)
	std::cout<<"Error: pooling operation matrix dimension not match"<<std::endl;
	exit(0);
    }
    //Images image = inputImage.back(); // images for current training sample
    Images image = inputImage[sample_index]; // images for current training sample
    std::vector<Matrix> & images = image.OutputImageFromKernel;
    Matrix &kernel_image = images[__coord.k];

    // get pooling stride
    auto dim = __w->Dimension();
    size_t i_size = dim.first, j_size = dim.second;

    size_t i_start = __coord.i * i_size;
    size_t j_start = __coord.j * j_size;

    // get pooling method
    PoolingMethod __poolingMethod = __layer->GetPoolingMethod();

    double z;
    if(__poolingMethod == PoolingMethod::Max) 
    {
	z = kernel_image.MaxInSection(i_start, i_start + i_size, j_start, j_start+j_size);
    } 
    else if(__poolingMethod == PoolingMethod::Average) 
    {
	z = kernel_image.AverageInSection(i_start, i_start + i_size, j_start, j_start+j_size);
    }
    else 
    {
	std::cout<<"Error: unspported pooling method, only max and average supported."<<std::endl;
	exit(0);
    }

    //__z.push_back(z);
    __z[sample_index] = z;
}

void Neuron::UpdateA(int sample_index)
{
    // update a for current training sample
    //double v = __z.back();
    double v = __z[sample_index];
    double a = -100.; // theorectially, a>=-1
    if(__funcType == ActuationFuncType::Sigmoid)
	a = __sigmoid(v);
    else if(__funcType == ActuationFuncType::Tanh)
	a = __tanh(v);
    else if(__funcType == ActuationFuncType::Relu)
	a = __relu(v);
    else
	std::cout<<"Error: unsupported actuation function type."<<std::endl;

    if(a < -1) 
    {
        std::cout<<"Error: Neuron::UpdateA(int sample_index), a<-1? something wrong."
	         <<endl;
	exit(0);
    }
    //__a.push_back(a);
    __a[sample_index] = a;
}

void Neuron::UpdateSigmaPrime(int sample_index)
{
    // update sigma^prime
    //if(__sigmaPrime.size() != __z.size()-1) 
    //{
	//std::cout<<"Error: computing sigma^prime needs z computed first."<<std::endl;
	//exit(0);
    //}
    //if(__sigmaPrime.size() != __a.size()-1) 
    //{
	//std::cout<<"Error: computing sigma^prime needs a computed first."<<std::endl;
	//exit(0);
    //}

    //double a = __a.back();
    //double z = __z.back();
    double a = __a[sample_index];
    double z = __z[sample_index];
    double sigma_prime = -100; // theoretically, it must between [0, 1]

    if(__funcType == ActuationFuncType::Sigmoid) 
    {
	sigma_prime = (1-a)*a;
    }
    else if(__funcType == ActuationFuncType::Tanh) 
    {
	sigma_prime = ( 2 / (exp(z) + exp(-z)) ) * ( 2 / (exp(z) + exp(-z)) );
    }
    else if(__funcType == ActuationFuncType::Relu) 
    {
	if( z > 0) sigma_prime = 1;
	else sigma_prime = 0;
    }
    else
	std::cout<<"Error: unsupported actuation function type in direvative."<<std::endl;
 
    if(sigma_prime < 0)
    {
        std::cout<<"Error: Neuron::UpdateSigmaPrime: sigma_prime<0? something wrong."
	         <<endl;
	exit(0);
    }

    //__sigmaPrime.push_back(sigma_prime);
    __sigmaPrime[sample_index] = sigma_prime;
}

void Neuron::UpdateDelta(int sample_index)
{
    // update delta for current layer
    if(__layer->GetType() == LayerType::output)
    {
	UpdateDeltaOutputLayer(sample_index);
    } 
    else if(__layer->GetType() == LayerType::fullyConnected)
    {
	UpdateDeltaFC(sample_index);
    } 
    else if(__layer->GetType() == LayerType::cnn)
    {
	UpdateDeltaCNN(sample_index);
    } 
    else if(__layer->GetType() == LayerType::pooling)
    {
	UpdateDeltaPooling(sample_index);
    } 
    else 
    {
	std::cout<<"Error: unsupported layer type."<<std::endl;
	exit(0);
    }
}

void Neuron::UpdateDeltaOutputLayer(int sample_index)
{
     // back propagation delta for output layer
    if(__sigmaPrime.size() <= 0) 
    {
	std::cout<<"Error: Neuron::UpdateDeltaOutputLayer() computing delta needs sigma^prime computed first."<<std::endl;
	std::cout<<"        "<<__delta.size()<<" deltas, "<<__sigmaPrime.size()<<" sigma^primes"<<endl;
	exit(0);
    }
    assert(__a.size() == __sigmaPrime.size()); // make sure all have been updated
    auto labels = __layer->GetDataInterface()->GetCurrentBatchLabel();
    assert(labels.size() == __a.size()); // make sure all samples have been processed

    //size_t batch_size = __a.size();

    // check
    auto dim = labels[0].Dimension(); // dim.first is neuron row number, dim.second is neuron collum number
    assert(__coord.i < dim.first);
    assert(dim.second == 1);
    //cout<<"neuron coord: "<<__coord<<endl;
    //assert(__coord.j == 0);

    //cout<<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"<<endl;
    //cout<<"sample index: "<<sample_index<<endl;
    //cout<<"neuron id: "<<__neuron_id<<endl;
    //Print();

    auto cost_func_type = __layer->GetCostFuncType(); // for $\partial C over \partial a$

    // label for current sample
    Matrix label_for_current_sample = labels[sample_index]; 
    //cout<<label_for_current_sample<<endl;
    // expected value for current sample current neuron
    float y_i = label_for_current_sample[__coord.i][0];
    //cout<<"---"<<y_i<<endl;

    // a value for current sample
    float a_i = __a[sample_index];
    //cout<<a_i<<endl;

    // sigma^\prime for current sample
    float sigma_prime_i = __sigmaPrime[sample_index];
    //cout<<"---"<<sigma_prime_i<<endl;

    // solve for dC/da, which is dependent on the type of cost function
    float dc_over_da = 0;
    if(cost_func_type == CostFuncType::cross_entropy)
    {
        dc_over_da = a_i  - y_i;
    }
    else 
    {
        // to be implemented
        dc_over_da = a_i - y_i; 
    }

    // delta for current sample current neuron
    float delta = dc_over_da * sigma_prime_i; 
    //cout<<"---"<<delta<<endl;

    // save delta for this batch this neuron
    __delta[sample_index] = delta;
    //cout<<"here..."<<endl;
}


void Neuron::UpdateDeltaFC(int sample_index)
{
    // back propagation delta for fully connected layer
    if(__sigmaPrime.size() <= 0) 
    {
	std::cout<<"Error: Neuron::UpdateDeltaFC() computing delta needs sigma^prime computed first."<<std::endl;
	std::cout<<"        "<<__delta.size()<<" deltas, "<<__sigmaPrime.size()<<" sigma^primes"<<std::endl;
	exit(0);
    }
    //cout<<"here..... Neuron::UpdateDeltaFC()"<<endl;

    Layer* __nextLayer = __layer->GetNextLayer();

    auto __deltaNext = __nextLayer->GetImagesDelta();
    //cout<<"delta images batch size: "<<__deltaNext.size()<<endl;
    Images image_delta_Next = __deltaNext[sample_index]; // get current sample delta
    std::vector<Matrix> &deltaNext = image_delta_Next.OutputImageFromKernel;
    if( deltaNext.size() != 1 ) 
    {
	std::cout<<"Error: Delta matrix dimension not match in FC layer"<<std::endl;
	exit(0);
    }
    Matrix delta = deltaNext[0];
    //cout<<delta<<endl;

    //cout<<"sample id: "<<sample_index<<endl;

    auto wv = __nextLayer->GetWeightMatrix();
    // 1) method 1
    Matrix w = Matrix::ConcatenateMatrixByI(*wv);

    auto w_dim = w.Dimension();
    if( w_dim.second < __coord.i ) 
    {
	std::cout<<"Error: weight matrix dimension not match in FC layer"<<std::endl;
	std::cout<<"Number of kernels: "<<wv->size()<<std::endl;
	std::cout<<"Neuron coord: "<<__coord<<std::endl;
	exit(0);
    }

    // back propogate delta
    w = w.Transpose();

    //cout<<w<<endl;
    auto dim = w.Dimension();
    w = w.GetSection(__coord.i, __coord.i+1, 0, dim.second);

    Matrix deltaCurrentLayer = w*delta;
    if(deltaCurrentLayer.Dimension().first != 1 || deltaCurrentLayer.Dimension().second != 1) 
    {
	std::cout<<"Error: back propagation delta, matrix dimension not match in FC layer."<<std::endl;
	exit(0);
    }

    // get sigma^\prime for current sample
    double s_prime = __sigmaPrime[sample_index];

    double v = deltaCurrentLayer[0][0];
    v = v*s_prime;

    __delta[sample_index] = v;

    //cout<<"here..... Neuron::UpdateDeltaFC()   end"<<endl;
}

void Neuron::UpdateDeltaCNN(int sample_index)
{
    // back propagate delta for cnn layer
    if(__sigmaPrime.size()<=0)
    {
	std::cout<<"Error: computing delta needs sigma^prime computed first."<<std::endl;
	exit(0);
    }
    double _sigma_prime = __sigmaPrime[sample_index];

    Layer* __nextLayer = __layer->GetNextLayer();

    auto deltaVecNext = __nextLayer->GetImagesDelta();
    auto weightVecNext = __nextLayer->GetWeightMatrix();

    Images &image_next_layer = deltaVecNext[sample_index]; // delta for current training sample
    std::vector<Matrix> & vec_delta_image = image_next_layer.OutputImageFromKernel;

    size_t C_next = vec_delta_image.size();
    if( C_next != weightVecNext->size() )
    {
	std::cout<<"Error: kernel number not match in cnn delta updating procedure."<<std::endl;
	exit(0);
    }

    // back propagation
    double tmp = 0;
    for(size_t d=0;d<C_next;d++)
    { // sum all kernels
	// get d^th delta and weight matrix of next layer
	Matrix delta = vec_delta_image[d];
	Matrix weight = (*weightVecNext)[d];

	auto weight_dimension = weight.Dimension();
	auto delta_dimension =  delta.Dimension();

	for(size_t p=0; p<weight_dimension.first;p++){
	    for(size_t q=0;q<weight_dimension.second;q++){
		double tmp_d = 0;
		int delta_index_p = __coord.i - p;
		int delta_index_q = __coord.j - q;
		if(delta_index_p >= 0 && delta_index_q >= 0 && delta_index_p<(int)delta_dimension.first && delta_index_q<(int)delta_dimension.second )
		    tmp_d = delta[delta_index_p][delta_index_q];
		double tmp_w = weight[p][q];
		tmp += tmp_d * tmp_w;
	    }
	}
    }

    tmp = tmp * _sigma_prime;

    __delta[sample_index] = tmp;
}

void Neuron::UpdateDeltaPooling(int sample_index)
{
    // back propagate delta for pooling layer
    // for pooling layer, we need to check which neuron the delta we should give to
    if(__a.size() <= 0) 
    {
	// error
	std::cout<<"Error: something wrong happend in error back propagation for pooling layer."<<std::endl;
	std::cout<<"       a matrix should be already calculated."<<std::endl;
	exit(0);
    }

    Layer* __nextLayer = __layer->GetNextLayer();

    auto deltaNext = __nextLayer->GetImagesDelta();
    Images & delta_image_for_current_sample = deltaNext[sample_index]; // delta image for current sample
    std::vector<Matrix> &deltaImages = delta_image_for_current_sample.OutputImageFromKernel; // matrix

    if(deltaImages.size() <= __coord.k) 
    {
	std::cout<<"Error: back propagate delta in pooling layer, matrix dimension not match."<<std::endl;
	exit(0);
    }

    // delta matrix from its next layer (l+1 layer)
    Matrix delta = deltaImages[__coord.k];
    // pooling matrix dimension
    auto dim = __w->Dimension();

    // (i, j) the pooling result coordinate in next layer
    size_t i = __coord.i%dim.first; 
    size_t j = __coord.j%dim.second;

    double error = delta[i][j];
    double a = __a[sample_index]; // a value

    if(__layer->GetPoolingMethod() == PoolingMethod::Max) 
    {
	// get all related input neurons, check if this neuron is the max one
	auto a_matrix_current_layer_vec = __layer->GetImagesA(); // batch images
	Images &image_current_sample = a_matrix_current_layer_vec[sample_index]; // current sample

	std::vector<Matrix> & matrix_current_sample = image_current_sample.OutputImageFromKernel; // matrix
	Matrix a_matrix = matrix_current_sample[__coord.k];

	double _tmp = 0;
	for(size_t ii = i*dim.first;ii<(i+1)*dim.first;ii++){
	    for(size_t jj = j*dim.second;jj<(j+1)*dim.second;jj++){
		if(_tmp < a_matrix[ii][jj]) _tmp = a_matrix[ii][jj];
	    }
	}
	if( a < _tmp ) error = 0; // this neuron is not the max one
    }
    else
    {
        // to be added
    }

    // for average pooling, give the same error to each neuron where it's related.
    __delta[sample_index] = error;
}

void Neuron::SetCoord(size_t i, size_t j, size_t k)
{
    __coord.i = i; __coord.j = j; __coord.k = k;
}

void Neuron::SetCoordI(size_t v)
{
    __coord.i = v;
}

void Neuron::SetCoordJ(size_t v)
{
    __coord.j = v;
}

void Neuron::SetCoordK(size_t v)
{
    __coord.k = v;
}

void Neuron::SetCoord(NeuronCoord c)
{
    __coord = c;
}

std::vector<double>& Neuron::GetAVector()
{
    return __a;
}

std::vector<double>& Neuron::GetDeltaVector()
{
    return __delta;
}

std::vector<double>& Neuron::GetZVector()
{
    return __z;
}

NeuronCoord Neuron::GetCoord()
{
    // get neuron coord
    return __coord;
}

void Neuron::ClearPreviousBatch()
{
    //__a.clear();           // obsolete
    //__delta.clear();       // obsolete
    //__z.clear();           // obsolete
    //__sigmaPrime.clear();  // obsolete

    if(__layer == nullptr)
    {
        std::cout<<"Error: Neuron::ClearPreviousBatch(): the layer info has not been set for this neuron."
	         <<endl;
        exit(0);
    }
    int batch_size = __layer->GetBatchSize();
    __a.resize(batch_size); // reset a
    __delta.resize(batch_size); // reset delta
    __z.resize(batch_size); // reset z
    __sigmaPrime.resize(batch_size);
}

void Neuron::Print()
{
    std::cout<<"--------------------------------------"<<std::endl;
    std::cout<<"neuron id: "<<__neuron_id<<std::endl;
    std::cout<<"active status: "<<__active_status<<std::endl;
    if(!__active_status) return;
    std::cout<<"w matrix: "<<std::endl;
    std::cout<<(*__w);
    std::cout<<"bias: "<<std::endl;
    std::cout<<(*__b);
    std::cout<<"Neruon Coord: "<<endl;
    std::cout<<__coord<<endl;
}
