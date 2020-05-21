#include <cstdlib>
#include <cmath>
#include <iostream>

#include "Layer.h"
#include "Neuron.h"
#include "Matrix.h"

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
    // after updateing weights and biase, 
    // namely after one training (one sample or one batch depends on user)
    // reset neurons for next computation
    __a.clear(); // reset a
    __delta.clear(); // reset delta
    __z.clear(); // reset z

    //__wGradient.clear(); // reset weight gradient
    //__bGradient.clear(); // reset bias gradient
}

void Neuron::SetLayer(Layer* l)
{
    // set layer that this neuron currently belongs to
    __layer = l;
}

void Neuron::SetPreviousLayer(Layer* l)
{
    // set neuron's prevous layer
    __previousLayer = l;
}

void Neuron::SetNextLayer(Layer* l)
{
    // set neurons's next layer
    __nextLayer = l;
}

void Neuron::UpdateZ()
{
    // update z for current layer
    if(__layer->GetType() == LayerType::fullyConnected)
    {
        std::cout<<"fully connected layer update z"<<std::endl;
	UpdateZFC();
    } 
    else if(__layer->GetType() == LayerType::cnn)
    {
        std::cout<<"cnn layer update z"<<std::endl;
	UpdateZCNN();
    } 
    else if(__layer->GetType() == LayerType::pooling)
    {
        std::cout<<"pooling layer update z"<<std::endl;
	UpdateZPooling();
    } 
    else 
    {
	std::cout<<"Error: unsupported layer type."<<std::endl;
	exit(0);
    }
}

void Neuron::UpdateZFC()
{
    // for fully connected layer, matrix reform are done by Layer class
    // currently layer and its previous layer are fully connected
    // weight matrix dimension will be: (1, M)
    auto _t = __previousLayer -> GetImagesA();
    if(_t.size() < 1) 
    {
	std::cout<<"Error: previous layer has not A image."<<std::endl;
	exit(0);
    }
    Images &images = _t.back(); // get images for current sample
    if(images.OutputImageFromKernel.size() != 1) 
    {
	std::cout<<"Eroor: layer type not match, expecting FC layer."<<std::endl;
	exit(0);
    }

    Matrix &image = images.OutputImageFromKernel[0];

    Matrix res = (*__w) * image;
    auto dim = res.Dimension();
    if(dim.first != 1 || dim.second != 1) 
    {
	std::cout<<"Error: wrong dimension, expecting 1D matrix."<<std::endl;
    }
    double z = res[0][0];
    z = z + (*__b)[0][0];
    __z.push_back(z);
}

void Neuron::UpdateZCNN()
{
    // cnn layer
    // every single output image needs input from all input images
    auto inputImage = __previousLayer->GetImagesA();
    auto w_dim = __w->Dimension();
    int stride = __layer->GetCNNStride();

    size_t i_start = __coord.i*stride;
    size_t j_start = __coord.j*stride;
    size_t i_end = i_start + w_dim.first;
    size_t j_end = j_start + w_dim.second;

    auto &current_sample_image = (inputImage.back()).OutputImageFromKernel;
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
    __z.push_back(z);
}

void Neuron::UpdateZPooling()
{
    // pooling layer
    // should be with cnn layer, just kernel matrix all elements=1, bias = 0;
    auto inputImage = __previousLayer->GetImagesA();
    if(inputImage.back().OutputImageFromKernel.size() < __coord.k)
    {
	// output image for current sample
	// for pooling layer, number of kernels (previous layer)  = number of kernels (current layer)
	std::cout<<"Error: pooling operation matrix dimension not match"<<std::endl;
	exit(0);
    }
    //Matrix image = inputImage[__coord.k];
    Images image = inputImage.back(); // images for current training sample
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

    __z.push_back(z);
}

void Neuron::UpdateA()
{
    // update a for current training sample
    if(__a.size() != __z.size()-1) 
    {
	std::cout<<"Error: computing a needs z computed first."<<std::endl;
	exit(0);
    }
    double v = __z.back();
    double a;
    if(__funcType == ActuationFuncType::Sigmoid)
	a = __sigmoid(v);
    else if(__funcType == ActuationFuncType::Tanh)
	a = __tanh(v);
    else if(__funcType == ActuationFuncType::Relu)
	a = __relu(v);
    else
	std::cout<<"Error: unsupported actuation function type."<<std::endl;
    __a.push_back(a);
}

void Neuron::UpdateSigmaPrime()
{
    // update sigma^prime
    if(__sigmaPrime.size() != __z.size()-1) 
    {
	std::cout<<"Error: computing sigma^prime needs z computed first."<<std::endl;
	exit(0);
    }
    if(__sigmaPrime.size() != __a.size()-1) 
    {
	std::cout<<"Error: computing sigma^prime needs a computed first."<<std::endl;
	exit(0);
    }

    double a = __a.back();
    double z = __z.back();
    double sigma_prime;

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

    __sigmaPrime.push_back(sigma_prime);
}

void Neuron::UpdateDelta()
{
    // update delta for current layer
    if(__layer->GetType() == LayerType::fullyConnected)
    {
	UpdateDeltaFC();
    } 
    else if(__layer->GetType() == LayerType::cnn)
    {
	UpdateDeltaCNN();
    } 
    else if(__layer->GetType() == LayerType::pooling)
    {
	UpdateDeltaPooling();
    } 
    else 
    {
	std::cout<<"Error: unsupported layer type."<<std::endl;
	exit(0);
    }
}

void Neuron::UpdateDeltaFC()
{
    // back propagation delta for fully connected layer
    if(__delta.size() != __sigmaPrime.size() - 1) 
    {
	std::cout<<"Error: computing delta needs sigma^prime computed first."<<std::endl;
	exit(0);
    }

    auto __deltaNext = __nextLayer->GetImagesDelta();
    Images image_delta_Next = __deltaNext.back(); // get current sample delta
    std::vector<Matrix> &deltaNext = image_delta_Next.OutputImageFromKernel;
    if( deltaNext.size() != 1 ) 
    {
	std::cout<<"Error: Delta matrix dimension not match in FC layer"<<std::endl;
	exit(0);
    }
    Matrix delta = deltaNext[0];

    auto wv = __nextLayer->GetWeightMatrix();
    if( wv->size() != 1 ) 
    {
	std::cout<<"Error: weight matrix dimension not match in FC layer"<<std::endl;
	exit(0);
    }
    Matrix w = (*wv)[0];
    // back propogate delta
    w = w.Transpose();
    auto dim = w.Dimension();
    w = w.GetSection(__coord.i, __coord.i+1, 0, dim.second);

    Matrix deltaCurrentLayer = w*delta;
    if(deltaCurrentLayer.Dimension().first != 1 || deltaCurrentLayer.Dimension().second != 1) 
    {
	std::cout<<"Error: back propagation delta, matrix dimension not match in FC layer."<<std::endl;
	exit(0);
    }

    double s_prime = __sigmaPrime.back();
    double v = deltaCurrentLayer[0][0];
    v = v*s_prime;
    __delta.push_back(v);
}

void Neuron::UpdateDeltaCNN()
{
    // back propagate delta for cnn layer
    if(__delta.size() != __sigmaPrime.size()-1)
    {
	std::cout<<"Error: computing delta needs sigma^prime computed first."<<std::endl;
	exit(0);
    }
    double _sigma_prime = __sigmaPrime.back();

    auto deltaVecNext = __nextLayer->GetImagesDelta();
    auto weightVecNext = __nextLayer->GetWeightMatrix();

    Images &image_next_layer = deltaVecNext.back(); // delta for current training sample
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
    __delta.push_back(tmp);
}

void Neuron::UpdateDeltaPooling()
{
    // back propagate delta for pooling layer
    // for pooling layer, we need to check which neuron the delta we should give to
    if(__delta.size() != __a.size() - 1) 
    {
	// error
	std::cout<<"Error: something wrong happend in error back propagation for pooling layer."<<std::endl;
	std::cout<<"       a matrix should be already calculated."<<std::endl;
	exit(0);
    }

    auto deltaNext = __nextLayer->GetImagesDelta();
    Images & delta_image_for_current_sample = deltaNext.back(); // delta image for current sample
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
    double a = __a.back(); // a value

    if(__layer->GetPoolingMethod() == PoolingMethod::Max) 
    {
	// get all related input neurons, check if this neuron is the max one
	auto a_matrix_current_layer_vec = __layer->GetImagesA(); // batch images
	Images &image_current_sample = a_matrix_current_layer_vec.back(); // current sample
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
    // for average pooling, give the same error to each neuron where it's related.
    __delta.push_back(error);
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
}
