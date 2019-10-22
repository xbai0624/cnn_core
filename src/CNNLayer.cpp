#include "CNNLayer.h"
#include "Neuron.h"
#include <iostream>

CNNLayer::CNNLayer(){
    // place holder
}

CNNLayer::~CNNLayer(){
    // place holder
}

void CNNLayer::Init(){
}

void CNNLayer::EpochInit(){
}

void CNNLayer::ProcessBatch(){
}

void CNNLayer::PostProcessBatch(){
}

void CNNLayer::BatchInit(){
}

void CNNLayer::ProcessSample(){
}

void CNNLayer::InitNeurons(){
}

std::vector<Matrix>& CNNLayer::GetValueMatrixA(){
    return __valueMatrixA;
}

std::vector<Matrix>& CNNLayer::GetValueMatrixZ(){
    return __valueMatrixZ;
}

std::vector<Matrix>& CNNLayer::GetValueMatrixDelta(){
    return __valueMatrixDelta;
}

void CNNLayer::UpdateActiveWeightsAndBias(){
    // update new active weights and bias
    // cnn layer
    // needs __activeFlag information
    __weightMatrixActive.clear();
    __biasVectorActive.clear();

    // weight matrix
    if(__weightMatrix.size() != __activeFlag.size()){
	std::cout<<"Error: update active weight for cnn needs active flag finished."<<std::endl;
	exit(0);
    }
    for(size_t k=0;k<__weightMatrix.size();k++) {
	Matrix _K = __weightMatrix[k];
	auto flag = __activeFlag[k];
	auto dim = _K.Dimension();

	if(flag.size() != dim.first || flag[0].size() != dim.second) {
	    std::cout<<"Error: updae cnn active weight matrix, dimension not match."<<std::endl;
	    exit(0);
	}

	for(size_t i=0;i<dim.first;i++){
	    for(size_t j=0;j<dim.second;j++){
		if(! flag[i][j] ) 
		    _K[i][j] = 0; // set weight matrix element to 0, in other words, drop out connection
	    }
	}

	__weightMatrixActive.push_back(_K);
    }

    // bias vector wont change
    __biasVectorActive = __biasVector;
}

void CNNLayer::AssignWeightsAndBiasToNeurons(){
    // only for active neurons
    // cnn layer dropout will change kernel dimension
    // MLP layer dropout will change weight matrix dimension
    //
    // Remember the code design, each neuron (both MLP and CNN type) will
    // have a matrix-form weight and scalar bias

    for(size_t k=0;k<__neuronDim.k;k++){
	for(size_t i=0;i<__neuronDim.i;i++){
	    for(size_t j=0;j<__neuronDim.j;j++){
		if(__neurons[k][i][j] -> IsActive()){
		    // active neuron
		    auto coord = __neurons[k][i][j] -> GetCoord();
		    if(__type == LayerType::cnn) {
			// cnn and pooling
			// note that we dont dropout kernel, dropout occurs inside kernel, we drop out connections
			__neurons[k][i][j] -> PassWeightPointer(&__weightMatrixActive[coord.k]);
			__neurons[k][i][j] -> PassBiasPointer(&__biasVectorActive[coord.k]);
		    } 
		}
	    }
	}
    }
}

void CNNLayer::DropOut(){
}

void CNNLayer::TransferValueFromActiveToOriginal_WB(){
    // after active weights and bias are updated,
    // transfer this change to original weights and bias

    if(__weightMatrix.size() != __weightMatrixActive.size()){
        std::cout<<"Error: cnn transfer weight, dimension not match."<<std::endl;
	exit(0);
    }
    if(__weightMatrix.size() != __activeFlag.size() ){
        std::cout<<"Error: cnn transfer flag."<<std::endl;
	exit(0);
    }

    size_t nK = __weightMatrix.size();
    auto dim = __weightMatrix[0].Dimension();
    for(size_t k=0;k<nK;k++){
        for(size_t i=0;i<dim.first;i++){
	    for(size_t j=0;j<dim.second;j++){
	        if(!__activeFlag[k][i][j]) continue;
		__weightMatrix[k][i][j] = __weightMatrixActive[k][i][j];
	    }
	}

	// bias
	__biasVector[k] = __biasVectorActive[k];
    }
}

void CNNLayer::UpdateImageForCurrentTrainingSample(){
    // re-organize value from neurons to matrix form,
    // only for current training sample

    ClearImage(); // clear previous training sample
    GetActiveNeuronDimension(); // get active neuron dimension for current training

    size_t i=0, j=0, k=0; // for active neuron coord
    for(auto &_kernel: __neurons){
	k++; 
	j=0;

	Matrix _m_A(__activeNeuronDim.i, __activeNeuronDim.j);
	Matrix _m_Z(__activeNeuronDim.i, __activeNeuronDim.j);
	Matrix _m_Delta(__activeNeuronDim.i, __activeNeuronDim.j);
	size_t index = 0;

	for(auto &_j: _kernel){
	    j++; 
	    i=0;
	    for(auto &_i: _j){
		i++;
		if(_i->IsActive()){
		    _m_A.FillElementByRow(index, _i->GetAVector().back());
		    _m_Z.FillElementByRow(index, _i->GetZVector().back());
		    _m_Delta.FillElementByRow(index, _i->GetDeltaVector().back());
		    index++;
		}
	    }
	}

	if(index != __activeNeuronDim.i*__activeNeuronDim.j ){
	    std::cout<<"Error: dimension not match in extracting neuron image."<<std::endl;
	    exit(0);
	}
	__valueMatrixA.push_back(_m_A);
	__valueMatrixZ.push_back(_m_Z);
	__valueMatrixDelta.push_back(_m_Delta);
    }
}

void CNNLayer::ClearImage(){
    // clear image matrix
    __valueMatrixA.clear();
    __valueMatrixZ.clear();
    __valueMatrixDelta.clear();
    __activeFlag.clear();
}

NeuronCoord CNNLayer::GetActiveNeuronDimension(){
    // get active neurons in x-y plane
    if(__neurons.size() <=0) {
	std::cout<<"Error: No active neurons in layer"<<std::endl;
	return NeuronCoord();
    }

    // NOTE: for CNN dropout, We will not drop out whole kernels, otherwise this 
    // is incorrect
    size_t k_dim = __neurons.size();

    size_t x_dim=0, y_dim=0;
    size_t y_size = __neurons[0].size();
    size_t x_size = __neurons[0][0].size();

    for(size_t i=0;i<x_size;i++){
	if(__neurons[0][0][i]->IsActive())
	    x_dim++;
    }
    for(size_t i=0;i<y_size;i++){
	for(size_t j=0;j<x_size;j++){
	    if(__neurons[0][i][j]->IsActive()){
		y_dim++;
		break;
	    }
	}
    }
    __activeNeuronDim.i = x_dim;
    __activeNeuronDim.j = y_dim;
    __activeNeuronDim.k = k_dim;

    return __activeNeuronDim;
}

void CNNLayer::SetCNNStride(int s){
    // set convolution stride
    __cnnStride = s;
}

void CNNLayer::SetDropOutFactor(float d){
    // set up drop out factor, it works like this:
    // for each neuron, we will generate a random number
    // if this random number <d, then set the neuron 
    // inactive
    __dropOut = d;
}

int CNNLayer::GetCNNStride(){
    return __cnnStride;
}

std::vector<Matrix>* CNNLayer::GetWeightMatrix(){
}

std::vector<double>* CNNLayer::GetBiasVector(){
}

LayerType CNNLayer::GetType(){
}

float CNNLayer::GetDropOutFactor(){
    return __dropOut;
}

std::vector<std::vector<std::vector<bool>>>& CNNLayer::GetActiveFlag(){
    // get active flags
    return __activeFlag;
}
