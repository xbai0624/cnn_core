#include <iostream>
#include <numeric> // std::iota

#include "ConstructLayer.h"
#include "Matrix.h"
#include "Neuron.h"
#include "Tools.h"

static float SGN(float x)
{
    if( x == 0) return 0;
    return x>0?1:-1;
}


ConstructLayer::ConstructLayer()
{
    // place holder
}

ConstructLayer::~ConstructLayer()
{
    // place holder
}

void ConstructLayer::Init()
{
}

void ConstructLayer::EpochInit()
{
}

void ConstructLayer::ProcessBatch()
{
}

void ConstructLayer::PostProcessBatch()
{
}

void ConstructLayer::BatchInit()
{
}

void ConstructLayer::ProcessSample()
{
}

void ConstructLayer::InitNeurons()
{
}

std::vector<Images>& ConstructLayer::GetImagesA()
{
}

std::vector<Images>& ConstructLayer::GetImagesZ()
{
}

std::vector<Images>& ConstructLayer::GetImagesDelta()
{
}

void ConstructLayer::UpdateCoordsForActiveNeuronFC()
{
}

void ConstructLayer::UpdateActiveWeightsAndBias()
{
}

void ConstructLayer::AssignWeightsAndBiasToNeurons()
{
}

void ConstructLayer::DropOut()
{
}

void ConstructLayer::__UpdateActiveFlagFC()
{
}

void ConstructLayer::TransferValueFromActiveToOriginal_WB()
{
}

void ConstructLayer::UpdateImageForCurrentTrainingSample()
{
}

void ConstructLayer::ClearImage()
{
}

NeuronCoord ConstructLayer::GetActiveNeuronDimension()
{
}

void ConstructLayer::SetDropOutFactor(float f)
{
}

std::vector<Matrix>* ConstructLayer::GetWeightMatrix()
{
}

std::vector<double>* ConstructLayer::GetBiasVector()
{
}

LayerType ConstructLayer::GetType()
{
}

float ConstructLayer::GetDropOutFactor()
{
}

std::vector<std::vector<std::vector<bool>>>& ConstructLayer::GetActiveFlag()
{
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
    if(a_images.back().SampleOutputImage.size() != 1 ) {
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
        Matrix a_matrix = a_images[i].SampleOutputImage[0]; // 'a' image from previous layer
        Matrix d_matrix = d_images[i].SampleOutputImage[0]; // 'd' image from current layer

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
        tmp_w_gradients.SampleOutputImage.push_back(dw);
        __wGradient.push_back(tmp_w_gradients); // push weight gradient for current training sample

        Images tmp_b_gradients;
        tmp_b_gradients.SampleOutputImage.push_back(d_matrix);
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
    if( __wGradient[0].SampleOutputImage.size() != 1 || __bGradient[0].SampleOutputImage.size()!=1) 
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
	dw  = dw + __wGradient[i].SampleOutputImage[0];
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
	db = db + __bGradient[i].SampleOutputImage[0];
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
        std::vector<Matrix> &a_matrix = a_image.SampleOutputImage;
        Images &d_image = deltaVec[nbatch];
        std::vector<Matrix> &d_matrix = d_image.SampleOutputImage;
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
            tmp_w_gradients.SampleOutputImage.push_back(dw); // push weight gradient for current training sample

            // update bias gradient
            Matrix db(1, 1);
            auto dim = delta.Dimension();
            double b_gradient = delta.SumInSection(0, dim.first, 0, dim.second);
            db[0][0] = b_gradient;
            tmp_b_gradients.SampleOutputImage.push_back(db);
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
            dw  = dw + __wGradient[i].SampleOutputImage[k];
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
            db = db + __bGradient[i].SampleOutputImage[k];
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
