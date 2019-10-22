#include <iostream>
#include <numeric> // std::iota

#include "FCLayer.h"
#include "Matrix.h"
#include "Neuron.h"
#include "Tools.h"

FCLayer::FCLayer(){
    // place holder
}

FCLayer::~FCLayer(){
    // place holder
}

void FCLayer::Init(){
}

void FCLayer::EpochInit(){
}

void FCLayer::ProcessBatch(){
}

void FCLayer::PostProcessBatch(){
}

void FCLayer::BatchInit(){
}

void FCLayer::ProcessSample(){
}

void FCLayer::InitNeurons(){
}

std::vector<Images>& FCLayer::GetImagesA(){
}

std::vector<Images>& FCLayer::GetImagesZ(){
}

std::vector<Images>& FCLayer::GetImagesDelta(){
}

void FCLayer::UpdateCoordsForActiveNeuronFC(){
}

void FCLayer::UpdateActiveWeightsAndBias(){
}

void FCLayer::AssignWeightsAndBiasToNeurons(){
}

void FCLayer::DropOut(){
}

void FCLayer::__UpdateActiveFlagFC(){
}

void FCLayer::TransferValueFromActiveToOriginal_WB(){
}

void FCLayer::UpdateImageForCurrentTrainingSample(){
}

void FCLayer::ClearImage(){
}

NeuronCoord FCLayer::GetActiveNeuronDimension(){
}

void FCLayer::SetDropOutFactor(float f){
}

std::vector<Matrix>* FCLayer::GetWeightMatrix(){
}

std::vector<double>* FCLayer::GetBiasVector(){
}

LayerType FCLayer::GetType(){
}

float FCLayer::GetDropOutFactor(){
}

std::vector<std::vector<std::vector<bool>>>& FCLayer::GetActiveFlag(){
}
