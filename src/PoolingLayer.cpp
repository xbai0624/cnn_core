#include "PoolingLayer.h"

PoolingLayer::PoolingLayer() {
    // place holder
}

PoolingLayer::~PoolingLayer() {
    // place holder
}

void PoolingLayer::Init(){
}

void PoolingLayer::EpochInit(){
}

void PoolingLayer::ProcessBatch(){
}

void PoolingLayer::PostProcessBatch(){
}

void PoolingLayer::BatchInit(){
}

void PoolingLayer::ProcessSample(){
}

void PoolingLayer::InitNeurons(){
}

std::vector<Matrix>& PoolingLayer::GetValueMatrixA(){
}

std::vector<Matrix>& PoolingLayer::GetValueMatrixZ(){
}

std::vector<Matrix>& PoolingLayer::GetValueMatrixDelta(){
}

void PoolingLayer::UpdateActiveWeightsAndBias(){
}

void PoolingLayer::AssignWeightsAndBiasToNeurons(){
}

void PoolingLayer::DropOut(){
}

void PoolingLayer::TransferValueFromActiveToOriginal_WB(){
}

void PoolingLayer::UpdateImageForCurrentTrainingSample(){
}

void PoolingLayer::ClearImage(){
}

NeuronCoord PoolingLayer::GetActiveNeuronDimension(){
}

void PoolingLayer::SetPoolingMethod(PoolingMethod m){
}

void PoolingLayer::SetDropOutFactor(float f){
}

PoolingMethod & PoolingLayer::GetPoolingMethod(){
}

std::vector<Matrix>* PoolingLayer::GetWeightMatrix(){
}

std::vector<double>* PoolingLayer::GetBiasVector(){
}

LayerType PoolingLayer::GetType(){
}

float PoolingLayer::GetDropOutFactor(){
}

std::vector<std::vector<std::vector<bool>>>& PoolingLayer::GetActiveFlag(){
}
