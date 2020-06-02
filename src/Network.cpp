#include "Network.h"

#include "DataInterface.h"
#include "Layer.h"
#include "ConstructLayer.h"

#include <iostream>
#include <chrono>

using namespace std;

Network::Network()
{
    // place holder
}

Network::~Network()
{
    // place holder
}

void Network::Init()
{
    // construct all layers
    ConstructLayers();

    // set number of epochs
    __numberOfEpoch = 10;
}

void Network::ConstructLayers()
{
    // Network structure: {Image->Input->CNN->pooling->FC->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat", LayerDimension::_2D);

    // 3) input layer
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, Regularization::Undefined, 0);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // 4) middle layer 3 : cnn layer
    LayerParameterList p_list4(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(2, 2), 0.1, true, 0.5, Regularization::L2, 0.1);
    Layer *l2 = new ConstructLayer(p_list4);
    l2->SetPrevLayer(layer_input);
    l2->Init();
 

    // 4) middle layer 2 : cnn layer
    LayerParameterList p_list3(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(2, 2), 0.1, true, 0.5, Regularization::L2, 0.1);
    Layer *l1 = new ConstructLayer(p_list3);
    l1->SetPrevLayer(l2);
    l1->Init();
 
    // 4) middle layer 1
    LayerParameterList p_list1(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 20, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, true, 0.5, Regularization::L2, 0.1);
    Layer *l0 = new ConstructLayer(p_list1);
    l0->SetPrevLayer(l1);
    l0->Init();

    // 5) output layer
    LayerParameterList p_list2(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0., Regularization::L2, 0.1);
    Layer* layer_output = new ConstructLayer(p_list2);
    layer_output -> SetPrevLayer(l0);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    l2->SetNextLayer(l1);
    l1->SetNextLayer(l0);
    l0->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleLayers.push_back(l2); // must be pushed in order
    __middleLayers.push_back(l1);
    __middleLayers.push_back(l0);
    __dataInterface = data_interface;
}

void Network::Train()
{
    __numberOfEpoch = 1; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    for(int i=0;i<__numberOfEpoch;i++)
    {
        std::cout<<"[------]Number of epoch: "<<i<<"/"<<__numberOfEpoch<<endl;
        UpdateEpoch();
    }
}

void Network::UpdateEpoch()
{
    int numberofBatches = __dataInterface -> GetNumberOfBatches();
    cout<<"......Info: "<<numberofBatches<<" batches in this epoch"<<endl;
    numberofBatches = 5; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQqq

    // initializations for epoch
    for(auto &i: __middleLayers)
    {
	i->EpochInit();
	i->EnableDropOut();
    }
    __outputLayer->EpochInit();  // output layer no dropout

    for(int i=0;i<numberofBatches;i++)
    {
        cout<<"......... training for batch: "<<i<<"/"<<numberofBatches<<endl;
        UpdateBatch();
    }
}

void Network::UpdateBatch()
{
    // initializations for batch
    for(auto &i: __middleLayers)
        i->BatchInit();
    __outputLayer->BatchInit(); // input layer do not need init

    auto t1 = std::chrono::high_resolution_clock::now();
    ForwardPropagateForBatch();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    std::cout<<"forward propagation cost: "<<dt1.count()<<" milli seconds"<<endl;
    BackwardPropagateForBatch();
    auto t3 = std::chrono::high_resolution_clock::now();
    auto dt2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2);
    std::cout<<"backward propagation cost: "<<dt2.count()<<" milli seconds"<<endl;
    UpdateWeightsAndBiasForBatch();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto dt3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3);
    std::cout<<"update w&b cost: "<<dt3.count()<<" milli seconds"<<endl;
}

void Network::ForwardPropagateForBatch()
{
    // prepare new batch data and label in Datainterface class
    __dataInterface->GetNewBatchData();
    __dataInterface->GetNewBatchLabel();

    // fill data to input layer
    __inputLayer->FillBatchDataToInputLayerA();

    // get batch size
    int sample_size = __dataInterface->GetBatchSize();

    // train each sample for the middle layers
    for(int sample_index=0;sample_index<sample_size;sample_index++)
    {
	for(auto &i: __middleLayers)
	    i->ForwardPropagateForSample(sample_index);

	// compute cost in output layer for each sample
	__outputLayer-> ComputeCostInOutputLayerForCurrentSample(sample_index);

	// after finished training, clear used samples in input layer
	// 1)
	// this is necessary; b/c program aways fetch the last sample in "__imageA" from InputLayer
	//                ---- so one need to pop_back() the used sample
	// 2)
	// not needed anymore; b/c changed from "dynamic increase" to "fixed length"
	//                ---- now for eaching training, program fetch an indexed sample, not the last sample     
	//__inputLayer->ClearUsedSampleForInputLayer();
    }
}

void Network::BackwardPropagateForBatch() 
{
    // get batch size
    int sample_size = __dataInterface->GetBatchSize(); 
     
    // backward
    int NLayers = __middleLayers.size();

    for(int i=0;i<sample_size;i++)
    {
	// first do output layer
	__outputLayer -> BackwardPropagateForSample(i);

	// then do middle layers
	for(int nlayer=NLayers-1; nlayer>=0; nlayer--)
	    __middleLayers[nlayer]->BackwardPropagateForSample(i);
    }

    /// no need for input layer
}

void Network::UpdateWeightsAndBiasForBatch()
{
    // update w&b for  middle layers
    for(auto &i: __middleLayers)
	i->UpdateWeightsAndBias();

    // update w&b for output layer
    __outputLayer -> UpdateWeightsAndBias();
}

std::vector<Matrix> Network::Classify()
{
    std::vector<Matrix> res;
    return res;
}
