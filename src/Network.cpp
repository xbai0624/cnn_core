#include "Network.h"

#include "DataInterface.h"
#include "Layer.h"
#include "ConstructLayer.h"

#include <iostream>
#include <chrono>
#include <thread>

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
    // Network structure: {Image->Input->CNN->pooling->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat", LayerDimension::_2D);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // 4) middle layer 3 : cnn layer ID=1
    LayerParameterList p_list4(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(3, 3), 0.1, false, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu);
    Layer *l2 = new ConstructLayer(p_list4);
    l2->SetPrevLayer(layer_input);
    l2->Init();
  
    // 4) middle layer 2 : pooling layer ID=2
    LayerParameterList p_list7(LayerType::pooling, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(2, 2), 0., false, 0., Regularization::Undefined, 0., ActuationFuncType::Relu);
    Layer *l5 = new ConstructLayer(p_list7);
    l5->SetPrevLayer(l2);
    l5->Init();
 
    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list1(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 4, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu);
    Layer *l0 = new ConstructLayer(p_list1);
    l0->SetPrevLayer(l5);
    l0->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list2(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax);
    Layer* layer_output = new ConstructLayer(p_list2);
    layer_output -> SetPrevLayer(l0);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l2->SetNextLayer(l5);
    l5->SetNextLayer(l0);
    l0->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleAndOutputLayers.push_back(l2); // must be pushed in order
    __middleAndOutputLayers.push_back(l5); // must be pushed in order
    __middleAndOutputLayers.push_back(l0);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;
}

/*
void Network::ConstructLayers() // test purelly fully connected
{
    // Network structure: {Image->Input->FC->FC->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat", LayerDimension::_1D);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_1D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list3(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 10, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0.5, Regularization::L2, 0.1, ActuationFuncType::Sigmoid);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(layer_input);
    l3->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l3);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    //l1->SetNextLayer(l2);
    //l2->SetNextLayer(l3);
    l3->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    //__middleAndOutputLayers.push_back(l1); // must be pushed in order
    //__middleAndOutputLayers.push_back(l2); // must be pushed in order
    __middleAndOutputLayers.push_back(l3);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;
}
*/

/*
void Network::ConstructLayers() // test fully connected + cnn
{
    // Network structure: {Image->Input->FC->FC->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat", LayerDimension::_2D);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // 4) middle layer 1 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 2, 
	    std::pair<size_t, size_t>(2, 2), 0.1, false, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list6(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 10, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu);
    Layer *l6 = new ConstructLayer(p_list6);
    l6->SetPrevLayer(l1);
    l6->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l6);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l6);
    //l2->SetNextLayer(l3);
    l6->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleAndOutputLayers.push_back(l1); // must be pushed in order
    //__middleAndOutputLayers.push_back(l2); // must be pushed in order
    __middleAndOutputLayers.push_back(l6);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;
}
*/

void Network::Train()
{
    __numberOfEpoch = 50; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    for(int i=0;i<__numberOfEpoch;i++)
    {
        std::cout<<"[------]Number of epoch: "<<i<<"/"<<__numberOfEpoch<<endl;
        UpdateEpoch();
    }

    // check accuracy track of training
    cout<<"accuracy: "<<endl;
    std::vector<float> & accuracy = __outputLayer->GetAccuracyForBatches();
    for(auto &i: accuracy)
        cout<<i<<",   "<<endl;

    // check cost track of training
    cout<<"cost: "<<endl;
    std::vector<float> &cost = __outputLayer->GetCostForBatches();
    for(auto &i: cost)
        cout<<i<<", "<<endl;
}

void Network::UpdateEpoch()
{
    int numberofBatches = __dataInterface -> GetNumberOfBatches();
    cout<<"......Info: "<<numberofBatches<<" batches in this epoch"<<endl;
    numberofBatches = 1; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQqq

    __dataInterface->Reset();

    // initializations for epoch
    for(auto &i: __middleAndOutputLayers)
    {
	i->EpochInit();
    }

    for(int i=0;i<numberofBatches;i++)
    {
        cout<<"......... training for batch: "<<i<<"/"<<numberofBatches<<endl;
        UpdateBatch();
    }
}

void Network::UpdateBatch()
{
    // initializations for batch
    for(auto &i: __middleAndOutputLayers)
        i->BatchInit();

    auto t1 = std::chrono::high_resolution_clock::now();
    ForwardPropagateForBatch();
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    //std::cout<<"forward propagation cost: "<<dt1.count()<<" milliseconds"<<endl;

    BackwardPropagateForBatch();
    //auto t3 = std::chrono::high_resolution_clock::now();
    //auto dt2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2);
    //std::cout<<"backward propagation cost: "<<dt2.count()<<" milliseconds"<<endl;

    UpdateWeightsAndBiasForBatch();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto dt3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t1);
    std::cout<<"        batch training cost: "<<dt3.count()<<" milliseconds"<<endl;

    // save training accuracy for batch
    __outputLayer -> SaveAccuracyAndCostForBatch();
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

    // output and middle layers
    for(int sample_index=0;sample_index<sample_size;sample_index++)
    {
	for(auto &i: __middleAndOutputLayers)
	{
	    i->ForwardPropagateForSample(sample_index);
	}
    }
}

void Network::BackwardPropagateForBatch() 
{
    // get batch size
    int sample_size = __dataInterface->GetBatchSize(); 
     
    // backward
    int NLayers = __middleAndOutputLayers.size();

    for(int i=0;i<sample_size;i++)
    {
	// output and middle layers
	for(int nlayer=NLayers-1; nlayer>=0; nlayer--)
	    __middleAndOutputLayers[nlayer]->BackwardPropagateForSample(i);
    }

    /// no need for input layer
}

void Network::UpdateWeightsAndBiasForBatch()
{
    // update w&b for output and  middle layers
    for(auto &i: __middleAndOutputLayers)
	i->UpdateWeightsAndBias();
}

std::vector<Matrix> Network::Classify()
{
    std::vector<Matrix> res;
    return res;
}
