#include "Network.h"

#include "DataInterface.h"
#include "Layer.h"
#include "ConstructLayer.h"

#include <iostream>

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
    // 1) Data interface, this is a tool class, for data prepare
    //DataInterface data_interface;
    DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat");
    //auto a = data_interface->GetNewBatchData();
    //for(auto &i: a) cout<<i<<endl;
    //auto b = data_interface.GetNewBatchLabel();

    // 2) declare all needed layers 
    Layer *layer_input = nullptr, *l0 = nullptr, *layer_output = nullptr;

    // 3) input layer
    layer_input = new ConstructLayer(LayerType::input);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Init rely on data_interface
    layer_input -> PassDataInterface(data_interface);
    layer_input->Init();
    cout<<"input layer test finished..."<<endl;

    // 4) middle layer
    l0 = new ConstructLayer(LayerType::fullyConnected, 20);
    l0->SetPrevLayer(layer_input);
    l0->SetNextLayer(layer_output);
    l0->Init();
    l0->EpochInit();
    l0->EnableDropOut();
    l0->BatchInit();
    //l0->Print();
    l0->ForwardPropagateForSample();
    auto images = l0->GetImagesA();
    cout<<" number of images: "<<images.size()<<endl;

    // 5) output layer
    // test output layer
    layer_output = new ConstructLayer(LayerType::output, 10); // output layer must be a fully connected layer
    layer_output -> SetPrevLayer(l0);
    layer_output -> Init();
    layer_output -> EpochInit(); // output layer no dropout
    layer_output -> BatchInit();
    //layer_output->Print();
    layer_output -> ComputeCostInOutputLayerForCurrentSample(); // output layer needs to compute cost function

    // 6) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleLayers.push_back(l0);
    __dataInterface = data_interface;
}

void Network::Train()
{
    for(int i=0;i<__numberOfEpoch;i++)
    {
        UpdateEpoch();
    }
}

void Network::UpdateEpoch()
{
    int numberofBatches = __dataInterface -> GetNumberOfBatches();

    for(int i=0;i<numberofBatches;i++)
    {
        UpdateBatch();
    }
}

void Network::UpdateBatch()
{
    ForwardPropagateForBatch();
    BackwardPropagateForBatch();
    UpdateWeightsAndBias();
}

void Network::ForwardPropagateForBatch()
{
    // prepare new batch data in Datainterface class
    __dataInterface->GetNewBatchData();
    __dataInterface->GetNewBatchLabel();

    // get batch size
    int sample_size = __dataInterface->GetBatchSize();

    // train each sample for the middle layers
    for(int i=0;i<sample_size;i++)
    {
	for(auto &i: __middleLayers)
	    i->ForwardPropagateForSample();
    }

    // compute cost in output layer for each sample
    __outputLayer-> ComputeCostInOutputLayerForCurrentSample();
}

void Network::BackwardPropagateForBatch() 
{
    // backward
    int NLayers = __middleLayers.size();
    for(int i=NLayers-1; i>=0;i--)
	__middleLayers[i]->BackwardPropagateForBatch();
}

void Network::UpdateWeightsAndBias()
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
