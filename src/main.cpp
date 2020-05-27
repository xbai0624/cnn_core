#include <iostream>
#include "Tools.h"
//#include "Layer.h"
//#include "ConstructLayer.h"
///#include "DataInterface.h"

#include "Network.h"

using namespace std;

int main(int argc, char* argv[])
{
/*
    // test datainterface
    //DataInterface data_interface;
    DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat");
    //auto a = data_interface->GetNewBatchData();
    //for(auto &i: a) cout<<i<<endl;
    //auto b = data_interface.GetNewBatchLabel();

    // test input layer
    Layer *layer_input;
    layer_input = new ConstructLayer(LayerType::input);
    layer_input -> PassDataInterface(data_interface); // NOTE: an data_interface class pointer must be passed to input layer before calling input_layer->Init() function
                                                      //       because Init rely on data_interface
    layer_input->Init();
    cout<<"input layer test finished..."<<endl;

    // test network
    Layer *l0 = nullptr, *l1 = nullptr, *l2 = nullptr, *layer_output = nullptr;

    // test middle layers
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

    // test output layer
    layer_output = new ConstructLayer(LayerType::output, 10); // output layer must be a fully connected layer
    layer_output -> SetPrevLayer(l0);
    layer_output -> Init();
    layer_output -> EpochInit(); // output layer no dropout
    layer_output -> BatchInit();
    //layer_output->Print();
    layer_output -> ComputeCostInOutputLayerForCurrentSample(); // output layer needs to compute cost function
*/

    Network *net_work = new Network();
    net_work->Init();

    cout<<"MAIN TEST SUCCESS!!!"<<endl;
    return 0;
}
