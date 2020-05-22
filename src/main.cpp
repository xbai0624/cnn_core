#include <iostream>
#include "Tools.h"
#include "Layer.h"
#include "ConstructLayer.h"
#include "DataInterface.h"

using namespace std;

int main(int argc, char* argv[])
{
    // test datainterface
    DataInterface data_interface;
    data_interface.test();


    // test input layer
    Layer *layer_input;
    layer_input = new ConstructLayer(LayerType::input);
    layer_input->Init(); // check (need this one)
    cout<<"input layer test finished..."<<endl;


    Layer *l0, *l1, *l2;

    // test network

    // test layers
    l0 = new ConstructLayer(LayerType::fullyConnected, 20);
    l0->SetPrevLayer(layer_input);
    l0->Init();
    l0->EpochInit();
    l0->EnableDropOut();
    l0->BatchInit();
    //l0->Print();
    l0->ForwardPropagate();
    auto images = l0->GetImagesA();
    cout<<"number of images: "<<images.size()<<endl;
/*
    l1 = new ConstructLayer(LayerType::fullyConnected, 10);
    l1->Connect(l0, l2);
    l1->Init();
    l1->EnableDropOut();
    l1->BatchInit();
    l1->Print();

    l2 = new ConstructLayer(LayerType::cnn, 5, std::pair<size_t, size_t>(6, 6));
    l2->Connect(l1);
    l2->Init();
    l2->EnableDropOut();
    l2->BatchInit();
    l2->Print();
*/

    cout<<"MAIN TEST SUCCESS!!!"<<endl;
    return 0;
}
