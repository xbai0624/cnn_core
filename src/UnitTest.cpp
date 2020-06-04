#include "UnitTest.h"

#include <iostream>

#include "DataInterface.h"
#include "Layer.h" // test layer
#include "ConstructLayer.h"

using namespace std;

UnitTest::UnitTest()
{
    // reserved
}

UnitTest::~UnitTest()
{
    // reserved
}

void UnitTest::Test()
{
    //TestImagesStruct();
    //TestMatrix();

    TestDNN();
}


void UnitTest::TestImagesStruct()
{
    // test Images struct in Layer.h file

    // 1) empty test
    Images __images;
    //Images v_image0 = __images.Vectorization();

    // 2) vectorization functionality test
    cout<<"vectorization test"<<endl;
    for(int i=0;i<4;i++)
    {
        Matrix kernel(3,4);
	kernel.Random(); // fill matrix with random numbers
	__images.OutputImageFromKernel.push_back(kernel);
    }
    for(auto &i: __images.OutputImageFromKernel)
        cout<<i<<endl<<endl;

    // 2-1) test copy
    cout<<"test copy."<<endl;
    Images c_image = __images;
    for(auto &i: c_image.OutputImageFromKernel)
        cout<<i<<endl<<endl;
    Images v_image = __images.Vectorization();
    for(auto &i: v_image.OutputImageFromKernel)
        cout<<i<<endl<<endl;

    // 3) dimension not match test
    //Matrix k(2, 4, 0);
    //__images.OutputImageFromKernel.push_back(k);
    //v_image = __images.Vectorization();

    // 4) tensorization test
    cout<<"Tensorization test"<<endl;
    Images tensor_image = v_image.Tensorization(3, 4);
    for(auto &i: tensor_image.OutputImageFromKernel)
        cout<<i<<endl<<endl;
}


void UnitTest::TestMatrix()
{
    Matrix m(4, 4);
    m.Random();

    cout<<"Test Matrix."<<endl;
    cout<<m<<endl;

    float v = m.MaxInSectionWithPadding(2, 6, 2, 6);
    cout<<v<<endl;

    cout<<"Test average in section with padding."<<endl;
    Matrix mm(4, 4, 1);
    cout<<mm<<endl;
    float vv = mm.AverageInSectionWithPadding(0, 6, 1, 6);
    cout<<vv<<endl;

    Matrix m1(3, 1, 0);
    Matrix m2(3, 1, 1);
    bool e = (m1 == m2);
    cout<<m1<<endl;
    cout<<m2<<endl;
    cout<<"matrix equal: "<<e<<endl;
}

void UnitTest::TestDNN()
{
    // setup a 1D data interface
    DataInterface *data_interface = new DataInterface("unit_test_data/data_signal.dat", "unit_test_data/data_cosmic.dat", LayerDimension::_1D);

    // setup a 1D input layer
    LayerParameterList p_list0(LayerType::input, LayerDimension::_1D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, Regularization::Undefined, 0, ActuationFuncType::Sigmoid);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // setup a FC layer
    LayerParameterList p_list3(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 5, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, true, 0.5, Regularization::L2, 0.1, ActuationFuncType::Sigmoid);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(layer_input);
    l3->Init();

    // setup an output layer
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l3);
    layer_output -> Init();

    // connect all  layers
    l3->SetNextLayer(layer_output); // This line is ugly, to be improved


    // loop epoch
    for(int epoch = 0;epoch<1;epoch++) // only 1 epoch
    { //
	// loop batch
	for(int nbatch = 0;nbatch<1;nbatch++) // only 1 batch
	{ //

	    // test start here
	    l3->BatchInit();
	    layer_output->BatchInit();

	    data_interface -> GetNewBatchData();
	    data_interface -> GetNewBatchLabel();
	    layer_input -> FillBatchDataToInputLayerA();

	    size_t sample_size = layer_input->GetBatchSize();
	    cout<<"sample size: "<<sample_size<<endl;

	    auto show_layer_in_forward = [&](Layer *l, size_t id)
	    {
		// show full a image
		cout<<"===layer=layer=layer=layer=layer=layer=layer=layer=layer==="<<endl;
		cout<<"layer id: "<<l->GetID()<<", "<<l->GetType()<<endl;

		cout<<"original w Matrix: "<<endl;
		for(auto &i: *(l->GetWeightMatrixOriginal()))
		    cout<<i<<endl;

		cout<<"active w Matrix: "<<endl;
		for(auto &i: *(l->GetWeightMatrix()))
		    cout<<i<<endl;

		cout<<"original bias vector: "<<endl;
		for(auto &i: *(l->GetBiasVectorOriginal()))
		    cout<<i<<endl;

		cout<<"active bias vector: "<<endl;
		for(auto &i: *(l->GetBiasVector()))
		    cout<<i<<endl;

		cout<<"active Z images: "<<endl;
		if((l->GetImagesActiveZ()).size() > 0)
		    for(auto &i: (l->GetImagesActiveZ())[id].OutputImageFromKernel)
			cout<<i<<endl;

		cout<<"full Z images: "<<endl;
		if((l->GetImagesFullZ()).size()>0)
		    for(auto &i: (l->GetImagesFullZ())[id].OutputImageFromKernel)
			cout<<i<<endl;

		cout<<"active a images: "<<endl;
		if((l->GetImagesActiveA()).size() > 0)
		    for(auto &i: (l->GetImagesActiveA())[id].OutputImageFromKernel)
			cout<<i<<endl;

		cout<<"full a images: "<<endl;
		if((l->GetImagesFullA()).size() > 0)
		    for(auto &i: (l->GetImagesFullA())[id].OutputImageFromKernel)
			cout<<i<<endl;
	    };

	    auto show_layer_in_backward = [&](Layer* l, size_t sample_id)
	    {
		cout<<"===layer=layer=layer=layer=layer=layer=layer=layer=layer==="<<endl;
		cout<<"layer id: "<<l->GetID()<<", "<<l->GetType()<<endl;

		cout<<"active delta Matrix: "<<endl;
		if((l->GetImagesActiveDelta()).size() > 0)
		for(auto &i: (l->GetImagesActiveDelta())[sample_id].OutputImageFromKernel)
		    cout<<i<<endl;

		cout<<"full delta Matrix: "<<endl;
		if((l->GetImagesFullDelta()).size() > 0)
		    for(auto &i: (l->GetImagesFullDelta())[sample_id].OutputImageFromKernel)
			cout<<i<<endl;
	    };

	    // loop sample forward direction
	    for(size_t sample_id = 0;sample_id < sample_size;sample_id++)
	    {
		cout<<"forward propagation sample: "<<sample_id<<endl;
		l3->ForwardPropagateForSample(sample_id);
		layer_output->ForwardPropagateForSample(sample_id);
	    }

	    //==============================================================
	    // use your eys here.... check forward propagation
	    //==============================================================

	    for(size_t id=0;id<sample_size;id++)
	    {
		cout<<"checking sample : "<<id<<endl;
		show_layer_in_forward(layer_input, id);
		show_layer_in_forward(l3, id);
		show_layer_in_forward(layer_output, id);

		getchar();
	    }


	    // loop sample backward direction


	} //
    } //
}































