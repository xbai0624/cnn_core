#include "UnitTest.h"

#include <iostream>

#include "Layer.h" // test layer

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
    TestImagesStruct();
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
