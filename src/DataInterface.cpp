#include <iostream>
#include "DataInterface.h"

DataInterface::DataInterface()
{
    // place holder
    // currently only accepts 2D image format
    test();
}

DataInterface::~DataInterface()
{
    // place holder
}

void DataInterface::test()
{
    // only one image
    Matrix image(std::pair<int, int>(10, 10), 1);
    //std::cout<<"input image: "<<std::endl<<image<<std::endl;

    Matrix image2 = image.Reshape(100, 1);
    //std::cout<<"input image: "<<std::endl<<image2<<std::endl;


    __data.push_back(image2);
}

std::vector<Matrix>& DataInterface::GetNewBatch()
{
    //
    //  here fill new batch of data
    //
    //  __data.clear();
    //  ...
    //
    //  to be continued ...
    //

    return __data;
}


