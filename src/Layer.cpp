#include "Layer.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream &os, const Filter2D &t)
{
    for(size_t i=0;i<t.__filter.size();i++)
    {
        for(size_t j=0;j<t.__filter[i].size();j++)
        {
            os<<std::setfill(' ')<<std::setw(4)<<t.__filter[i][j];
        }
        os<<std::endl;
    }
    return os;
}

std::ostream & operator<<(std::ostream& os, const NeuronCoord &c)
{
    os<<std::setfill(' ')<<std::setw(4)<<c.i
	<<std::setfill(' ')<<std::setw(4)<<c.j
	<<std::setfill(' ')<<std::setw(4)<<c.k
	<<std::endl;
    return os;
}

std::ostream & operator<<(std::ostream &os, const Images & images)
{
    os<<"images from all kernels during one training sample:"<<std::endl;
    for(size_t i=0;i<images.OutputImageFromKernel.size();i++)
    {
        os<<"kernel: "<<i<<std::endl;
	Matrix m = (images.OutputImageFromKernel)[i];
	os<<m<<std::endl;
    }
    return os;
}



int Layer::__layerCount = 0;

Layer::Layer()
{
    // place holder
    __layerID = __layerCount;
    __layerCount++;
}

Layer::~Layer()
{
    // place holder
}
