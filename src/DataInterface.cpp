#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>

#include "DataInterface.h"

using namespace std;

DataInterface::DataInterface()
{
    // place holder
    // currently only accepts 2D image format
    test();
}


DataInterface::DataInterface(const char* p_signal, const char* p_cosmic)
{
    // for code development
    loadFile(p_signal, test_training_signal);
    loadFile(p_cosmic, test_training_cosmic);
}


DataInterface::~DataInterface()
{
    // place holder
}

int DataInterface::GetNumberOfBatches()
{
    // this line is to maliciously cause troulbe, what a fuck!
    assert(test_training_signal.size() == test_training_cosmic.size());
    int total_entries = test_training_signal.size();

    int res = total_entries/gBatchSize;
    if(total_entries%gBatchSize == 0) return res;
    else return res+1;
}

void DataInterface::test()
{
    // only one image
    Matrix image(std::pair<int, int>(10, 10), 1);
    //std::cout<<"input image: "<<std::endl<<image<<std::endl;

    Matrix image2 = image.Reshape(100, 1);
    //std::cout<<"input image: "<<std::endl<<image2<<std::endl;


    __data.push_back(image2);

    // label for this image
    Matrix label1(std::pair<int, int>(10, 1), 0);
    label1[0][0] = 1.;
    __label.push_back(label1);
}

std::vector<Matrix>& DataInterface::GetNewBatchData()
{
    //
    //  here fill new batch of data
    //
    //  __data.clear();
    //  ...
    //
    //  to be continued ...
    //

    __data.clear();
    __label.clear();


    //---------------------------------------------
    // prepare data
    int offset = gDataIndex * gBatchSize;
    for(int i=0;i<gBatchSize;i++) // signal data
    {
        Matrix M = test_training_signal[offset+i].Reshape(100, 1);
        __data.push_back(M);
    }
    for(int i=0;i<gBatchSize;i++) // cosmic data
    {
        Matrix M = test_training_cosmic[offset+i].Reshape(100, 1);
        __data.push_back(M);
    }

    gDataIndex++;

    //---------------------------------------------
    // prepare label
    Matrix signal_label_m(2, 1, 0); // signal label
    signal_label_m[0][0] = 1;
    Matrix cosmic_label_m(2, 1, 0); // cosmic label
    cosmic_label_m[1][0] = 1;
    for(int i=0;i<gBatchSize;i++)
    {
        __label.push_back(signal_label_m);
    }
    for(int i=0;i<gBatchSize;i++)
    {
        __label.push_back(cosmic_label_m);
    }

    return __data;
}


std::vector<Matrix>& DataInterface::GetNewBatchLabel()
{
    if(gLabelIndex + 1 != gDataIndex)
    {
        cout<<"Error: DataInterface data & label are not aligned."<<endl;
	exit(0);
    }
    gLabelIndex++;

    return __label;
}

void DataInterface::loadFile(const char* path, std::vector<Matrix> &contents)
{
    // this only for code development, it reads data in ./test_data/ directory
    fstream f(path, fstream::in);
    string line;
    while(getline(f, line))
    {
        istringstream iss(line);
	string tmp;
	vector<double> vec;
	while(iss>>tmp)
	{
	    if(tmp.size() > 0) 
	    {
	        double a = stod(tmp);
		vec.push_back(a);
	    }
	}
	assert(vec.size() == 27);

	Matrix m(10, 10);
	for(size_t i=0;i<vec.size();i+=3)
	{
	    int ii = vec[i];
	    int jj = vec[i+1];
	    float val = vec[i+2];

	    m[ii][jj] = val;
	}

	contents.push_back(m);
    }

    // *** implement the image dimension
    Matrix tmp = contents[0].Reshape(100, 1);
    __dataDimension = tmp.Dimension();
}
