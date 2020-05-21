#ifndef DATA_INTERFACE_H
#define DATA_INTERFACE_H

#include "Matrix.h"
#include <vector>

class DataInterface
{
public:
    DataInterface();
    ~DataInterface();

    int GetBatchSize(){return __data.size();};
    std::vector<Matrix>& GetNewBatch();

    void test();

private:
    std::vector<Matrix> __data; // a vector of 2d image
};

#endif
