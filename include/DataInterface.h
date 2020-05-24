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
    std::vector<Matrix>& GetNewBatchLabel();
    std::vector<Matrix>& GetCurrentBatch(){return __data;};
    std::vector<Matrix>& GetCurrentBatchLabel(){return __label;};


    void test();

private:
    std::vector<Matrix> __data; // a vector of 2d image
    std::vector<Matrix> __label; // a vector of 2d image labels, used for training
};

#endif
