#ifndef DATA_INTERFACE_H
#define DATA_INTERFACE_H

#include "Matrix.h"
#include <vector>

class DataInterface
{
public:
    DataInterface();
    DataInterface(const char* path1, const char* path2); // for code development
    ~DataInterface();

    int GetBatchSize(){return __data.size();};
    void SetBatchSize(int s){gBatchSize = s;};
    int GetNumberOfBatches();

    std::vector<Matrix>& GetNewBatchData();
    std::vector<Matrix>& GetNewBatchLabel();
    std::vector<Matrix>& GetCurrentBatchData(){return __data;};
    std::vector<Matrix>& GetCurrentBatchLabel(){return __label;};

    std::pair<size_t, size_t> GetDataDimension(){return __dataDimension;};

    void test();

    void loadFile(const char* path, std::vector<Matrix> &m); // for code development

private:
    int gBatchSize = 100;
    int gDataIndex = 0; // indicate which batch
    int gLabelIndex = 0;

    std::vector<Matrix> __data; // a vector of 2d image
    std::vector<Matrix> __label; // a vector of 2d image labels, used for training

    std::vector<Matrix> test_training_signal; // just for code development, loading all training data into this memory
    std::vector<Matrix> test_training_cosmic; // just for code development

    std::pair<size_t, size_t> __dataDimension;
};

#endif
