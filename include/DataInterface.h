#ifndef DATA_INTERFACE_H
#define DATA_INTERFACE_H

#include "Matrix.h"
#include "Layer.h"

#include <vector>

class DataInterface
{
public:
    DataInterface();
    DataInterface(const char* path1, const char* path2); // for code development
    ~DataInterface();

    int GetBatchSize(){return gBatchSize;};
    void SetBatchSize(int s){gBatchSize = s;};
    int GetNumberOfBatches();

    // get out data in Matrix form
    std::vector<Matrix>& GetNewBatchData();
    std::vector<Matrix>& GetNewBatchLabel();
    std::vector<Matrix>& GetCurrentBatchData(){return __data;};
    std::vector<Matrix>& GetCurrentBatchLabel(){return __label;};

    // reform the data in Images form; 
    std::vector<Images>& GetNewBatchDataImage();
    std::vector<Images>& GetNewBatchLabelImage();
    std::vector<Images>& GetCurrentBatchDataImage(){return __data_image;};
    std::vector<Images>& GetCurrentBatchLabelImage(){return __label_image;};
 
    // a helper
    void UpdateBatch(std::vector<Matrix>& data_image, std::vector<Matrix>& label_image);

    std::pair<size_t, size_t> GetDataDimension(){return __dataDimension;};

    void test();

    void loadFile(const char* path, std::vector<Matrix> &m); // for code development

private:
    int gBatchSize = 100;
    int gDataIndex = 0; // indicate which batch
    int gLabelIndex = 0;

    // Get out data in Matrix form
    std::vector<Matrix> __data; // a vector of 2d image
    std::vector<Matrix> __label; // a vector of 2d image labels, used for training
    // Get out data in Images form
    std::vector<Images> __data_image; // a vector of 2d image
    std::vector<Images> __label_image; // a vector of 2d image labels, used for training

    std::pair<size_t, size_t> __dataDimension;

    //  Load all data to memory
    std::vector<Matrix> test_training_signal; // just for code development, loading all training data into this memory
    std::vector<Matrix> test_training_cosmic; // just for code development

};

#endif
