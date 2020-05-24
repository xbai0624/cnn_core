/*
 * neuron class
 *     the weights are defined as a matrix, so we can unify cnn neurons and MLP neurons
 *         --MLP neurons weight matrix dimension is (1, n); 
 *         --CNN neuron weight matrix dimension is (m, n) --- each CNN neuron will have one independent copy of its corresponding kernel
 *
 *     cnn neuron and mlp neuron all only have one bias;
 */
#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstddef>

enum class ActuationFuncType 
{
    Sigmoid,
    Tanh,
    Relu
};

//#include "Matrix.h"
//#include "Layer.h"

class Matrix;
class Layer;
struct NeuronCoord;

class Neuron 
{
public:
    Neuron();
    ~Neuron();

    void PassWeightPointer(Matrix *);
    void PassBiasPointer(Matrix *);

    // 
    void ForwardPropagateForSample();
    void BackwardPropagateForBatch();

    bool IsActive(); // for dropout
    void Disable();
    void Enable();

    void Reset(); // reset for next training

    double __sigmoid(double); // actuation function
    double __tanh(double);
    double __relu(double);

    // setters
    void SetLayer(Layer* ); // layer information
    void SetPreviousLayer(Layer*);
    void SetNextLayer(Layer *);
    void SetCoord(size_t _i=0, size_t _j=0, size_t _k=0); // set neuron(pixel) coordinates in current layer
    void SetCoordI(size_t);
    void SetCoordJ(size_t);
    void SetCoordK(size_t);
    void SetCoord(NeuronCoord);
    void SetActuationFuncType(ActuationFuncType t);

    void UpdateA(); // update matrix a; a computation is independent of layer type, no need to design helper functions for different type layers
    void UpdateSigmaPrime(); // update derivative of a over z; sigma^prime

    void UpdateDelta(); // update matrix delta
    void UpdateDeltaCNN();
    void UpdateDeltaFC();
    void UpdateDeltaPooling();

    void UpdateZ(); // update matrix z
    void UpdateZCNN();
    void UpdateZFC();
    void UpdateZPooling();

    // getters 
    std::vector<double> & GetAVector();
    std::vector<double> & GetDeltaVector();
    std::vector<double> & GetZVector();
    NeuronCoord GetCoord();

    // 
    void Print();

private:
    static long __neuron_Count;
    long __neuron_id;
    Matrix *__w;  // weights
    Matrix *__b;  // bias : should be a 1 element matrix

    // for batch training, after each batch, these vectors will be cleared for next batch
    std::vector<double> __a; // the length is dependent on training batch size
    std::vector<double> __delta;
    std::vector<double> __z;
    std::vector<double> __sigmaPrime; // derivative of Z

    // the layer that this neurons belongs to
    Layer *__layer;
    Layer *__previousLayer;
    Layer *__nextLayer;

    // for dropout algorithm, dropout algorithm will disable
    // some neurons in a layer
    bool __active_status = true; 
    // actuation function type used in this neuron
    ActuationFuncType __funcType = ActuationFuncType::Sigmoid;

    NeuronCoord __coord; // active coordinates
};

#endif
