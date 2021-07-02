#include "NeuralNetwork.h"
#include "Matrix.h"
#include <cstdlib>

using namespace std;

NeuralNetwork::NeuralNetwork(vector<int> shapeInput, float learningRate){
    learningRate = learningRate;
    for(int i = 0; i < shape.size() - 1; ++i){
        Matrix weightMatrix(shape[i + 1], shape[i]);
        weightMatrix = weightMatrix.applyFunction(
            [](const float &val){
                return (float)rand() / RAND_MAX;
        });
        weights.push_back(weightMatrix);
    }

    for(int i = 0; i < shape.size() - 1; ++i){
        Matrix biasMatrix(shape[i + 1], i);
        biasMatrix = biasMatrix.applyFunction(
            [](const float &val){
                return (float)rand() / RAND_MAX;
        });
        bias.push_back(biasMatrix);
    }

    values.resize(shape.size());
}

