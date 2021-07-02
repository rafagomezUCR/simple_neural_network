#pragma once

#include "Matrix.h"
#include <vector>

using namespace std;

class NeuralNetwork {
    public:
        vector<int> shape;
        vector<Matrix> weights;
        vector<Matrix> values;
        vector<Matrix> bias;
        float learningRate;
    public:
        NeuralNetwork();
        NeuralNetwork(vector<int> inputShape, float):shape(inputShape){};
};