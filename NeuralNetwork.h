#pragma once

#include "Matrix.h"
#include "Matrix.cpp"

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
        NeuralNetwork(vector<int>, float);
        void initializeRandom(Matrix &);
        bool feedForward(vector<float>);
        bool backPropagate(vector<float>);
        Matrix activation(Matrix &);
        Matrix dActivation(Matrix &);
        vector<float> getPred();
};