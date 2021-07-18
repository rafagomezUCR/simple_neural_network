#pragma once

#include "Matrix.h"
#include "Matrix.cpp"

class NeuralNetwork {
    public:
        std::vector<int> shape;
        std::vector<Matrix> w;
        std::vector<Matrix> a;
        std::vector<Matrix> b;
        std::vector<std::string> activations;
        std::vector<float> total_errors;
        float learningRate;
    public:
        NeuralNetwork();
        NeuralNetwork(std::vector<int>, float);
        void initializeRandom(Matrix &);
        bool feedForward(std::vector<float>, bool);
        bool backPropagate(std::vector<float>);
        Matrix activation(Matrix &, std::string);
        Matrix dActivation(Matrix &, std::string);
        std::vector<float> standardize(std::vector<float> &);
        std::vector<float> getPred();
        void print_errors();
};