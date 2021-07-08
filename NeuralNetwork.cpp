#include "NeuralNetwork.h"
#include <cstdlib>

using namespace std;

NeuralNetwork::NeuralNetwork(vector<int> shapeInput, float learningRate){
    shape = shapeInput;
    learningRate = learningRate;
    for(int i = 0; i < shape.size() - 1; ++i){
        Matrix weightMatrix(shape[i + 1], shape[i]);
        initializeRandom(weightMatrix);
        weights.push_back(weightMatrix);
    }

    for(int i = 0; i < shape.size() - 1; ++i){
        Matrix biasMatrix(shape[i + 1], 1);
        initializeRandom(biasMatrix);
        bias.push_back(biasMatrix);
    }

    values.resize(shape.size());
}

void NeuralNetwork::initializeRandom(Matrix &target){
    for(int i = 0; i < target.rows; ++i){
        for(int j = 0; j < target.cols; ++j){
            target.at(i, j) = (float)rand() / RAND_MAX;
        }
    }
}

inline float sigmoid(float x){
    return 1.0 / (1 + exp(-x));
}

inline float DSigmoid(float x){
    return x * (1 - x);
}

bool NeuralNetwork::feedForward(vector<float> input){
    if(input.size() != shape[0]){
        return false;
    }
    Matrix valueMatrix(input.size(), 1);
    for(int i = 0; i < input.size(); ++i){
        valueMatrix.matrix[i][0] = input[i];
    }
    for(int i = 0; i < weights.size(); ++i){
        values[i] = valueMatrix;
        valueMatrix = weights[i].matMult(valueMatrix);
        //valueMatrix = valueMatrix.matMult(weights[i]);
        valueMatrix = valueMatrix.matAdd(bias[i]);
        valueMatrix = valueMatrix.applyFunction(sigmoid);
    }
    values[weights.size()] = valueMatrix;
    return true;
}

bool NeuralNetwork::backPropagate(vector<float> targetOutput){
    if(targetOutput.size() != shape.back()){
        return false;
    }
    Matrix error(targetOutput.size(), 1);
    for(int i = 0; i < targetOutput.size(); ++i){
        error.matrix[i][0] = targetOutput[i];
    }
    Matrix neg = values.back();
    neg = neg.negative(neg);
    error.matAdd(neg);
    for(int i = weights.size()-1; i >= 0; --i){
        Matrix prevError = weights[i].transpose().matMult(error);
        //Matrix prevError = error.matMult(weights[i].transpose());
        Matrix dOutput = values[i + 1].applyFunction(DSigmoid);
        Matrix gradient = error.elementMult(dOutput);
        gradient = gradient.scalarMult(learningRate);
        Matrix weightGradients = values[i].transpose().matMult(gradient);
        weights[i] = weights[i].matAdd(weightGradients);
        bias[i] = bias[i].matAdd(gradient);
        error = prevError;
    }
    return true;
}


vector<vector<float>> NeuralNetwork::getPred(){
    return values.back().matrix;
}
