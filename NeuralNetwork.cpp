#include <cmath>
#include <cassert>
#include <functional>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(vector<int> shapeInput, float lr){
    shape = shapeInput;
    learningRate = lr;
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

void NeuralNetwork::initializeRandom(Matrix &m){
    for(int i = 0; i < m.rows; ++i){
        for(int j = 0; j < m.cols; ++j){
            m.at(i, j) = (float)rand() / RAND_MAX;
        }
    }
}

bool NeuralNetwork::feedForward(vector<float> input){
    if(input.size() != shape[0]){
        return false;
    }
    Matrix valueMatrix(input.size(), 1);
    valueMatrix.matrix = input;
    for(int i = 0; i < weights.size(); ++i){
        values[i] = valueMatrix;
        valueMatrix = weights[i].matMult(valueMatrix);
        valueMatrix = valueMatrix.matAdd(bias[i]);
        valueMatrix = activation(valueMatrix);
    }
    values[weights.size()] = valueMatrix;
    return true;
}

bool NeuralNetwork::backPropagate(vector<float> targetOutput){
    if(targetOutput.size() != shape.back()){
        return false;
    }
    Matrix error(targetOutput.size(), 1);
    error.matrix = targetOutput;
    Matrix output = values.back().negative();
    error = error.matAdd(output);
    for(int i = weights.size()-1; i >= 0; --i){
        Matrix trans = weights[i].transpose();
        Matrix prevError = trans.matMult(error);
        Matrix dOutput = dActivation(values[i + 1]);
        Matrix gradient = error.elementMult(dOutput);
        gradient = gradient.scalarMult(learningRate);
        Matrix tran2 = values[i].transpose();
        Matrix weightGradients = gradient.matMult(tran2);
        weights[i] = weights[i].matAdd(weightGradients);
        bias[i] = bias[i].matAdd(gradient);
        error = prevError;
    }

    return true;
}

Matrix NeuralNetwork::activation(Matrix &m){
    Matrix output(m.rows, m.cols);
    for(int i = 0; i < m.rows; ++i){
        for(int j = 0; j < m.cols; ++j){
            output.at(i, j) = 1.0 / (1 + exp(-m.at(i, j)));
        }
    }
    return output;
}

Matrix NeuralNetwork::dActivation(Matrix &m){
    Matrix output(m.rows, m.cols);
    for(int i = 0; i < m.rows; ++i){
        for(int j = 0; j < m.cols; ++j){
            output.at(i, j) = m.at(i, j) * (1 - m.at(i, j));
        }
    }
    return output;
}


vector<float> NeuralNetwork::getPred(){
    return values.back().matrix;
}
