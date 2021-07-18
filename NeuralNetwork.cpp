#include <cmath>
#include <cassert>
#include <vector>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(std::vector<int> shapeInput, float lr){
    shape = shapeInput;
    learningRate = lr;
    activations.resize(shape.size() - 1);
    for(int i = 0; i < shape.size() - 1; ++i){
        Matrix weightMatrix(shape[i + 1], shape[i]);
        initializeRandom(weightMatrix);
        w.push_back(weightMatrix);
        if(i == shape.size() - 2){
            activations[i] = "sigmoid";
        }
        else{
            activations[i] = "sigmoid";
        }
    }
    for(int i = 0; i < shape.size() - 1; ++i){
        Matrix biasMatrix(shape[i + 1], 1);
        initializeRandom(biasMatrix);
        b.push_back(biasMatrix);
    }
    a.resize(shape.size());
}

void NeuralNetwork::initializeRandom(Matrix &m){
    for(int i = 0; i < m.rows; ++i){
        for(int j = 0; j < m.cols; ++j){
            m.at(i, j) = (float)rand() / RAND_MAX;
        }
    }
}

bool NeuralNetwork::feedForward(std::vector<float> input, bool std){
    if(input.size() != shape[0]){
        std::cout << "data input shape does not equal network input shape" << std::endl;
        return false;
    }
    if(std == true){
        input = standardize(input);
    }
    Matrix neurons_output(input.size(), 1);
    neurons_output.matrix = input;
    for(int i = 0; i < w.size(); ++i){
        a[i] = neurons_output;
        neurons_output = w[i].matMult(neurons_output);
        neurons_output = neurons_output.matAdd(b[i]);
        neurons_output = activation(neurons_output, activations[i]);
    }
    a[w.size()] = neurons_output;
    return true;
}

bool NeuralNetwork::backPropagate(std::vector<float> targetOutput){
    if(targetOutput.size() != shape.back()){
        std::cout << "data ouput shape does not equal network output shape" << std::endl;
        return false;
    }
    Matrix target(targetOutput.size(), 1);
    target.matrix = targetOutput;
    Matrix error = target.matSub(a.back());
    Matrix total_error = error.square();
    total_errors.push_back(total_error.sumMatrix(total_error));
    for(int i = w.size()-1; i >= 0; --i){
        Matrix da = error;
        Matrix dz = dActivation(a[i + 1], activations[i]);
        Matrix delta = da.elementMult(dz);
        delta = delta.scalarMult(learningRate);
        Matrix dw = a[i].transpose();
        Matrix gradient = delta.matMult(dw);
        Matrix wTrans = w[i].transpose();
        error = wTrans.matMult(error);
        w[i] = w[i].matAdd(gradient);
        b[i] = b[i].matAdd(delta);
    }
    return true;
}

Matrix NeuralNetwork::activation(Matrix &m, std::string activation){
    Matrix output(m.rows, m.cols);
    if(activation == "sigmoid"){
        for(int i = 0; i < m.rows; ++i){
            for(int j = 0; j < m.cols; ++j){
                output.at(i, j) = 1.0 / (1 + exp(-m.at(i, j)));
            }
        }
    }
    else if(activation == "relu"){
        for(int i = 0; i < m.rows; ++i){
            for(int j = 0; j < m.cols; ++j){
                output.at(i, j) = std::max(0.0f, m.at(i, j));
            }
        }
    }
    return output;
}

Matrix NeuralNetwork::dActivation(Matrix &m, std::string activation){
    Matrix output(m.rows, m.cols);
    if(activation == "sigmoid"){
        for(int i = 0; i < m.rows; ++i){
            for(int j = 0; j < m.cols; ++j){
                output.at(i, j) = m.at(i, j) * (1 - m.at(i, j));
            }
        }
    }
    else if(activation == "relu"){
        for(int i = 0; i < m.rows; ++i){
            for(int j = 0; j < m.cols; ++j){
                if(m.at(i, j) >= 0){
                    output.at(i, j) = 1;
                }
                else{
                    output.at(i, j) = 0;
                }
            }
        }
    }
    return output;
}

std::vector<float> NeuralNetwork::standardize(std::vector<float> &input){
    float mean = 0;
    float std = 0;
    std::vector<float> result;
    for(int i = 0; i < input.size(); ++i){
        mean += input[i];
    }
    mean /= input.size();
    for(int i = 0; i < input.size(); ++i){
        std += std::pow(input[i] - mean, 2);
    }
    std = std::sqrt(std / input.size());
    result.resize(input.size());
    for(int i = 0; i < input.size(); ++i){
        result[i] = (input[i] - mean) / std;
    }
    return result;
}

void NeuralNetwork::print_errors(){
    for(int i = 0; i < 20; ++i){
        std::cout << total_errors[i] << std::endl;
    }
    std::cout << std::endl;
    for(int i = total_errors.size() - 20; i < total_errors.size() - 1; ++i){
        std::cout << total_errors[i] << std::endl;
    }
}


std::vector<float> NeuralNetwork::getPred(){
    return a.back().matrix;
}
