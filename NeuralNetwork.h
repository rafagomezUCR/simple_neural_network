#pragma once

#include "Matrix.h"
#include <cmath>
#include <cassert>
#include <functional>
#include <vector>

using namespace std;

inline float sigmoid(float x){
    return 1.0 / (1 + exp(-x));
}

inline float DSigmoid(float x){
    return x * (1 - x);
}

class NeuralNetwork {
    public:
        vector<int> shape;
        vector<Matrix> weights;
        vector<Matrix> values;
        vector<Matrix> bias;
        float learningRate;
    public:
        NeuralNetwork(){};

        NeuralNetwork(vector<int> shapeInput, float lr){
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

        void initializeRandom(Matrix &m){
            for(int i = 0; i < m.rows; ++i){
                for(int j = 0; j < m.cols; ++j){
                    m.at(i, j) = (float)rand() / RAND_MAX;
                }
            }
        };

        bool feedForward(vector<float> input){
            if(input.size() != shape[0]){
                return false;
            }
            Matrix valueMatrix(input.size(), 1);
            for(int i = 0; i < input.size(); ++i){
                valueMatrix.matrix[i] = input[i];
            }
            for(int i = 0; i < weights.size(); ++i){
                values[i] = valueMatrix;
                valueMatrix = valueMatrix.matMult(weights[i]);
                valueMatrix = valueMatrix.matAdd(bias[i]);
                valueMatrix = valueMatrix.applyFunction(sigmoid);
            }
            values[weights.size()] = valueMatrix;
            for(int i = 0; i < values.back().rows; ++i){
                for(int j = 0; j < values.back().cols; ++j){
                    cout << values.back().at(i, j) << " ";
                }
            }
            cout << endl;
            return true;
        };

        bool backPropagate(vector<float> targetOutput){
            if(targetOutput.size() != shape.back()){
                return false;
            }
            Matrix error(targetOutput.size(), 1);
            error.matrix = targetOutput;
            Matrix output = values.back().negative();
            for(int i = 0; i < output.rows; ++i){
                for(int j = 0; j < output.cols; ++j){
                    cout << output.at(i, j) << " ";
                }
            }
            cout << endl;
            for(int i = 0; i < error.rows; ++i){
                for(int j = 0; j < error.cols; ++j){
                    cout << error.at(i, j) << " ";
                }
            }
            cout << endl;
            error = error.matAdd(output);
            error.applyFunction([](const float &val){
                return val * val;
            });
            for(int i = 0; i < error.rows; ++i){
                for(int j = 0; j < error.cols; ++j){
                    cout << error.at(i, j) << " ";
                }
            }
            cout << endl;
            for(int i = weights.size()-1; i >= 0; --i){
                Matrix trans = weights[i].transpose();
                Matrix prevError = error.matMult(trans);
                Matrix dOutput = values[i + 1].applyFunction(DSigmoid);
                Matrix gradient = error.elementMult(dOutput);
                gradient = gradient.scalarMult(learningRate);
                Matrix tran2 = values[i].transpose();
                Matrix weightGradients = tran2.matMult(gradient);
                weights[i] = weights[i].matAdd(weightGradients);
                bias[i] = bias[i].matAdd(gradient);
                error = prevError;
            }

            return true;
        };

        vector<float> getPred(){
            return values.back().matrix;
        };
};