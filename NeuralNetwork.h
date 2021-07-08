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
            valueMatrix.matrix = input;
            for(int i = 0; i < weights.size(); ++i){
                values[i] = valueMatrix;
                valueMatrix = weights[i].matMult(valueMatrix);
                valueMatrix = valueMatrix.matAdd(bias[i]);
                valueMatrix = valueMatrix.applyFunction(sigmoid);
            }
            values[weights.size()] = valueMatrix;
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
            error = error.square();
            for(int i = 0; i < error.rows; ++i){
                for(int j = 0; j < error.cols; ++j){
                    cout << error.at(i, j) << " ";
                }
            }
            cout << endl;
            Matrix dOutput = values[2].applyFunction(DSigmoid);
            Matrix gradient = error.elementMult(dOutput);
            gradient = gradient.scalarMult(learningRate);
            for(int i = 0; i < gradient.rows; ++i){
                for(int j = 0; j < gradient.cols; ++j){
                    cout << gradient.at(i, j) << " ";
                }
            }
            cout << endl;
            for(int i = 0; i < values[1].rows; ++i){
                for(int j = 0; j < values[1].cols; ++j){
                    cout << values[1].at(i, j) << " ";
                }
            }
            cout << endl;
            Matrix tran2 = values[1].transpose();
            cout << endl;
            for(int i = 0; i < tran2.rows; ++i){
                for(int j = 0; j < tran2.cols; ++j){
                    cout << tran2.at(i, j) << " ";
                }
            }
            cout << endl;
            gradient.size();
            tran2.size();
            Matrix weightGradients = tran2.matMult(gradient);
            for(int i = 0; i < weightGradients.rows; ++i){
                for(int j = 0; j < weightGradients.cols; ++j){
                    cout << weightGradients.at(i, j) << " ";
                }
            }
            cout << endl;
            for(int i = 0; i < weights[1].rows; ++i){
                for(int j = 0; j < weights[1].cols; ++j){
                    cout << weights[1].at(i, j) << " ";
                }
            }
            cout << endl;
            weights[1] = weights[1].matAdd(weightGradients);
            for(int i = 0; i < weights[1].rows; ++i){
                for(int j = 0; j < weights[1].cols; ++j){
                    cout << weights[1].at(i, j) << " ";
                }
            }
            cout << endl;
            bias[1] = bias[1].matAdd(gradient);

            //Matrix trans = weights[1].transpose();
            //Matrix prevError = error.matMult(trans);
            Matrix dOutput2 = values[0 + 1].applyFunction(DSigmoid);
            Matrix gradient2 = error.elementMult(dOutput2);
            gradient2 = gradient2.scalarMult(learningRate);
            Matrix tran22 = values[0].transpose();
            Matrix weightGradients2 = tran22.matMult(gradient2);
            bias[0] = bias[0].matAdd(gradient2);
            /*for(int i = weights.size()-1; i >= 0; --i){
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
            }*/

            return true;
        };

        vector<float> getPred(){
            return values.back().matrix;
        };
};