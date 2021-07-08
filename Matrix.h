#pragma once

#include <cmath>
#include <cassert>
#include <functional>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;

class Matrix {
    public:
        int rows;
        int cols;
        vector<float> matrix;
    public:
        Matrix(){};

        Matrix(int r, int c){
            rows = r;
            cols = c;
            matrix.resize(rows * cols);
        };

        float &at(int r, int c){
            //r * cols gets you to the row wanted
            //just add the column you want
            return matrix[r * cols + c];
        };

        Matrix matMult(Matrix &m2){
            assert(rows == m2.cols);
            Matrix output(m2.rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < m2.cols; ++j){
                    for(int k = 0; k < m2.rows; ++k){
                        output.at(i, j) += at(i, k) * m2.at(k, j);
                    }
                }
            }
            return output;
        };

        Matrix scalarMult(float scalar){
            Matrix output(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(i, j) = at(i, j) * scalar;
                }
            }
            return output;
        };

        Matrix elementMult(Matrix &m2){
            assert(rows == m2.rows && cols == m2.cols);
            Matrix output(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(i, j) = at(i, j) * m2.at(i, j);
                }
            }
            return output;
        };

        Matrix matAdd(Matrix &m2){
            assert(rows == m2.rows && cols == m2.cols);
            Matrix output(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(i, j) = at(i, j) + m2.at(i, j);
                }
            }
            return output;
        };

        Matrix scalarAdd(float scalar){
            Matrix output(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(i, j) = at(i, j) + scalar;
                }
            }
            return output;
        };

        Matrix negative(){
            Matrix output(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(i, j) = at(i, j) * -1;
                }
            }
            return output;
        };

        Matrix transpose(){
            Matrix output(cols, rows);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(i, j) = at(j, i);
                }
            }
            return output;
        };

        Matrix applyFunction(function<float(const float &)> func){
            Matrix output(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    output.at(j, i) = func(at(i, j));
                }
            }
            return output;
        };
};