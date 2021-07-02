#include "Matrix.h"
#include <cmath>
#include <cassert>

using namespace std;

Matrix::Matrix(int rows, int cols){
    rows = rows;
    cols = cols;
    matrix.resize(rows);
    for(int i = 0; i < rows; ++i){
        matrix[i].resize(cols, 0.0);
    }
}

float &Matrix::at(int rows, int cols){
    return matrix[rows][cols];
}

Matrix Matrix::matMult(Matrix &target){
    assert(cols == target.rows);
    Matrix output(rows, target.cols);
    // CAN MESS UP HERE
    for(int i = 0; i < target.rows; ++i){
        for(int j = 0; j < target.cols; ++i){
            for(int k = 0; k < cols; ++i){
                output.at(i, j) += at(i, k) * target.at(k, j);
            }
        }
    }
    return output;
}

Matrix Matrix::scalarMult(float scalar){
    Matrix output(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(i, j) = at(i, j) * scalar;
        }
    }
    return output;
}

Matrix Matrix::matAdd(Matrix &target){
    assert(rows == target.rows && cols == target.cols);
    Matrix output(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(i, j) = at(i, j) + target.at(i, j);
        }
    }
    return output;
}

Matrix Matrix::scalarAdd(float scalar){
    Matrix output(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(i, j) = at(i, j) + scalar;
        }
    }
    return output;
}

Matrix Matrix::negative(){
    Matrix output(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(i, j) = -at(i, j);
        }
    }
    return output;
}

Matrix Matrix::transpose(){
    Matrix output(cols, rows);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(j, i) = at(i, j);
        }
    }
    return output;
}

Matrix Matrix::applyFunction(function<float(const float&)> func){
    Matrix output(cols, rows);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(j, i) = func(at(i, j));
        }
    }
    return output;
}