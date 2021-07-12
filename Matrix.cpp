
#include "Matrix.h"
#include <cmath>
#include <cassert>

using namespace std;

Matrix::Matrix(){}

Matrix::Matrix(int r, int c){
    rows = r;
    cols = c;
    matrix.resize(rows * cols);
}

float &Matrix::at(int r, int c){
    //r * cols gets you to the row wanted
    //just add the column you want
    return matrix[r * cols + c];
}

Matrix Matrix::matMult(Matrix &m2){
    assert(cols = m2.rows);
    Matrix output(rows, m2.cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < m2.cols; ++j){
            for(int k = 0; k < cols; ++k){
                output.at(i, j) += at(i, k) * m2.at(k, j);
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

Matrix Matrix::elementMult(Matrix &m2){
    assert(rows == m2.rows && cols == m2.cols);
    Matrix output(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(i, j) = at(i, j) * m2.at(i, j);
        }
    }
    return output;
}

Matrix Matrix::matAdd(Matrix &m2){
    assert(rows == m2.rows && cols == m2.cols);
    Matrix output(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            output.at(i, j) = at(i, j) + m2.at(i, j);
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
            output.at(i, j) = at(i, j) * -1;
        }
    }
    return output;
}

Matrix Matrix::transpose(){
    Matrix output(cols, rows);
    for(int i = 0; i < output.rows; ++i){
        for(int j = 0; j < output.cols; ++j){
            output.at(i, j) = at(j, i);
        }
    }
    return output;
}