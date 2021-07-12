#pragma once

#include <vector>

using namespace std;

class Matrix {
    public:
        int rows;
        int cols;
        vector<float> matrix;
    public:
        Matrix();
        Matrix(int, int);
        float &at(int, int);
        Matrix matMult(Matrix &);
        Matrix scalarMult(float);
        Matrix elementMult(Matrix &);
        Matrix matAdd(Matrix &);
        Matrix scalarAdd(float);
        Matrix negative();
        Matrix transpose();
};