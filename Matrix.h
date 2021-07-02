#pragma once

#include <cmath>
#include <functional>
#include <vector>

using namespace std;

class Matrix {
    public:
        int rows;
        int cols;
        vector<vector<float>> matrix;
    public:
        Matrix();
        Matrix(int, int);
        float &at(int, int);
        Matrix matMult(Matrix &);
        Matrix scalarMult(float);
        Matrix matAdd(Matrix &);
        Matrix scalarAdd(float);
        Matrix negative();
        Matrix transpose();
        Matrix applyFunction(function<float(const float &)>);
};