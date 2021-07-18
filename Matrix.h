#pragma once

#include <vector>

class Matrix {
    public:
        int rows;
        int cols;
        std::vector<float> matrix;
    public:
        Matrix();
        Matrix(int, int);
        float &at(int, int);
        Matrix matMult(Matrix &);
        Matrix scalarMult(float);
        Matrix elementMult(Matrix &);
        Matrix matAdd(Matrix &);
        Matrix matSub(Matrix &);
        Matrix scalarAdd(float);
        Matrix square();
        Matrix negative();
        Matrix transpose();
        float sumMatrix(Matrix &);
        void print();
};