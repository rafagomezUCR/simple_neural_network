#pragma once

#include <vector>
#include <string>

class ETL{
    public:
        std::vector<std::vector<float>> data;
        std::vector<std::vector<float>> targetOutput;
        bool header;
        int rows;
        int cols;
    public:
        ETL();
        ETL(bool, int, int);
        void read_csv(std::string);
        std::vector<std::vector<float>> getData();
        std::vector<std::vector<float>> getOutput();
};