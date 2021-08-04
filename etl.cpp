#include "ETL.h"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

ETL::ETL(){};

ETL::ETL(bool h, int r, int c){
    header = h;
    if(h == true){
        rows = r - 1;
    }
    else{
        rows = r;
    }
    cols = c;
    data.resize(rows);
    for(int i = 0; i < rows; ++i){
        data[i].resize(c - 1);
    }
    targetOutput.resize(rows);
    for(int i = 0; i < rows; ++i){
        targetOutput[i].resize(1);
    }
}

void ETL::read_csv(std::string filename){
    std::ifstream file;
    file.open(filename);
    std::string line;
    std::vector<float> temp;
    if(header == true){
        getline(file, line);
    }
    while(getline(file, line)){
        std::stringstream ss(line);
        while(getline(ss, line, ',')){
            float f = atof(line.c_str());
            temp.push_back(f);
        }
    }
    int temp_it = 0;
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            if(j == cols - 1){
                targetOutput[i][0] = temp[temp_it];
                temp_it++;
            }
            else{
                data[i][j] = temp[temp_it];
                temp_it++;
            }
        }
    }
    file.close();
}

std::vector<std::vector<float>> ETL::getData(){
    return data;
}

std::vector<std::vector<float>> ETL::getOutput(){
    return targetOutput;
}