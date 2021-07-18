#include <iostream>
#include <time.h>
#include "NeuralNetwork.h"
#include "NeuralNetwork.cpp"
#include "etl.h"
#include "etl.cpp"

int main(){
    srand(time(0));
    std::vector<int> shape = {2, 2, 1};
    NeuralNetwork nn(shape, 0.5);
    bool std = false;


    // ETL etl(true, 304, 14);
    // etl.read_csv("heart.csv");
    // std::vector<std::vector<float>> targetInputs = etl.getData();
    // std::vector<std::vector<float>> targetOutputs = etl.getOutput();


    std::vector<std::vector<float>> targetInputs = {
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
    };
    std::vector<std::vector<float>> targetOutputs = {
        {1.0},
        {1.0},
        {0.0},
        {0.0},
    };

    int epoch = 10000;
    std::cout << "training" << std::endl;
    for(int i = 0; i < epoch; ++i){
        int index = rand() % targetInputs.size();
        nn.feedForward(targetInputs[index], std);
        nn.backPropagate(targetOutputs[index]);
    }
    std::cout << "training done" << std::endl;
    for(int i = 0; i < targetInputs.size(); ++i){
        nn.feedForward(targetInputs[i], std);
        std::vector<float> preds = nn.getPred();
        std::cout << i << " -> " << preds[0] << std::endl;
    }
    return 0;
}