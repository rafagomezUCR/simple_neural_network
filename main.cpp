#include <iostream>
#include <time.h>
#include "NeuralNetwork.h"
#include "NeuralNetwork.cpp"

using namespace std;

int main(){
    srand(time(0));
    vector<int> shape = {2, 3, 1};
    NeuralNetwork nn(shape, 0.1);
    vector<vector<float>> targetInputs = {
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
    };
    vector<vector<float>> targetOutputs = {
        {0.0},
        {1.0},
        {0.0},
        {0.0},
    };
    int epoch = 100000;
    cout << "training" << endl;
    for(int i = 0; i < epoch; ++i){
        int index = rand() % 4;
        nn.feedForward(targetInputs[index]);
        nn.backPropagate(targetOutputs[index]);
    }
    cout << "training done" << endl;
    for(int i = 0; i < targetInputs.size(); ++i){
        nn.feedForward(targetInputs[i]);
        vector<float> preds = nn.getPred();
        cout << targetInputs[i][0] << "," << targetInputs[i][1] << "->" << preds[0] << endl;
    }
    return 0;
}