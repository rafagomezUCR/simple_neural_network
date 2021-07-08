#include <iostream>
#include "NeuralNetwork.h"
#include "Matrix.h"

using namespace std;

int main(){
    /*Matrix m(2, 4);
    Matrix m2(2, 3);
    m.matrix[0] = 1;
    m.matrix[1] = 2;
    m.matrix[2] = 3;
    m.matrix[3] = 1;
    m.matrix[4] = 6;
    m.matrix[5] = 8;
    m.matrix[6] = 4;
    m.matrix[7] = 3;

    m2.matrix[0] = 2;
    m2.matrix[1] = 1;
    m2.matrix[2] = 1;
    m2.matrix[3] = 6;
    m2.matrix[4] = 3;
    m2.matrix[5] = 4;

    Matrix trans = m.transpose();
    trans.print();
    trans.size();

    Matrix trans2 = m2.transpose();
    trans2.print();
    trans2.size();*/

    
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
        {0.0},
        {1.0},
        {1.0},
    };
    nn.feedForward(targetInputs[0]);
    //nn.backPropagate(targetOutputs[0]);
    int epoch = 1000;
    cout << "training" << endl;
    for(int i = 0; i < epoch; ++i){
        int index = rand() % 4;
        nn.feedForward(targetInputs[index]);
        //nn.backPropagate(targetOutputs[index]);
    }
    cout << "training done" << endl;
    for(int i = 0; i < targetInputs.size(); ++i){
        nn.feedForward(targetInputs[i]);
        vector<float> preds = nn.getPred();
        cout << targetInputs[i][0] << "," << targetInputs[i][1] << "->" << preds[i] << endl;
    }
    return 0;
}