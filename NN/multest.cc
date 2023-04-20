#include <iostream>
#include "matrixMul.c"
#include "vectorAdd.c"

#define SIZE 3

using namespace std;
int main() {

    float *mA = (float*)malloc(sizeof(float) * SIZE * SIZE);
    float *mB = (float*)malloc(sizeof(float) * 1);
    float *mC = (float*)malloc(sizeof(float) * SIZE);

    // Initialize input matrices with random values
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            mA[i * SIZE + j] = i+j;
        }
    }
    
    mB[0] = 1;
    mB[1] = 2;
    mB[2] = 2;

    matmul(mA, mB, mC, 3, 3, 3, 1);

    cout<<"Mul print:\n";
    for(int i=0; i < 1; i++) {
        for(int j=0; j<3; j++) {
            cout<<mC[i*SIZE + j]<<" ";
        }
        cout<<endl;
    }

    // vectoradd(mB, mC, mC, SIZE, 1);

    // cout<<"Add print:\n";
    // for(int i=0; i < 1; i++) {
    //     for(int j=0; j<3; j++) {
    //         cout<<mC[i*3 + j]<<" ";
    //     }
    //     cout<<endl;
    // }
}