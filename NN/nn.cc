#include <iostream>
#include <cmath>
#include <algorithm>
#include "matrixMul.c"
#include "vectorAdd.c"


using namespace std;

class NeuralNetwork {
private:
    int num_inputs_;
    int num_hidden_;
    int num_outputs_;
    float* weights_input_hidden_;
    float* weights_hidden_output_;
    float* biases_hidden_;
    float* biases_output_;

    float* hidden_layer;
    float* output;
public:
    NeuralNetwork(int num_inputs, int num_hidden, int num_outputs) : num_inputs_(num_inputs), num_hidden_(num_hidden), num_outputs_(num_outputs)
    {
        // initialize weights and biases

        weights_input_hidden_ = (float*)malloc(num_inputs_ * num_hidden_ * sizeof(float));
        weights_hidden_output_ = (float*)malloc(num_hidden_ * num_outputs_ * sizeof(float));
        biases_hidden_ = (float*)malloc(num_hidden_ * sizeof(float));
        biases_output_ = (float*)malloc(num_outputs_ * sizeof(float));
        hidden_layer = (float*)malloc(num_hidden_ * sizeof(float));
        output = (float*)malloc(sizeof(float) * num_outputs_);


        fill(weights_input_hidden_, weights_input_hidden_+ (num_inputs_ * num_hidden_), 1);
        fill(weights_hidden_output_, weights_hidden_output_+ (num_hidden_ * num_outputs_), 1);
        fill(biases_hidden_, biases_output_ + num_hidden_, 1);
        fill(biases_output_, biases_output_ + num_outputs_, 1);
    }

    void forward(float* inputs) {
        matmul(inputs, weights_input_hidden_, hidden_layer, 1, num_inputs_, num_inputs_, num_hidden_);
        vectoradd(hidden_layer, biases_hidden_, hidden_layer, num_hidden_, 1);

        matmul(hidden_layer, weights_hidden_output_, output, 1, num_hidden_, num_hidden_, num_outputs_);
        vectoradd(output, biases_output_, output, num_outputs_, 1);

        float *threshold_array = (float*)malloc(num_outputs_ * sizeof(float));
        fill(threshold_array, threshold_array+num_outputs_, 1);
        vectoradd(output, threshold_array, output, num_outputs_, 3);
    }

    void train(float* inputs, float* targets, float learning_rate) {

        float *learning_array = (float*)malloc(max(num_inputs_ * num_hidden_, num_hidden_ * num_outputs_) * sizeof(float));
        fill(learning_array, learning_array+max(num_inputs_ * num_hidden_, num_hidden_ * num_outputs_), learning_rate);
        // forward pass
        forward(inputs);

        // calculate error signals
        float *output_error = (float*)malloc(num_outputs_ * sizeof(float));
        vectoradd(output, targets, output_error, num_outputs_, 2);

        float *hidden_error = (float*)malloc(num_hidden_ * sizeof(float));
        matmul(weights_hidden_output_, output_error, hidden_error, num_hidden_, num_outputs_, num_outputs_, 1);

        // update weights and biases
        float *weights_hidden_output_error = (float*)malloc(num_hidden_ * num_outputs_ * sizeof(float));
        matmul(hidden_layer, output_error, weights_hidden_output_error, num_hidden_, 1, 1, num_outputs_);
        vectoradd(weights_hidden_output_error, learning_array, weights_hidden_output_error, num_hidden_ * num_outputs_, 4);
        vectoradd(weights_hidden_output_, weights_hidden_output_error, weights_hidden_output_, num_hidden_ * num_outputs_, 2);

        float *biases_output_error = (float*)malloc(num_outputs_*sizeof(float));
        vectoradd(output_error, learning_array, biases_output_error, num_outputs_, 4);
        vectoradd(biases_output_, biases_output_error, biases_output_, num_outputs_, 2);

        float *weights_input_hidden_error = (float*)malloc(num_inputs_ * num_hidden_ * sizeof(float));
        matmul(inputs, hidden_error, weights_input_hidden_error, num_inputs_, 1, 1, num_hidden_);
        vectoradd(weights_input_hidden_error, learning_array, weights_input_hidden_error, num_inputs_ * num_hidden_, 4);
        vectoradd(weights_input_hidden_, weights_input_hidden_error, weights_input_hidden_, num_inputs_ * num_hidden_, 2);

        float *biases_hidden_error = (float*)malloc(num_hidden_*sizeof(float));
        vectoradd(hidden_error, learning_array, biases_hidden_error, num_hidden_, 4);
        vectoradd(biases_hidden_, biases_hidden_error, biases_hidden_, num_hidden_, 2);        

        // Printing
        cout<<"Output: ";
        for(int i=0; i<num_outputs_; i++) {
            cout<<(output[i] == 0)<<" ";
        }
        cout<<"; ";
        cout<<"Targets: ";
        for(int i=0; i<num_outputs_; i++) {
            cout<<targets[i]<<" ";
        }
        cout<<endl;
    }
};

int main() {
    // example usage
    NeuralNetwork nn(2, 3, 1);
    
    float input[4][2] = {{1, 0}, {0,1}, {0,0}, {1,1}};
    float targets[4][1] = {{1}, {1}, {0}, {0}};

    int epochs = 5;
    for(int j=0; j < epochs; j++) {
        for(int i =0; i < 4; i++) {
            nn.train(input[i], targets[i], 0.2);
        }
    }
    
    return 0;
}
