__kernel void vector_add(__global const float *A, __global const float *B, __global float *C, __global int *type) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    if(*type == 1) {
        C[i] = A[i] + B[i];
    }
    else if(*type == 2) {
        C[i] = A[i] - B[i];
    }
    else if(*type == 3) {
        C[i] = A[i] > B[i] ? 1 : 0;
    }
    else if(*type == 4) {
        C[i] = A[i] * B[i];
    }
}