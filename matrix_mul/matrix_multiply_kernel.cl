
__kernel void matrix_multiply(__global const int *a, __global const int *b, __global int *c) {
    // Get the row and column indices of the current work item
    int sq = 3;
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Compute the dot product of the row from A and the column from B
    int sum = 0;
    for (int i = 0; i < sq; i++) {
        sum += a[row * sq + i] * b[i * sq + col];
    }

    // Store the result in the output matrix
    c[row * sq + col] = sum;
}
