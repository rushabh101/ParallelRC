
__kernel void matrix_multiply(__global const float *a, __global const float *b, __global float *c, __global int *args) {
    // Get the row and column indices of the current work item
    int comm = args[0]; //col a & row b
    int row_a = args[1];
    int col_b = args[2];
    int row = get_global_id(0);
    int col = get_global_id(1);

    // printf("row: %d; col: %d\n", row, col);
    // Compute the dot product of the row from A and the column from B
    int sum = 0;
    for (int i = 0; i < comm; i++) {
        sum += a[row * comm + i] * b[i * col_b + col];
        // printf("ddd %d", row * row_a + i);
    }

    // Store the result in the output matrix
    c[row * col_b + col] = sum;
}
