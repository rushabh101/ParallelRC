#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

// Matrix dimensions
#define SQUARE 3
#define ROWS_A SQUARE
#define COLS_A SQUARE
#define ROWS_B COLS_A
#define COLS_B SQUARE

int main() {
    // Allocate memory for input and output matrices
    int *matrixA = (int*)malloc(sizeof(int) * ROWS_A * COLS_A);
    int *matrixB = (int*)malloc(sizeof(int) * ROWS_B * COLS_B);
    int *matrixC = (int*)malloc(sizeof(int) * ROWS_A * COLS_B);

    // Initialize input matrices with random values
    for (int i = 0; i < ROWS_A; i++) {
        for (int j = 0; j < COLS_A; j++) {
            matrixA[i * COLS_A + j] = i+j;
        }
    }
    for (int i = 0; i < ROWS_B; i++) {
        for (int j = 0; j < COLS_B; j++) {
            matrixB[i * COLS_B + j] = i+j;
        }
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("matrix_multiply_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    // printf("%d\n", device_id);


    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each matrix
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ROWS_A * COLS_A * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ROWS_B * COLS_B * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS_A * COLS_B * sizeof(int), NULL, &ret);

    // Copy the input matrices to the device
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, ROWS_A * COLS_A * sizeof(int), matrixA, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, ROWS_B * COLS_B * sizeof(int), matrixB, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    // ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&ROWS_A);
    // ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&COLS_A);
    // ret = clSetKernelArg(kernel, 5, sizeof(int), (void *)&COLS_B);

    // Set the global and local workgroup sizes
    size_t global_item_size[2] = { ROWS_A, COLS_B };
    size_t local_item_size[2] = { ROWS_A, COLS_B };

    // Execute the kernel on the device
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    // Read the output matrix from the device
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, ROWS_A * COLS_B * sizeof(int), matrixC, 0, NULL, NULL);

    // Print the output matrix
    printf("Matrix C:\n");
    for (int i = 0; i < ROWS_A; i++) {
        for (int j = 0; j < COLS_B; j++) {
            printf("%d ", matrixC[i * COLS_B + j]);
        }
        printf("\n");
    }

    // Cleanup
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(source_str);

    return 0;
}