#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

// Matrix dimensions
#define SQUARE 3
#define ROW_A SQUARE
#define COL_A SQUARE
#define ROW_B COL_A
#define COL_B SQUARE

float matmul(float *matrixA, float *matrixB, float *matrixC, int rowA, int colA, int rowB, int colB) {

    int args[3] = {rowB, rowA, colB};
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
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, rowA * colA * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, rowB * colB * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rowA * colB * sizeof(float), NULL, &ret);
    cl_mem args_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 3 * sizeof(int), NULL, &ret);

    // Copy the input matrices to the device
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, rowA * colA * sizeof(float), matrixA, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, rowB * colB * sizeof(float), matrixB, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, args_mem_obj, CL_TRUE, 0, 3 * sizeof(int), args, 0, NULL, NULL);

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
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&args_mem_obj);

    // Set the global and local workgroup sizes
    size_t global_item_size[2] = { rowA, colB };
    size_t local_item_size[2] = { rowA, colB };

    // Execute the kernel on the device
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    // Read the output matrix from the device
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, rowA * colB * sizeof(float), matrixC, 0, NULL, NULL);

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
    // free(matrixA);
    // free(matrixB);
    // free(matrixC);
    free(source_str);

    return 0;
}