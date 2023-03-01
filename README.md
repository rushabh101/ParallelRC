## Using OpenCL instead of CUDA

Initally, I had planned to use CUDA to implement parallel programming on GPUs, 
however, although CUDA is a very popular library, finding use in DL training, 
it can only run on NVIDIA GPUs that support CUDA.

OpenCL targets many architectures and will be supported on almost all devices. 
It has similar parallel processing abilities and syntax when compared to CUDA.


## Installing and Using OpenCL
### Linux
You'll need to be able to compile and run OpenCL code on your systems.

In order to compile, we'll need to install the relevant libraries.

---

```
sudo apt install build-essential -y
```
This installs GNU compiler set, as well as some other useful tools

```
sudo apt install opencl-headers ocl-icd-opencl-dev -y
```
This installs the OpenCL libraries needed to compile C/C++ OpenCL code


### Windows
Use WSL and follow the steps in the Linux Section

### Drivers
In order to correctly execute OpenCL code, we will the need the respective drivers. These vary from platform and manufacturer, 
you will be able to find the packages on the manufacturers page (Nvidia, Intel, AMD).

If the correct drivers are not installed, you will likely get garbage values or zeroes on running an OpenCL program.

## Vector Addition using OpenCL

### vector_add_kernel.cl
This is the kernel code. The kernel is what will run in parallel on the target compute device.

```C
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}
```
### driver.c
This is the main code. A part by part explanation can be found below.
```C
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    printf("%d\n", device_id);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Process in groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

    // Display the result to the screen
    for(i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}
```


## Explanation

Standard initializing of variables and the arrays we will use.
```C
int i;
const int LIST_SIZE = 1024;
int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
for(i = 0; i < LIST_SIZE; i++) {
    A[i] = i;
    B[i] = LIST_SIZE - i;
}
```

---

OpenCL requires us to have the kernel code stored in a string. Here we open the kernel file and store its contents in the string source_str.
```C
// Load the kernel source code into the array source_str
FILE *fp;
char *source_str;
size_t source_size;

fp = fopen("vector_add_kernel.cl", "r");
if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
}
source_str = (char*)malloc(MAX_SOURCE_SIZE);
source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
fclose( fp );
```

---

Here we get the details of the platform and the OpenCL supported device, i.e. the device where our kernel will run in parallel.
```C
// Get platform and device information
cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;   
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
```

---

Using the device id, we open an OpenCL context.

An OpenCL context is the main object which is used by the OpenCL runtime for managing objects such as command queues, memory, program, and kernel objects, and for executing kernels on the one or more devices specified in the context.

A command queue is the object responsible for sending commands to the specified device.
```C
// Create an OpenCL context
cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
printf("%d\n", device_id);

// Create a command queue
cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
```

---

When executing a kernel, the specified device will access the data from its own memory.
OpenCL communicates with the global memory of the device.
In order to allocate this, we use buffers.
```C
// Create memory buffers on the device for each vector 
cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);

// Copy the lists A and B to their respective memory buffers
ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
```

---

From the earlier source_str and the OpenCL context, we create the kernel program, build it, and create a kernel which will execute on the specified device, all using inbuilt OpenCL functions
```C
// Create a program from the kernel source
cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

// Build the program
ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

// Create the OpenCL kernel
cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
```

---

We set the arguments of the kernel to the buffers we created.
```C
// Set the arguments of the kernel
ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
```

---

Here we specifiy the number of parallel kernels which will be processed using local_item_size and the size of the entire batch using global_item_size.

Then we execute the kernel with the given parameters.
```C
// Execute the OpenCL kernel on the list
size_t global_item_size = LIST_SIZE; // Process the entire lists
size_t local_item_size = 64; // Process in groups of 64
ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
```

---

Similar to storing data into buffers, we now load our final data from the buffer.
```C
// Read the memory buffer C on the device to the local variable C
int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
```

---

Straightforward, print the output and close all the created OpenCl objects.
```C
// Display the result to the screen
for(i = 0; i < LIST_SIZE; i++)
    printf("%d + %d = %d\n", A[i], B[i], C[i]);

// Clean up
ret = clFlush(command_queue);
ret = clFinish(command_queue);
ret = clReleaseKernel(kernel);
ret = clReleaseProgram(program);
ret = clReleaseMemObject(a_mem_obj);
ret = clReleaseMemObject(b_mem_obj);
ret = clReleaseMemObject(c_mem_obj);
ret = clReleaseCommandQueue(command_queue);
ret = clReleaseContext(context);
free(A);
free(B);
free(C);
return 0;
```

---

### Compilation

you can compile and run the code using:
```
gcc driver.c -o vectorAddition -lOpenCL
./vectorAddition
```

### Homework
Implement an OpenCL program which multiplies 2 matrices of arbitrary size and outputs the value.

HINTS: 
- For 3 given matrixes where A*B=C, one kernel execution should compute one element of C.
- Try to use B as its transpose.
- Read documentation for ```get_global_id()``` and ```clEnqueueNDRangeKernel()```