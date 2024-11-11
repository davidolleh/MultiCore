#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#pragma optimize( "" , off )


#define VECTOR_SIZE 16384
#define LOCAL_SIZE 256

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {
	FILE* file;
	fopen_s(&file, file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
}


void main() {
	cl_int err;

	// Platform ID
	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	// Device ID
	cl_device_id device;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	// Create Context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	// Create Command Queue
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	// Create Program Object
	size_t kernel_source_size;
	char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	// Build Program
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	build_error(program, device, err);
	CHECK_ERROR(err);

	// Create Vector A, B, C
	int *A = (int*)malloc(sizeof(int) * VECTOR_SIZE);
	int *B = (int*)malloc(sizeof(int) * VECTOR_SIZE);
	int *C = (int*)malloc(sizeof(int) * VECTOR_SIZE);

	// Initial Vector A, B
	cl_ushort idx;
	for (idx = 0; idx < VECTOR_SIZE; idx++) {
		A[idx] = rand() % 100;
		B[idx] = rand() % 100;
	}

	// Create kernel
	cl_kernel kernel;

	kernel = clCreateKernel(program, "vec_add", &err);
	CHECK_ERROR(err);

	clock_t start;
	start = clock();

	// Create Buffer
	cl_mem bufA, bufB, bufC;

	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * VECTOR_SIZE, NULL, &err);
	CHECK_ERROR(err);

	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * VECTOR_SIZE, NULL, &err);
	CHECK_ERROR(err);

	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * VECTOR_SIZE, NULL, &err);
	CHECK_ERROR(err);

	// Write Buffer
	err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(int) * VECTOR_SIZE, A, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(int) * VECTOR_SIZE, B, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Set Kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	CHECK_ERROR(err);

	start = clock();
	size_t global_size = VECTOR_SIZE;
	size_t local_size = LOCAL_SIZE;
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Read Buffer 
	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * VECTOR_SIZE, C, 0, NULL, NULL);
	CHECK_ERROR(err);

	printf("Execution time: %lfsec\n", (float)(clock() - start) / CLOCKS_PER_SEC);

	start = clock();
	for (idx = 0; idx < VECTOR_SIZE; idx++) {

		if (A[idx] + B[idx] != C[idx]) {
			printf("Verification failed! A[%d] = %d, B[%d] = %d, C[%d] = %d\n", idx, A[idx], idx, B[idx], idx, C[idx]);
			break;
		}
	}
	printf("\n");
	printf("Execution time: %lfsec\n", (float)(clock() - start) / CLOCKS_PER_SEC);

	if (idx == VECTOR_SIZE) {
		printf("Verification success!\n");
	}



	err = clReleaseMemObject(bufA);
	err = clReleaseMemObject(bufB);
	err = clReleaseMemObject(bufC);
	free(A);
	free(B);
	free(C);
	err = clReleaseKernel(kernel);
	err = clReleaseProgram(program);
	err = clReleaseCommandQueue(queue);
	err = clReleaseContext(context);

	free(kernel_source);
}