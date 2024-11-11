#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#define ROWA 1000
#define COLUMNA_ROWB 1000
#define COLUMNB 1000

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

int main() {
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

	float *A = (float *)malloc(sizeof(float) * ROWA * COLUMNA_ROWB);
	float *B = (float *)malloc(sizeof(float) * COLUMNA_ROWB * COLUMNB);
	float *C = (float *)malloc(sizeof(float) * ROWA * COLUMNB);
	float *checkC = (float *)malloc(sizeof(float) * ROWA * COLUMNB);

	for (int i = 0; i < ROWA; i++) {
		for (int j = 0; j < COLUMNA_ROWB; j++) {
			A[i * COLUMNA_ROWB + j] = (rand() % 10000 + 1) * 0.01f;
		}
	}

	for (int i = 0; i < COLUMNA_ROWB; i++) {
		for (int j = 0; j < COLUMNB; j++) {
			B[i * COLUMNB + j] = (rand() % 10000 + 1) * 0.01f;
		}
	}

	// Create kernel
	cl_kernel kernel;
	cl_mem bufA, bufB, bufC;

	kernel = clCreateKernel(program, "mat_mul_seq", &err);
	CHECK_ERROR(err);

	// Create Buffer
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROWA * COLUMNA_ROWB, NULL, &err);
	CHECK_ERROR(err);

	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * COLUMNA_ROWB * COLUMNB, NULL, &err);
	CHECK_ERROR(err);

	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * ROWA * COLUMNB, NULL, &err);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(float) * ROWA * COLUMNA_ROWB, A, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(float) * COLUMNA_ROWB * COLUMNB, B, 0, NULL, NULL);
	CHECK_ERROR(err);

	clock_t start;
	clock_t end;

	start = clock();
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	CHECK_ERROR(err);

	int row = ROWA;
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &row);
	CHECK_ERROR(err);

	int columnA = COLUMNA_ROWB;
	err = clSetKernelArg(kernel, 4, sizeof(cl_int), &columnA);
	CHECK_ERROR(err);


	int columnB = COLUMNB;
	err = clSetKernelArg(kernel, 5, sizeof(cl_int), &columnB);
	CHECK_ERROR(err);
	end = clock();

	printf("Send Vector A, B to GPU : %lf seconds elapsed\n", (double)end - start);


	start = clock();
	// Execute Kernel
	size_t global_size[2] = { COLUMNB, ROWA };
	size_t local_size[2] = { 16, 16 };
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	end = clock();

	printf("Calculate C : %lf seconds elapsed\n", (double)end - start);

	// Read Buffer
	start = clock();
	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * ROWA * COLUMNB, C, 0, NULL, NULL);
	CHECK_ERROR(err);

	end = clock();
	printf("Receive C from GPU : %lf seconds elapsed\n", (double)end - start);

	for (int i = 0; i < ROWA; i++) {
		for (int j = 0; j < COLUMNB; j++) {
			checkC[i * COLUMNB + j] = 0.0f;
			for (int k = 0; k < COLUMNA_ROWB; k++) {
				checkC[i * COLUMNB + j] += A[i * COLUMNA_ROWB + k] * B[k * COLUMNB + j];
			}
		}
	}

	for (int i = 0; i < ROWA; i++) {
		for (int j = 0; j < COLUMNB; j++) {
			if (checkC[i * COLUMNB + j] != C[i * COLUMNB + j]) {
				printf("%lf :  %lf  zz ", checkC[i * COLUMNB + j], C[i * COLUMNB + j]);
				printf("Verification failed! row: %d, columne %d\n", i + 1, j + 1);
				break;
			}
		}
	}

	err = clReleaseMemObject(bufA);
	err = clReleaseMemObject(bufB);
	err = clReleaseMemObject(bufC);
	free(A);
	free(B);
	free(C);
	free(checkC);
	err = clReleaseKernel(kernel);
	err = clReleaseProgram(program);
	err = clReleaseCommandQueue(queue);
	err = clReleaseContext(context);

	free(kernel_source);

}