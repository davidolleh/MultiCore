__kernel void mat_mul_seq(
	__global float *A, 
	__global float *B, 
	__global float *C,
	int ROWA, // m
	int COLAROWB,// k
	int COLUMNB // n
) {
	int column = get_global_id(0);
	int row = get_global_id(1);

	float intermediateValue = 0.0f;
	for (int k = 0; k < COLAROWB; k++) {
		intermediateValue += A[row* COLAROWB + k] * B[k * COLUMNB + column];
	}

	C[row * COLUMNB + column] = intermediateValue;
}