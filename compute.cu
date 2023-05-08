#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

// parallel matrix construction  -- do in main to pass values to other functions easier
// __gloabal__ void pMatrixConstruct()


// parallel compute acceleration matrix
__global__ void pComputation(vector3 *hPos, vector3 *accels, double *mass){
	int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
	int ind = (NUMENTITIES * row) + col;

	int i = row;
	int j = col;
	if (ind < NUMENTITIES * NUMENTITIES){
		if (i == j) {
			FILL_VECTOR(accels[ind], 0, 0, 0);
		} else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPos[i][k] - hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[ind], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	}
}


// Parallel matrix sum and update
__global__ void pSum(vector3 *accels, vector3 *accel_sum, vector3 *hPos, vector3 *hVel){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int i = row;
	if (i < NUMENTITIES){
		FILL_VECTOR(accel_sum[i], 0, 0, 0);
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0;k < 3; k++) {
				accel_sum[i][k] += accels[i * NUMENTITIES+ j][k];
			}
		}
		// updated vel and pos
		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[i][k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}
	}

}


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	vector3 *dev_hPos, *dev_hVel, *dev_acc, *dev_sum;
	double *dev_mass;

	// create matrixes for calculation and division of work
	int blocksD = ceilf( NUMENTITIES / 16.0f  );
	int threadsD = ceilf( NUMENTITIES / (float)blocksD );
	dim3 gridDim(blocksD, blocksD, 1);
	dim3 blockDim(threadsD, threadsD, 1);
	//dim3 bDim(16,16);                            // threads in block
	//dim3 gDim((NUMENTITIES +bDim.x - 1) / bDim.x, (NUMENTITIES + bDim.y - 1) / bDim.y);  // blocks

	// allocate gpu variables
	cudaMalloc((void**) &dev_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dev_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dev_acc, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dev_sum, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dev_mass, sizeof(double) * NUMENTITIES);

	// copy variables to gpu
	cudaMemcpy(dev_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	// compute accelerations
	pComputation<<<gridDim, blockDim>>>(dev_hPos, dev_acc, dev_mass);
	cudaDeviceSynchronize();

	// sum matrices and update values
	pSum<<<gridDim.x, blockDim.x>>>(dev_acc, dev_sum, dev_hPos, dev_hVel);
	cudaDeviceSynchronize();

	// copy gpu results back to host
	cudaMemcpy(hPos, dev_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dev_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	// cleanup gpu memory
	cudaFree(dev_hPos);
	cudaFree(dev_hVel);
	cudaFree(dev_mass);
	cudaFree(dev_acc);
}
// OG compute:
//	//make an acceleration matrix which is NUMENTITIES squared in size;
//	int i,j,k;
//	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
//	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
//	for (i=0;i<NUMENTITIES;i++)
//		accels[i]=&values[i*NUMENTITIES];
//	//first compute the pairwise accelerations.  Effect is on the first argument.
//	for (i=0;i<NUMENTITIES;i++){
//		for (j=0;j<NUMENTITIES;j++){
//			if (i==j) {
//				FILL_VECTOR(accels[i][j],0,0,0);
//			}
//			else{
//				vector3 distance;
//				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
//				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
//				double magnitude=sqrt(magnitude_sq);
//				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
//				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
//			}
//		}
//	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
//	for (i=0;i<NUMENTITIES;i++){
//		vector3 accel_sum={0,0,0};
//		for (j=0;j<NUMENTITIES;j++){
//			for (k=0;k<3;k++)
//				accel_sum[k]+=accels[i][j][k];
//		}
//		//compute the new velocity based on the acceleration and time interval
//		//compute the new position based on the velocity and time interval
//		for (k=0;k<3;k++){
//			hVel[i][k]+=accel_sum[k]*INTERVAL;
//			hPos[i][k]=hVel[i][k]*INTERVAL;
//		}
//	}
//	free(accels);
//	free(values);
//}
