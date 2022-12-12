#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

//#define N 10
//#define P 10

//Initialisation d'une matrice 
void MatrixInit(float *M, int n, int p) {
    for(int i=0 ; i < n*p ; i++){
        M[i] = (float)rand() / (float)RAND_MAX    +   (rand() % 2) - 1; //Création d'un float entre - 1 et 1
    }
}


void MatrixPrint(float *M, int n, int p) {
    for (int i=0 ; i<n ; i++){ //Sur les lignes 
        for (int j=0 ; j<p ; j++){ //Sur les colonnes
            printf("%f     ", M[i*p + j]) ;
        }
        printf("\n") ;
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for(int i = 0; i < n*p; i++){ //Parcours toutes les valeurs de la matrice
        Mout[i] = M1[i] + M2[i];
    }
}


_global_ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Handling arbitrary vector size
    if (line<n && row<p){
        Mout[line*n + row] = M1[line*n + row] + M2[line*n + row];
    }
}


void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i=0 ; i<n ; i++){
        for (int j=0 ; j<n ; j++){
            float s = 0 ;
            for (int k=0 ; k<n ; k++){
                s = s + M1[i*n + k]*M2[k*n + j] ;
            }
            Mout[i*n + j] = s ;
        }
    }
}

_global_ void CudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;

    if (row < n && col < n) {

        for (int i = 0; i < n; i++) {
            value += M1[row * n + i] * M2[i * n + col];
    }

    Mout[row * n + col] = value;
  }
}

int main(int argc, char* argv[]) {
    //Initialistions des paramètres
    float *M1, *M2, *M3, *M4, *M5, *M6 ;
    float *raw_data, *C1_data, *S1_data, *C1_kernel ;
    float *d_M1, *d_M2, *d_M4, *d_M6 ;
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel ;


    if(argc!=4){
        return EXIT_FAILURE;
    }
    int N = atoi(argv[2]);
    int P = atoi(argv[3]);
    
    
    //Allocution de mémoire pour une matrice sur le CPU
    M1 = (float*)malloc(sizeof(float) * N * P) ;
    M2 = (float*)malloc(sizeof(float) * N * P) ; 
    M3 = (float*)malloc(sizeof(float) * N * P) ; 
    M4 = (float*)malloc(sizeof(float) * N * P) ;
    M5 = (float*)malloc(sizeof(float) * N * P) ;
    M6 = (float*)malloc(sizeof(float) * N * P) ;
    raw_data=(float*)malloc(sizeof(float) * 32 * 32) ;
    C1_data=(float*)malloc(sizeof(float) * 28 * 28 * 6) ; 
    S1_data=(float*)malloc(sizeof(float) * 14 * 14 * 6 ) ;
    C1_kernel=(float*)malloc(sizeof(float) * 5 * 5 * 6) ;

    //Allocution de mémoire pour une matrice sur le GPU
    cudaMalloc((void**)&d_M1, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M2, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M4, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M6, sizeof(float) * N * P);
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6 );
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

    //Process
    //Initialisation des matrices
    MatrixInit(M1, N, P) ;
    MatrixInit(M2, N, P) ;
    MatrixInit(raw_data, 32 , 32) ;
    MatrixInit(C1_kernel, 6 , 5*5) ;
    
    for(int i = 0; i < 32*32; i++){ //Initialisation des valeurs entre 0 et 1
        raw_data[i] = abs(raw_data[i]);
    }
    for(int i = 0; i < 32*32; i++){ //Initialisation des valeurs entre 0 et 1
        C1_kernel[i] = abs(C1_kernel[i]);
    }
    
    //Copie des données du CPU vers le GPU (device)
    cudaMemcpy(d_M1, M1, sizeof(float) * N * P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * N * P, cudaMemcpyHostToDevice);



    // Configurer les paramètres de la convolution
    dim3 dimGrid (N,P,1) ;
    dim3 blockGrid (1,1,1) ;
    
    int kernel_width = 5;
    int kernel_height = 5;
    int stride = 1;
    int padding = 0;

    if(strcmp(argv[1],"cpu") == 0){
        printf("CPU\n");
        MatrixAdd(M1, M2, M3, N, P) ; //Addition de matrices sur le CPU
        MatrixMult(M1, M2, M5, N); //Multiplications de matrices sur le CPU
    }
    
    if(strcmp(argv[1],"gpu")==0){
        printf("GPU\n");
        cudaMatrixAdd<<<dimGrid,blockGrid>>>(d_M1, d_M2, d_M4, N, P) ; //Addition de matrics sur le GPU
        CudaMatrixMult<<<dimGrid,blockGrid>>>(d_M1, d_M2, d_M6, N) ;
        // Exécuter la convolution sur le GPU
        conv2D<<<1, 6>>>(C1_data, raw_data, kernels, kernel_width, kernel_height, stride, padding);
    }

    //Copie des données du GPU vers le CPU (local) 
    cudaMemcpy(M4, d_M4, sizeof(float) * N * P, cudaMemcpyDeviceToHost) ;
    cudaMemcpy(M6, d_M6, sizeof(float) * N * P, cudaMemcpyDeviceToHost) ;
    // Récupérer le résultat de la convolution sur le CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float)*6 * 28 * 28, cudaMemcpyDeviceToHost);

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
    //Affichage
    MatrixPrint(M1, N, P) ;
    MatrixPrint(M2, N, P) ;
    MatrixPrint(M3, N, P) ;
    MatrixPrint(M4, N, P) ;
    MatrixPrint(M5, N, P) ;
    MatrixPrint(M6, N, P) ;
    MatrixPrint(raw_data, 32, 32) ;
    MatrixPrint(C1_data, 6, 28*28) ;
    MatrixPrint(S1_data, 6, 14*14) ;
    MatrixPrint(C1_kernel, 6, 5*5) ;

    //Libération de la mémoire du CPU

    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    //Libération de la mémoire du GPU
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_M4);
    cudaFree(d_M6);
    cudaFree(raw_data);
    cudaFree(C1_data);
    cudaFree(S1_data);
    cudaFree(C1_kernel);
}