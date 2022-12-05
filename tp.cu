#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 4
#define P 4

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


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for(int i = 0; i < n*p; i++){
        Mout[i] = M1[i] + M2[i];
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

__global__ void CudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
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


int main() {
    //Initialistions des paramètres
    float *M1, *M2, *M3, *M4, *M5, *M6;
    float *d_M1, *d_M2, *d_M4, *d_M6;
    
    
    //Allocution de mémoire pour une matrice sur le CPU
    M1 = (float*)malloc(sizeof(float) * N * P) ;
    M2 = (float*)malloc(sizeof(float) * N * P) ; 
    M3 = (float*)malloc(sizeof(float) * N * P) ; 
    M4 = (float*)malloc(sizeof(float) * N * P) ;
    M5 = (float*)malloc(sizeof(float) * N * P) ;
    M6 = (float*)malloc(sizeof(float) * N * P) ;

    //Allocution de mémoire pour une matrice sur le GPU
    cudaMalloc((void**)&d_M1, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M2, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M4, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M6, sizeof(float) * N * P);

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

    //Process
    //Initialisation des matrices
    MatrixInit(M1, N, P) ;
    MatrixInit(M2, N, P) ;

    //Copie des données du CPU vers le GPU (device)
    cudaMemcpy(d_M1, M1, sizeof(float) * N * P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * N * P, cudaMemcpyHostToDevice);

    MatrixAdd(M1, M2, M3, N, P) ; //Addition de matrices sur le CPU
    MatrixMult(M1, M2, M5, N); //Multiplications de matrices sur le CPU

    cudaMatrixAdd<<<1,1>>>(d_M1, d_M2, d_M4, N, P) ; //Addition de matrics sur le GPU
    CudaMatrixMult<<<1,1>>>(d_M1, d_M2, d_M6, N);

    //Copie des données du GPU vers le CPU (local) 
    cudaMemcpy(M4, d_M4, sizeof(float) * N * P, cudaMemcpyDeviceToHost) ;
    cudaMemcpy(M6, d_M6, sizeof(float) * N * P, cudaMemcpyDeviceToHost) ;

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
    //Affichage
    MatrixPrint(M1, N, P) ;
    MatrixPrint(M2, N, P) ;
    MatrixPrint(M3, N, P) ;
    MatrixPrint(M4, N, P) ;
    MatrixPrint(M5, N, P) ;
    MatrixPrint(M6, N, P) ;


    //Libération de la mémoire du CPU

    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);

    //Libération de la mémoire du GPU
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_M4);
    cudaFree(d_M6);

}