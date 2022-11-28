#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 4
#define P 5

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

/*
_global_ void CudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {}

_global_ void CudaMatrixMult(float *M1, float *M2, float *Mout, int n) {}
*/

int main() {
    //Initialistions des paramètres
    float *M1, *M2, *M3;
    
    
    //Allocution de mémoire pour une matrice 
    M1 = (float*)malloc(sizeof(float) * N * P) ;
    M2 = (float*)malloc(sizeof(float) * N * P) ; 
    M3 = (float*)malloc(sizeof(float) * N * P) ; 

    //Process
    MatrixInit(M1, N, P) ;
    MatrixInit(M2, N, P) ;
    MatrixAdd(M1, M2, M3, N, P) ;

    MatrixPrint(M1, N, P) ;
    MatrixPrint(M2, N, P) ;
    MatrixPrint(M3, N, P) ;


    //Libération de la mémoire

    free(M1);
    free(M2);
    free(M3);
/*   
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    MatrixInit(*d_out, 10, 4) ;
    MatrixPrint(*d_out, 10, 4) ;

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);

*/
}