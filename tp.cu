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

//Affichage d'une matrice 
void MatrixPrint(float *M, int n, int p) {
    for (int i=0 ; i<n ; i++){ //Sur les lignes 
        for (int j=0 ; j<p ; j++){ //Sur les colonnes
            printf("%f     ", M[i*p + j]) ;
        }
        printf("\n") ;
    }
    printf("\n");
}
//Addition de matrices
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

//Multiplication de matrices 
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
__global__ void Conv2D(int M_ligne, int M_colonne, float* M, int kernel_size, int nb_kernel, float* kernel, int Mout_ligne, int Mout_colonne, float* Mout) {
    
    //Calcul du thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= Mout_ligne || line>= Mout_colonne) {
        return;
    }

    // Initialisation
    float value = 0;

    // Boucle sur le thread
    for (int k = 0; k < nb_kernel; k++) {
        for (int i = -kernel_size/2; i <= kernel_size/2; i++) {
            for (int j = -kernel_size/2; j <= kernel_size/2; j++) {
                // Indices de la matrice d'entrée
                int m_row = row + i;
                int m_line = line + j;
                if (m_row >= 0 && m_row < M_ligne && m_line >= 0 && m_line < M_colonne) {
                    //Convolue le kernel associé
                    value += M[m_row * M_colonne + m_ligne] * kernel[k * kernel_size * kernel_size + (i + kernel_size/2) * kernel_size + (j + kernel_size/2)];
                }
            }
        }
    }

    // Copie de la valeur de sortie
    Mout[row * Mout_colonne + ligne] = value;
}

//Sous-echantillonnage
__global__ void MeanPool(int M_ligne, int M_colonne, int M_prof, float* M, int meanpool_size, int Mout_ligne, int Mout_colonne, float* Mout) {
    // Calculate the indices of the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int line = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is within the bounds of the output matrix
    if (row >= Mout_ligne || line >= Mout_colonne) {
        return;
    }

    // Initialize the output value for the current thread
    float value = 0;

    // Perform the mean pooling for the current thread
    for (int k = 0; k < M_prof; k++) {
        for (int i = 0; i < meanpool_size; i++) {
            for (int j = 0; j < meanpool_size; j++) {
                // Calculate the indices of the input matrix
                int m_row = row * meanpool_size + i;
                int m_lin = line * meanpool_size + j;

                // Make sure the indices are within the bounds of the input matrix
                if (m_row >= 0 && m_row < M_ligne && m_lin >= 0 && m_lin < M_colonne) {
                    // Add the value to the output
                    value += M[k * M_ligne * M_colonne + m_row * M_colonne + m_lin];
                }
            }
        }
    }

    // Calculate the mean value and store it in the output matrix
    Mout[row * Mout_colonne + line] = value / (M_prof * meanpool_size * meanpool_size);
}

//Fonction d'activation
__device__ float* activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof){
    
    int lin = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lin < M_ligne && col < M_colonne){
        
        int tot_M = M_ligne * M_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            M[lin * M_colonne + row + n_prof * tot_M] = tanh(M[lin * M_colonne + row + n_prof * tot_M]);
        }
            
    }
            
    return M;
}


//Fonction principale
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
    raw_data=(float*)malloc(sizeof(float) * 32 * 32* 1) ;
    C1_data=(float*)malloc(sizeof(float) * 28 * 28 * 6) ; 
    S1_data=(float*)malloc(sizeof(float) * 14 * 14 * 6 ) ;
    C1_kernel=(float*)malloc(sizeof(float) * 5 * 5 * 6) ;

    //Allocution de mémoire pour une matrice sur le GPU
    cudaMalloc((void**)&d_M1, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M2, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M4, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M6, sizeof(float) * N * P);
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32* 1);
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
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);


    // Configurer les paramètres de la convolution
    dim3 dimGrid (N,P,1) ;
    dim3 blockGrid (1,1,1) ;
    

    //--------------CPU--------------
    if(strcmp(argv[1],"cpu") == 0){
        printf("CPU\n");
        MatrixAdd(M1, M2, M3, N, P) ; //Addition de matrices sur le CPU
        MatrixMult(M1, M2, M5, N); //Multiplications de matrices sur le CPU
    }
    
    
    //--------------GPU--------------
    if(strcmp(argv[1],"gpu")==0){
        printf("GPU\n");
        cudaMatrixAdd<<<dimGrid,blockGrid>>>(d_M1, d_M2, d_M4, N, P) ; //Addition de matrics sur le GPU
        CudaMatrixMult<<<dimGrid,blockGrid>>>(d_M1, d_M2, d_M6, N) ;
        // Exécuter la convolution sur le GPU
        Conv2D<<<blockGrid, dimGrid>>>(32, 32, d_raw_data, 5, 6, d_C1_kernel, 28, 28, d_C1_data);
        MeanPool<<<blockGrid, dimGrid>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
        Tanh<<<blockGrid, dimGrid>>>(d_C1_data, 28, 28, 6);  
        cudaDeviceSynchronize();
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