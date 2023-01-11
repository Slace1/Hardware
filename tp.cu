#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// ------------------------------- définition des fonctions ----------------------------------------------
// Langage C pour CPU ----------------------
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
void MatrixAdd(float *M1, float *M2, float *M_sortie, int n, int p) {
    for(int i = 0; i < n*p; i++){ //Parcours toutes les valeurs de la matrice
        M_sortie[i] = M1[i] + M2[i];
    }
}

//Multiplication de matrices 
void MatrixMult(float *M1, float *M2, float *M_sortie, int n) {
    for (int i=0 ; i<n ; i++){
        for (int j=0 ; j<n ; j++){
            float s = 0 ;
            for (int k=0 ; k<n ; k++){
                s = s + M1[i*n + k]*M2[k*n + j] ;
            }
            M_sortie[i*n + j] = s ;
        }
    }
}

//Langage CUDA pour GPU -------------------
//Addition de matrices 
__global__ void cudaMatrixAdd(float *M1, float *M2, float *M_sortie, int n, int p) { 
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (line<n && row<p){
        M_sortie[line*n + row] = M1[line*n + row] + M2[line*n + row];
    }
}

// Addition appelé sur le GPU et executé sur le GPU 
__device__ float* cudaMatrixAddGB(float *M1, float *M2, float *M_sortie, int n, int p) { 

    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (line<n && row<p){
        M_sortie[line*n + row] = M1[line*n + row] + M2[line*n + row];
    }
    return M_sortie; 
}

//Multiplication de matrices sur GPU avec appel du CPU
__global__ void CudaMatrixMult(float *M1, float *M2, float *M_sortie, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0;

    if (row < n && line < n) {

        for (int i = 0; i < n; i++) {
            s += M1[row * n + i] * M2[i * n + line];
    }

    M_sortie[row * n + line] = s;
  }
}

//Multiplication de matrices sur GPU avec appel du GPU
__device__ float* CudaMatrixMultNP(float *M1, float *M2, float *M_sortie, int n, int p, int m){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0;

    if (row < n && line < n) {

        for (int i = 0; i < n; i++) {
            s += M1[row * n + i] * M2[i * n + line];
    }

    M_sortie[row * n + line] = s;
  }
  return M_sortie; 
}

// Définition des différentes couches ----------

//Convolution2D
__global__ void Conv2D(float* M, float* kernel, float* M_sortie, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Msortie_ligne, int Msortie_colonne){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int line = blockIdx.x * blockDim.x + threadIdx.x;

    float s;

    if (row < Msortie_ligne && line < Msortie_colonne){
        
        int k_k = kernel_size * kernel_size;
        int size_M = Msortie_ligne * Msortie_colonne;
        
        for (int n_k = 0; n_k < nb_kernel; n_k++){
            s = 0.0;
            
            for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    
                    s += M[(row + kernel_lig) * M_colonne + (line + kernel_col)] * kernel[kernel_lig * kernel_size + kernel_col + n_k * k_k];
                    
                }
            }
            
            M_sortie[row * Msortie_colonne + line + n_k * size_M] = s;
        }
    }
}

//Sous-echantillonnage
__global__ void MeanPool(float* M, float* M_sortie, int M_ligne, int M_colonne, int M_prof, int meanpool_size, int Msortie_ligne, int Msortie_colonne){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int line = blockIdx.x * blockDim.x + threadIdx.x;

    if (row % meanpool_size == 0 && line % meanpool_size == 0){
        
        float s;
        int meanpool_size_2 = meanpool_size * meanpool_size;
        int tot_M = M_ligne * M_colonne;
        int size_M = Msortie_ligne * Msortie_colonne;
        
        for (int n_profondeur = 0; n_profondeur < M_prof; n_profondeur++){
            s = 0.0;
            
            for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
                for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                    s += M[(row + meanpool_lig) * M_colonne + line + meanpool_col + n_profondeur * tot_M] / meanpool_size_2;
            
                }
            }
            if (row == 0){
                M_sortie[row * Msortie_colonne + (line / meanpool_size) + n_profondeur * size_M] = s;
            }
            else if (line == 0){
                M_sortie[(row / meanpool_size) * Msortie_colonne + line + n_profondeur * size_M] = s;
            }
            else{
                M_sortie[(row / meanpool_size) * Msortie_colonne + (line / meanpool_size) + n_profondeur * size_M] = s;
            }
        }
    }
}

//Fonction d'activation
__device__ float* Tanh(float* M, int M_ligne, int M_colonne, int M_prof){
    
    int line = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (line < M_ligne && row < M_colonne){
        
        int tot_M = M_ligne * M_colonne;
        
        for (int n_profondeur = 0; n_profondeur < M_prof; n_profondeur++){
            M[line * M_colonne + row + n_profondeur * tot_M] = tanh(M[line * M_colonne + row + n_profondeur * tot_M]);
        }
            
    }
            
    return M;
}

// Couche Linéaire M_sortie = W * M + b
__global__ void cudaDense(float* d_M, float* d_Mout, float* d_W, float* d_b, int n, int p, int m){

    d_Mout = CudaMatrixMultNP(d_M, d_W, d_Mout, n, p, m);
    d_Mout = cudaMatrixAddGB(d_Mout, d_b, d_Mout, n, m);
    
}

//Pour appeler la fonction device depuis le GPU 
__global__ void kernelT(float* M, int M_ligne, int M_colonne, int M_prof){
    Tanh(M, M_ligne, M_colonne, M_prof) ;
}


//Fonction principale
int main(int argc, char* argv[]) {
    //Initialistions des paramètres
    float *M1, *M2, *M3, *M4, *M5, *M6 ;
    float *raw_data, *C1_data, *S1_data, *C1_kernel,*C2_data, *S2_data, *W_data, *B_data, *Out_data;
    float *d_M1, *d_M2, *d_M4, *d_M6 ;
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel,*d_C2_data, *d_S2_data, *d_W_data, *d_B_data, *d_Out_data;


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
    C2_data=(float*)malloc(sizeof(float) * 10 * 10 * 16) ; 
    S2_data=(float*)malloc(sizeof(float) * 5 * 5 * 16 ) ;
    W_data=(float*)malloc(sizeof(float) * 400 * 120) ; 
    B_data=(float*)malloc(sizeof(float) * 120  ) ;
    Out_data=(float*)malloc(sizeof(float) * 400  ) ;

    //Allocution de mémoire pour une matrice sur le GPU
    cudaMalloc((void**)&d_M1, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M2, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M4, sizeof(float) * N * P);
    cudaMalloc((void**)&d_M6, sizeof(float) * N * P);
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32* 1);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6 );
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C2_data, sizeof(float) * 10 * 10 * 16);
    cudaMalloc((void**)&d_S2_data, sizeof(float) * 5 * 5  * 16 );
    cudaMalloc((void**)&d_W_data, sizeof(float) * 400 * 120);
    cudaMalloc((void**)&d_B_data, sizeof(float) * 120 );
    cudaMalloc((void**)&d_Out_data, sizeof(float) * 400 );
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
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_data, C2_data, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_data, W_data, sizeof(float) * 400 * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Out_data, Out_data, sizeof(float) * 400 , cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_data, B_data, sizeof(float) * 120 , cudaMemcpyHostToDevice);

    // Configurer les paramètres de la convolution
    dim3 dimGrid (32,32) ;
    dim3 blockGrid (1,1) ;
    

    //--------------CPU--------------
    if(strcmp(argv[1],"cpu") == 0){
        printf("CPU\n");
        MatrixAdd(M1, M2, M3, N, P) ; //Addition de matrices sur le CPU
        MatrixMult(M1, M2, M5, N); //Multiplications de matrices sur le CPU
    }
    
    
    //--------------GPU--------------
    if(strcmp(argv[1],"gpu")==0){

        // Partie 1------------------
        
        printf("GPU\n");
        cudaMatrixAdd<<<dimGrid,blockGrid>>>(d_M1, d_M2, d_M4, N, P) ; //Addition de matrics sur le GPU
        CudaMatrixMult<<<dimGrid,blockGrid>>>(d_M1, d_M2, d_M6, N) ;
        
        //Partie 2-------------
        
        // Exécuter la convolution sur le GPU
        //Conv2D<<<blockGrid,dimGrid>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
        //kernelT<<<blockGrid,dimGrid>>>(d_C1_data, 28, 28, 6);
        //MeanPool<<<blockGrid,dimGrid>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
        //cudaDeviceSynchronize();

        //Partie 3 ------------
        //layers
        Conv2D<<<blockGrid,dimGrid>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
        cudaDeviceSynchronize();

        kernelT<<<blockGrid,dimGrid>>>(d_C1_data, 28, 28, 6);
        cudaDeviceSynchronize();

        MeanPool<<<blockGrid,dimGrid>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
        cudaDeviceSynchronize();

        Conv2D<<<blockGrid,dimGrid>>>(d_S1_data, d_C1_kernel, d_C2_data, 14, 14, 6, 16, 10, 10);
        cudaDeviceSynchronize();

        kernelT<<<blockGrid,dimGrid>>>(d_C2_data, 10, 10, 16);
        cudaDeviceSynchronize();

        MeanPool<<<blockGrid,dimGrid>>>(d_C2_data, d_S2_data, 10, 10, 16, 2, 5, 5);
        cudaDeviceSynchronize();
        
        cudaDense<<<blockGrid,dimGrid>>>(d_C2_data, d_Out_data, d_W_data, d_B_data, 1, 400, 120);
        cudaDeviceSynchronize();
    }

    //Copie des données du GPU vers le CPU (local) 
    cudaMemcpy(M4, d_M4, sizeof(float) * N * P, cudaMemcpyDeviceToHost) ;
    cudaMemcpy(M6, d_M6, sizeof(float) * N * P, cudaMemcpyDeviceToHost) ;

    // Récupérer le résultat de la convolution sur le CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 6 * 28 * 28, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 6 * 14 * 14, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * 16 * 10 * 10, cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * 16 * 5 * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(W_data, d_W_data, sizeof(float) * 400 * 120, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_data, d_B_data, sizeof(float) * 120, cudaMemcpyDeviceToHost);
    cudaMemcpy(Out_data, d_Out_data, sizeof(float) * 400, cudaMemcpyDeviceToHost);
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
    //Affichage
    //MatrixPrint(M1, N, P) ;
    //MatrixPrint(M2, N, P) ;
    //MatrixPrint(M3, N, P) ;
    //MatrixPrint(M4, N, P) ;
    //MatrixPrint(M5, N, P) ;
    //MatrixPrint(M6, N, P) ;
    MatrixPrint(raw_data, 32, 32) ;
    MatrixPrint(C1_kernel, 5, 5) ;
    MatrixPrint(C1_data, 28, 28) ;
    MatrixPrint(S1_data, 14, 14) ;
    MatrixPrint(C2_data, 10, 10) ;
    MatrixPrint(S2_data, 5, 5) ;

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
    free(C2_data);
    free(S2_data);
    free(W_data);
    free(B_data);
    free(Out_data);

    //Libération de la mémoire du GPU
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_M4);
    cudaFree(d_M6);
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C2_data);
    cudaFree(d_S2_data);
    cudaFree(d_W_data);
    cudaFree(d_B_data);
    cudaFree(d_Out_data);
}