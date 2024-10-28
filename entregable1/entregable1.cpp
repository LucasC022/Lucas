/*
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 * Curso 2024/25
 *
 * BASICO 4 : "Sincronización"
 * >> Hacer que los hilos de ejecución de un kernel trabajen de forma cooperativa
 *
 * AUTOR: Dominguez Simon Aitor
 * FECHA: 20/10/2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define FILAS 7
#define COLUMNAS 25

 // Kernel para desplazar las filas de la matriz una posición hacia abajo de forma cíclica
__global__ void desplazarFilas(int* matriz, int filas, int columnas) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < columnas) {
        int ultimo_valor = matriz[(filas - 1) * columnas + col];

        for (int fil = filas - 1; fil > 0; fil--) {
            matriz[fil * columnas + col] = matriz[(fil - 1) * columnas + col];
        }

        matriz[col] = ultimo_valor;
    }
}

// Función para obtener propiedades del dispositivo CUDA
__host__ void propiedades_Device(int deviceID) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);

    int cudaCores;
    int SM = deviceProp.multiProcessorCount;
    int major = deviceProp.major;
    int minor = deviceProp.minor;
    const char* archName;

    switch (major) {
    case 1: archName = "TESLA"; cudaCores = 8; break;
    case 2: archName = "FERMI"; cudaCores = (minor == 0) ? 32 : 48; break;
    case 3: archName = "KEPLER"; cudaCores = 192; break;
    case 5: archName = "MAXWELL"; cudaCores = 128; break;
    case 6: archName = "PASCAL"; cudaCores = 64; break;
    case 7: archName = (minor == 0) ? "VOLTA" : "TURING"; cudaCores = 64; break;
    case 8: archName = "AMPERE"; cudaCores = 64; break;
    case 9: archName = "HOPPER"; cudaCores = 128; break;
    default: archName = "DESCONOCIDA"; cudaCores = 0; break;
    }

    int totalCores = SM * cudaCores;

    printf("***************************************************\n");
    printf("DEVICE 0: %s\n", deviceProp.name);
    printf("***************************************************\n");
    printf("CUDA Toolkit                          : 8.0\n");
    printf("Capacidad de Computo                  : %d.%d\n", major, minor);
    printf("Arquitectura CUDA                     : %s\n", archName);
    printf("No. de MultiProcesadores              : %d\n", SM);
    printf("No. de CUDA Cores (%dx%d)             : %d\n", cudaCores, SM, totalCores);
    printf("No. maximo de Hilos (por bloque)      : %d\n", deviceProp.maxThreadsPerBlock);
    printf("Memoria Global (total)                : %zu MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    printf("***************************************************\n");
}

int main() {
    propiedades_Device(0);

    dim3 hilos(25);
    dim3 bloques((COLUMNAS + hilos.x - 1) / hilos.x);

    int h_matriz[FILAS * COLUMNAS];
    for (int i = 0; i < FILAS; i++) {
        int valor_fila = i + 1;
        for (int j = 0; j < COLUMNAS; j++) {
            h_matriz[i * COLUMNAS + j] = valor_fila;
        }
    }

    int* d_matriz;
    size_t size = FILAS * COLUMNAS * sizeof(int);
    cudaMalloc(&d_matriz, size);
    cudaMemcpy(d_matriz, h_matriz, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    desplazarFilas << <bloques, hilos >> > (d_matriz, FILAS, COLUMNAS);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float tiempo;
    cudaEventElapsedTime(&tiempo, start, stop);

    cudaMemcpy(h_matriz, d_matriz, size, cudaMemcpyDeviceToHost);

    cudaFree(d_matriz);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n***************************************************\n");
    printf("KERNEL de 1 bloque con %d HILOS:\n", bloques.x * hilos.x);
    printf(" eje x -> %d hilos\n", hilos.x);
    printf(" eje y -> %d hilos\n", hilos.y);
    printf("Tiempo de ejecucion: %f ms\n", tiempo);
    printf("***************************************************\n");

    printf("\nMATRIZ ORIGINAL:\n");
    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLUMNAS; j++) {
            printf("%d ", i + 1);
        }
        printf("\n");
    }

    printf("\nMATRIZ FINAL:\n");
    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLUMNAS; j++) {
            printf("%d ", h_matriz[i * COLUMNAS + j]);
        }
        printf("\n");
    }
    getchar();
    return 0;
}
