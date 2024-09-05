#include <math.h>
#include <stdio.h>
#include <cblas.h>

extern "C"
{
    void matrix_multiply(int M, int K, int N, double* X, double* Y, double* Z) {
        for (int i = 0; i < M; i++) {          
            for (int j = 0; j < N; j++) {      
                Z[i * N + j] = 0.0;            
                for (int k = 0; k < K; k++) {  
                    Z[i * N + j] += X[i * K + k] * Y[k * N + j];
                }
            }
        }
    }

    void matrix_multiply_blas(int M, int K, int N, double* X, double* Y, double* Z) {
        // M: ROW OF A
        // N: COL OF B
        // K: COL OF A/ ROW OF B
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, X, K, Y, N, 0.0, Z, N);
    }

    void compute_A_matrix(double *A, const double *v, const double *theta, const double *v_dot, 
                            int horizon, const double Ts)
    {
        int Dx = 4;
        for (int i = 0; i < horizon; ++i)
        {
            A[(0 * Dx + 0) * horizon + i] = 1;
            A[(0 * Dx + 2) * horizon + i] = cos(theta[i])*Ts;
            A[(0 * Dx + 3) * horizon + i] = -(v[i]*Ts + (v_dot[i]*pow(Ts, 2))/2)*sin(theta[i]);
            A[(1 * Dx + 2) * horizon + i] = sin(theta[i])*Ts;
            A[(1 * Dx + 3) * horizon + i] = (v[i]*Ts + (v_dot[i]*pow(Ts, 2))/2)*cos(theta[i]);
            A[(1 * Dx + 1) * horizon + i] = 1;
            A[(2 * Dx + 2) * horizon + i] = 1;
            A[(3 * Dx + 3) * horizon + i] = 1;
        }
    }
    
    void compute_B_matrix(double* B, const double *theta, 
                            int horizon, const double Ts) 
    {
        int Du = 2;

        for (int i = 0; i < horizon; ++i) {
            B[(0 * Du + 0) * horizon + i] = pow(Ts, 2)*cos(theta[i])/2;
            B[(1 * Du + 0) * horizon + i] = pow(Ts, 2)*sin(theta[i])/2;
            B[(2 * Du + 0) * horizon + i] = Ts; 
            B[(3 * Du + 1) * horizon + i] = Ts; 
        }
    }

    void compute_next_state(double *x_next, 
                            const double *px, const double *py, const double *v, const double *theta,
                            const double *a, const double *theta_dot,
                            int horizon, const double Ts)

    {
        double A[4*4]={0}; double B[4*2]={0};
        double x_cur[4] = {*px, *py, *v, *theta};
        double u_cur[2] = {*a, *theta_dot};
        compute_A_matrix(A, v, theta, a, horizon, Ts);
        compute_B_matrix(B, theta, horizon, Ts);

        double Ax_cur[4]={0};
        matrix_multiply(4, 4, 1, A, x_cur, Ax_cur);  // 4x4 * 4x1 = 4x1
        double Bu_cur[4]={0};
        matrix_multiply(4, 2, 1, B, u_cur, Bu_cur);  // 4x2 * 2x1 = 4x1

        for (int i = 0; i < 4; ++i) {
            x_next[i] = Ax_cur[i] + Bu_cur[i];
        }
    }
}