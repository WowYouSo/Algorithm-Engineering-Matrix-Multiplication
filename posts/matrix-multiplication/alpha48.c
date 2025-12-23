#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "bench.h"




static inline void alpha48_mul4x4_add(const double *A, int lda,
                                      const double *B, int ldb,
                                      double *C, int ldc)
{
    const double a11 = A[0*(size_t)lda + 0];
    const double a12 = A[0*(size_t)lda + 1];
    const double a13 = A[0*(size_t)lda + 2];
    const double a14 = A[0*(size_t)lda + 3];
    const double a21 = A[1*(size_t)lda + 0];
    const double a22 = A[1*(size_t)lda + 1];
    const double a23 = A[1*(size_t)lda + 2];
    const double a24 = A[1*(size_t)lda + 3];
    const double a31 = A[2*(size_t)lda + 0];
    const double a32 = A[2*(size_t)lda + 1];
    const double a33 = A[2*(size_t)lda + 2];
    const double a34 = A[2*(size_t)lda + 3];
    const double a41 = A[3*(size_t)lda + 0];
    const double a42 = A[3*(size_t)lda + 1];
    const double a43 = A[3*(size_t)lda + 2];
    const double a44 = A[3*(size_t)lda + 3];

    const double b11 = B[0*(size_t)ldb + 0];
    const double b12 = B[0*(size_t)ldb + 1];
    const double b13 = B[0*(size_t)ldb + 2];
    const double b14 = B[0*(size_t)ldb + 3];
    const double b21 = B[1*(size_t)ldb + 0];
    const double b22 = B[1*(size_t)ldb + 1];
    const double b23 = B[1*(size_t)ldb + 2];
    const double b24 = B[1*(size_t)ldb + 3];
    const double b31 = B[2*(size_t)ldb + 0];
    const double b32 = B[2*(size_t)ldb + 1];
    const double b33 = B[2*(size_t)ldb + 2];
    const double b34 = B[2*(size_t)ldb + 3];
    const double b41 = B[3*(size_t)ldb + 0];
    const double b42 = B[3*(size_t)ldb + 1];
    const double b43 = B[3*(size_t)ldb + 2];
    const double b44 = B[3*(size_t)ldb + 3];

    double s[48];
    double t[48];
    double p[48];

    {
        const double x16 = a13+a24;
        const double x17 = a11-a22;
        const double x18 = a34+a43;
        const double x19 = a31-a42;
        const double x20 = a32-a41;
        const double x21 = a14+a23;
        const double x22 = a33+a44;
        const double x23 = a12-a21;
        const double x24 = -a12-a22;
        const double x25 = a13-a23;
        const double x26 = a32-a42;
        const double x27 = -a34-a44;
        const double x28 = a33+a43;
        const double x29 = a14-a24;
        const double x30 = a31-a41;
        const double x31 = a11+a21;
        const double x32 = -x16-x21;
        const double x33 = x17+x23;
        const double x34 = x18-x22;
        const double x35 = x19-x20;
        const double x36 = x17-x23;
        const double x37 = x16-x21;
        const double x38 = x19+x20;
        const double x39 = x18+x22;
        const double x40 = x26-x30;
        const double x41 = x31-x24;
        const double x42 = x27+x28;
        const double x43 = x25+x29;
        const double x44 = -x40-x32;
        const double s24 = x33+x42;
        const double x46 = a32+a42;
        const double s8 = x35+x43;
        const double s34 = x41-x34;
        const double x49 = a34-a44;
        const double x50 = a31+a41;
        const double x51 = a33-a43;
        const double x52 = a14+a24;
        const double x53 = a13+a23;
        const double x54 = a12-a22;
        const double x55 = a11-a21;
        const double s9 = x30+x31;
        const double s12 = x25+x28;
        const double x58 = x19+x22;
        const double x59 = x35-x32;
        const double x62 = -x38-x37;
        const double x63 = x51+x36+x49;
        const double x64 = x16+x17;
        const double s42 = x30-x31;
        const double x66 = x21+x23;
        const double s36 = x24-x26;
        const double s22 = x27-x29;
        const double x71 = x18+x20;
        const double s38 = x26+x24;
        const double x73 = x38-x37;
        const double x74 = x50+x37+x46;
        const double x75 = x39-x36;
        const double x76 = x33-x34;
        const double x77 = x32+x35;
        const double x78 = x33+x34;
        const double x79 = x53+x38-x52;
        const double x80 = x36+x39;
        const double x81 = x39+x55-x54;
        const double s2 = x25-x28;
        const double s6 = x29+x27;
        const double s0 = x44-x63;
        const double s1 = x30-x55;
        const double s3 = x32+x42;
        const double s4 = -x74-s24;
        const double s5 = x76+x62;
        const double s7 = x62-x76;
        const double s10 = x73-x78;
        const double s11 = x66+x71;
        const double s13 = x53-x28;
        const double s14 = s42-s2;
        const double s15 = -x73-x78;
        const double s16 = x41-x35;
        const double s17 = x59-x80;
        const double s18 = x66-x71;
        const double s19 = x75+x77;
        const double s20 = x25-x51;
        const double s21 = x58+x64;
        const double s23 = s8+x81;
        const double s25 = x63+x44;
        const double s26 = x46-x24;
        const double s27 = -x44;
        const double s28 = x81-s8;
        const double s29 = x52-x27;
        const double s30 = x26+x54;
        const double s31 = x58-x64;
        const double s32 = x74-s24;
        const double s33 = x34+x43;
        const double s35 = x59+x80;
        const double s37 = -s22-s36;
        const double s39 = x33+x40;
        const double s40 = x77-x75;
        const double s41 = x29+x49;
        const double s43 = s9+s12;
        const double s44 = x79-s34;
        const double s45 = x79+s34;
        const double s46 = x50-x31;
        const double s47 = s6-s38;
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
        s[4] = s4;
        s[5] = s5;
        s[6] = s6;
        s[7] = s7;
        s[8] = s8;
        s[9] = s9;
        s[10] = s10;
        s[11] = s11;
        s[12] = s12;
        s[13] = s13;
        s[14] = s14;
        s[15] = s15;
        s[16] = s16;
        s[17] = s17;
        s[18] = s18;
        s[19] = s19;
        s[20] = s20;
        s[21] = s21;
        s[22] = s22;
        s[23] = s23;
        s[24] = s24;
        s[25] = s25;
        s[26] = s26;
        s[27] = s27;
        s[28] = s28;
        s[29] = s29;
        s[30] = s30;
        s[31] = s31;
        s[32] = s32;
        s[33] = s33;
        s[34] = s34;
        s[35] = s35;
        s[36] = s36;
        s[37] = s37;
        s[38] = s38;
        s[39] = s39;
        s[40] = s40;
        s[41] = s41;
        s[42] = s42;
        s[43] = s43;
        s[44] = s44;
        s[45] = s45;
        s[46] = s46;
        s[47] = s47;
    }

    {
        const double y16 = b11-b14;
        const double y17 = b21-b23;
        const double y18 = b31-b34;
        const double y19 = b41-b43;
        const double y20 = -b12-b13;
        const double t17 = y17+y19;
        const double y22 = b32+b33;
        const double t2 = b12+b32;
        const double t6 = b42-b22;
        const double t7 = y18+y16;
        const double t14 = b12-b31;
        const double t15 = y16-y18;
        const double t16 = y17+y20;
        const double t19 = y19-y17;
        const double t22 = t17-b24-b44;
        const double t33 = y18-b42-b44;
        const double t35 = y20-y22;
        const double t38 = b21-b41;
        const double t39 = -y16-b22-b24;
        const double t42 = b11+b31;
        const double t43 = b13-b34-y16-y22;
        const double t47 = b22+b41;
        const double v48 = t22-t6;
        const double v49 = t19-t35;
        const double v50 = t33-t39;
        const double v51 = t16+v49/2;
        const double v52 = v48/2-t47;
        const double v53 = t14+t43;
        const double v54 = -t33-t39;
        const double v55 = t14-t43;
        const double g56 = v54/2;
        const double v56 = v51+g56;
        const double g57 = v53/2;
        const double v57 = g57+v52;
        const double v58 = v51-g56;
        const double v59 = v52-g57;
        const double t24 = v48+v55;
        const double t3 = t16+v49;
        const double t8 = v48-v55;
        const double t21 = v50+v49;
        const double v65 = t7-t21/2;
        const double v66 = t2-t14;
        const double v67 = t15+v58;
        const double v68 = t47+t6;
        const double v69 = -t38-t47;
        const double v71 = t42+t14;
        const double v72 = v56-t17;
        const double g73 = t24/2;
        const double g74 = t8/2;
        const double g75 = (t35+v50+t19)/2;
        const double t13 = v66-t3;
        const double t11 = v58*2;
        const double t30 = v69-t39;
        const double t31 = v49-v50;
        const double t1 = t39+v71;
        const double t18 = v56*2;
        const double t25 = g75+v57;
        const double t20 = v66-t33;
        const double t4 = v65+g73;
        const double t46 = t16+v71;
        const double t29 = v68-t3;
        const double t5 = v54-t15;
        const double t10 = v50-t7;
        const double t12 = v55-t2;
        const double t34 = v59*2;
        const double t28 = v72+g74;
        const double t23 = g75-g74;
        const double t9 = t42+v53;
        const double t32 = v67-g73;
        const double t37 = v48-t47;
        const double t27 = v57*2;
        const double t45 = v65-v59;
        const double t44 = v67-v59;
        const double t26 = t16+v69;
        const double t36 = v52*2-t38;
        const double t0 = v57+v72;
        const double t41 = t33+v68;
        const double t40 = t17-v51*2;
        t[0] = t0;
        t[1] = t1;
        t[2] = t2;
        t[3] = t3;
        t[4] = t4;
        t[5] = t5;
        t[6] = t6;
        t[7] = t7;
        t[8] = t8;
        t[9] = t9;
        t[10] = t10;
        t[11] = t11;
        t[12] = t12;
        t[13] = t13;
        t[14] = t14;
        t[15] = t15;
        t[16] = t16;
        t[17] = t17;
        t[18] = t18;
        t[19] = t19;
        t[20] = t20;
        t[21] = t21;
        t[22] = t22;
        t[23] = t23;
        t[24] = t24;
        t[25] = t25;
        t[26] = t26;
        t[27] = t27;
        t[28] = t28;
        t[29] = t29;
        t[30] = t30;
        t[31] = t31;
        t[32] = t32;
        t[33] = t33;
        t[34] = t34;
        t[35] = t35;
        t[36] = t36;
        t[37] = t37;
        t[38] = t38;
        t[39] = t39;
        t[40] = t40;
        t[41] = t41;
        t[42] = t42;
        t[43] = t43;
        t[44] = t44;
        t[45] = t45;
        t[46] = t46;
        t[47] = t47;
    }

    for (int i = 0; i < 48; ++i) {
        p[i] = s[i] * t[i];
    }

    {
        const double p0 = p[0];
        const double p1 = p[1];
        const double p2 = p[2];
        const double p3 = p[3];
        const double p4 = p[4];
        const double p5 = p[5];
        const double p6 = p[6];
        const double p7 = p[7];
        const double p8 = p[8];
        const double p9 = p[9];
        const double p10 = p[10];
        const double p11 = p[11];
        const double p12 = p[12];
        const double p13 = p[13];
        const double p14 = p[14];
        const double p15 = p[15];
        const double p16 = p[16];
        const double p17 = p[17];
        const double p18 = p[18];
        const double p19 = p[19];
        const double p20 = p[20];
        const double p21 = p[21];
        const double p22 = p[22];
        const double p23 = p[23];
        const double p24 = p[24];
        const double p25 = p[25];
        const double p26 = p[26];
        const double p27 = p[27];
        const double p28 = p[28];
        const double p29 = p[29];
        const double p30 = p[30];
        const double p31 = p[31];
        const double p32 = p[32];
        const double p33 = p[33];
        const double p34 = p[34];
        const double p35 = p[35];
        const double p36 = p[36];
        const double p37 = p[37];
        const double p38 = p[38];
        const double p39 = p[39];
        const double p40 = p[40];
        const double p41 = p[41];
        const double p42 = p[42];
        const double p43 = p[43];
        const double p44 = p[44];
        const double p45 = p[45];
        const double p46 = p[46];
        const double p47 = p[47];
        const double e5 = p20-p41+p33;
        const double v61 = p34-p44;
        const double e7 = p9*2-p40+p11;
        const double e8 = p26-p46-p16;
        const double e12 = p30+p39+p1;
        const double e15 = p24-p32;
        const double v62 = p0+p27;
        const double v71 = (p46-p13)*2;
        const double v70 = (p0+p44)/4;
        const double v69 = (p32-p28)/4;
        const double v68 = (p41-p30)*2;
        const double v65 = (p20-p1)*2;
        const double v63 = (p29-p26)*2;
        const double e28 = v61-p18;
        const double e29 = v61-p15*2-e15;
        const double v54 = e8+e12;
        const double v51 = v69+(e12-p47-p14-e8)/2;
        const double e35 = p36*2+p5-p15+v62+e28+e7;
        const double e38 = (p43-p9+p12)*2+v54;
        const double e39 = p43+p37+v54;
        const double e40 = p36-p9+e39;
        const double e43 = (p14-p47-e38)/2-v70-e5;
        const double z21 = p21+p45+p11-p4+v68+v63+e29;
        const double z8 = p8+p28+v62+e29+(e7-e39)*2;
        const double z6 = p6+v69-e43;
        const double z3 = p3+e5-p29-p13+e38;
        const double z2 = p2+v69+e43;
        const double w43 = (p42+v70-v51)/2;
        const double w42 = (p38+v70+v51)/2;
        const double w41 = (p31+p45+p4+p5*2+v71+e15+v65+e28)/8;
        const double w40 = (p25+p45-(p9+p36)*2)/4;
        const double w39 = (p23+e40*2-p4)/4;
        const double w38 = z21/8;
        const double w37 = z8/8;
        const double w36 = z6/2;
        const double w35 = z3/4;
        const double w34 = -z2/2;
        const double e51 = (v68-p7-p15)/4-w43-w38;
        const double e52 = (p17+e35)/4-w37;
        const double c31 = w42+w37;
        const double c11 = w37-w42;
        const double c32 = -w37-w36;
        const double e54 = w35-w41;
        const double w24 = w38+w35;
        const double w29 = (z8-z21)/8;
        const double c12 = (z6-z3)/2-w37;
        const double w23 = -e52;
        const double c21 = e52-w43;
        const double e55 = (p40+v63-p35+z3)/4+w29-w34;
        const double e56 = e54+(p5-p10+v65)/4-w36;
        const double e57 = (p19+v71+e35)/4-w41-c31;
        const double e60 = w23+(p22+p12+e40)/2;
        const double c41 = w23-w43;
        const double w20 = e60;
        const double w17 = e54-e60;
        const double c42 = w34-w20;
        const double c22 = w20-(z3+z2)/2;
        const double c34 = w29-e56;
        const double c14 = w29+e56;
        const double c33 = w24-w40-e57;
        const double c13 = w24-w39+e57;
        const double c44 = e51-w17;
        const double c24 = e51+w17;
        const double c43 = w40-w41+e55;
        const double c23 = w41+w39+e55;
        C[0*(size_t)ldc + 0] += c11;
        C[0*(size_t)ldc + 1] += c12;
        C[0*(size_t)ldc + 2] += c13;
        C[0*(size_t)ldc + 3] += c14;
        C[1*(size_t)ldc + 0] += c21;
        C[1*(size_t)ldc + 1] += c22;
        C[1*(size_t)ldc + 2] += c23;
        C[1*(size_t)ldc + 3] += c24;
        C[2*(size_t)ldc + 0] += c31;
        C[2*(size_t)ldc + 1] += c32;
        C[2*(size_t)ldc + 2] += c33;
        C[2*(size_t)ldc + 3] += c34;
        C[3*(size_t)ldc + 0] += c41;
        C[3*(size_t)ldc + 1] += c42;
        C[3*(size_t)ldc + 2] += c43;
        C[3*(size_t)ldc + 3] += c44;
    }
}


typedef struct {
    int M, N, K;
    int Mp, Np, Kp;
    bool use_pad;
    bool cached_pads;
    double *Ap;
    double *Bp;
    double *Cp;
} alpha48_ctx_t;

static inline int round_up4(int x) {
    return (x + 3) & ~3;
}

algo_ctx_t alpha48_init(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)threads;
    alpha48_ctx_t *x = (alpha48_ctx_t*)calloc(1, sizeof(alpha48_ctx_t));
    if (!x) return NULL;
    x->M = M; x->N = N; x->K = K;

    const bool want_pad = (cfg ? cfg->alpha48_pad_4 : true);

    const bool ok_mul4 = ((M % 4) == 0) && ((N % 4) == 0) && ((K % 4) == 0);

    if (ok_mul4) {
        x->use_pad = false;
        x->Mp = M; x->Np = N; x->Kp = K;
        return (algo_ctx_t)x;
    }

    if (!want_pad) {
        free(x);
        return NULL;
    }

    x->use_pad = true;
    x->Mp = round_up4(M);
    x->Np = round_up4(N);
    x->Kp = round_up4(K);

    const size_t Asz = (size_t)x->Mp * (size_t)x->Kp;
    const size_t Bsz = (size_t)x->Kp * (size_t)x->Np;
    const size_t Csz = (size_t)x->Mp * (size_t)x->Np;

    x->Ap = (double*)calloc(Asz, sizeof(double));
    x->Bp = (double*)calloc(Bsz, sizeof(double));
    x->Cp = (double*)malloc(Csz * sizeof(double));

    if (!x->Ap || !x->Bp || !x->Cp) {
        free(x->Ap); free(x->Bp); free(x->Cp); free(x);
        return NULL;
    }

    x->cached_pads = false;
    return (algo_ctx_t)x;
}

void alpha48_destroy(algo_ctx_t ctx) {
    alpha48_ctx_t *x = (alpha48_ctx_t*)ctx;
    if (!x) return;
    free(x->Ap);
    free(x->Bp);
    free(x->Cp);
    free(x);
}

static inline void alpha48_gemm_core(const double *A, int lda,
                                    const double *B, int ldb,
                                    double *C, int ldc,
                                    int M, int N, int K) {
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            double *Cblk = &C[(size_t)i * (size_t)ldc + (size_t)j];
            for (int k = 0; k < K; k += 4) {
                const double *Ablk = &A[(size_t)i * (size_t)lda + (size_t)k];
                const double *Bblk = &B[(size_t)k * (size_t)ldb + (size_t)j];
                alpha48_mul4x4_add(Ablk, lda, Bblk, ldb, Cblk, ldc);
            }
        }
    }
}

bool alpha48_run(algo_ctx_t ctx, const double *A, const double *B, double *C,
                 int M, int N, int K, int threads) {
    (void)threads;
    alpha48_ctx_t *x = (alpha48_ctx_t*)ctx;
    if (!x) return false;
    if (x->M != M || x->N != N || x->K != K) return false;

    if (!x->use_pad) {
        alpha48_gemm_core(A, K, B, N, C, N, M, N, K);
        return true;
    }

    const int Mp = x->Mp, Np = x->Np, Kp = x->Kp;
    const size_t Asz = (size_t)Mp * (size_t)Kp;
    const size_t Bsz = (size_t)Kp * (size_t)Np;
    const size_t Csz = (size_t)Mp * (size_t)Np;

    if (!x->cached_pads) {
        memset(x->Ap, 0, Asz * sizeof(double));
        memset(x->Bp, 0, Bsz * sizeof(double));

        for (int i = 0; i < M; ++i) {
            memcpy(&x->Ap[(size_t)i * (size_t)Kp], &A[(size_t)i * (size_t)K], (size_t)K * sizeof(double));
        }
        for (int i = 0; i < K; ++i) {
            memcpy(&x->Bp[(size_t)i * (size_t)Np], &B[(size_t)i * (size_t)N], (size_t)N * sizeof(double));
        }
        x->cached_pads = true;
    }

    memset(x->Cp, 0, Csz * sizeof(double));

    alpha48_gemm_core(x->Ap, Kp, x->Bp, Np, x->Cp, Np, Mp, Np, Kp);

    for (int i = 0; i < M; ++i) {
        memcpy(&C[(size_t)i * (size_t)N], &x->Cp[(size_t)i * (size_t)Np], (size_t)N * sizeof(double));
    }

    return true;
}

void alpha48_ops(int M, int N, int K,
                 unsigned long long *muls,
                 unsigned long long *adds,
                 unsigned long long *const_muls,
                 const bench_config_t *cfg) {
    const bool want_pad = (cfg ? cfg->alpha48_pad_4 : true);
    const bool ok_mul4 = ((M % 4) == 0) && ((N % 4) == 0) && ((K % 4) == 0);

    int Me = M, Ne = N, Ke = K;
    if (!ok_mul4) {
        if (!want_pad) {
            if (muls) *muls = 0;
            if (adds) *adds = 0;
            if (const_muls) *const_muls = 0;
            return;
        }
        Me = round_up4(M);
        Ne = round_up4(N);
        Ke = round_up4(K);
    }

    const unsigned long long Mb = (unsigned long long)(Me / 4);
    const unsigned long long Nb = (unsigned long long)(Ne / 4);
    const unsigned long long Kb = (unsigned long long)(Ke / 4);
    const unsigned long long blocks = Mb * Nb * Kb;

    const unsigned long long rank_muls_per = 48ULL;
    const unsigned long long adds_per = 320ULL;
    const unsigned long long const_muls_per = 49ULL;

    if (muls) *muls = rank_muls_per * blocks;
    if (adds) *adds = adds_per * blocks;
    if (const_muls) *const_muls = const_muls_per * blocks;
}

size_t alpha48_extra_mem(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)threads;
    const bool want_pad = (cfg ? cfg->alpha48_pad_4 : true);
    const bool ok_mul4 = ((M % 4) == 0) && ((N % 4) == 0) && ((K % 4) == 0);

    if (ok_mul4 || !want_pad) {
        return 0;
    }

    int Mp = round_up4(M);
    int Np = round_up4(N);
    int Kp = round_up4(K);

    size_t Asz = (size_t)Mp * (size_t)Kp;
    size_t Bsz = (size_t)Kp * (size_t)Np;
    size_t Csz = (size_t)Mp * (size_t)Np;

    return (Asz + Bsz + Csz) * sizeof(double);
}
