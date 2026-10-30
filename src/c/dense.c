/*
 * Dense matrix multiplication C = A x B: the Assignment 1 baseline and three
 * sequential optimisations of it.
 *
 * Usage: dense <method> <A.bin> <B.bin> <n> <warmup> <reps> <batch> [block] [C_out.bin]
 *
 *   method  ijk        Assignment 1 kernel, unchanged (i-j-k, scalar accumulator)
 *           ikj        loop interchange: B and C are read/written along rows
 *           transpose  B is transposed into a scratch buffer, then i-j-k over B^T
 *                      (the transposition is inside the timed region)
 *           blocked    ikj order inside square tiles of side `block`
 *           blocked_unroll  same tiles, plus 4-way unroll-and-jam over i: one
 *                      load of B[k][j] feeds four rows of C (register blocking)
 *   block   tile side for the blocked methods (ignored otherwise; 0 = default 256)
 *
 * Inputs, timing, output lines and memory metric follow Assignment 1:
 * A.bin and B.bin hold n*n little-endian float64 values, row-major. Inputs
 * are loaded and every buffer (C, and B^T for "transpose") is allocated before
 * timing starts. Each sample times `batch` consecutive multiplications with
 * a monotonic clock and prints
 *     <phase>,<index>,<elapsed_ns>,<checksum>
 * followed at the end by "peak_rss_kb,<peak working set in KiB>".
 * Every method writes all of C on each call (ikj and blocked zero C inside
 * the timed region), so samples never depend on the previous one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>

static double *read_matrix(const char *path, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    double *m = malloc(count * sizeof(double));
    if (!m) { fprintf(stderr, "out of memory\n"); exit(2); }
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "%s: short read\n", path);
        exit(1);
    }
    fclose(f);
    return m;
}

/* Baseline: the Assignment 1 kernel, copied verbatim. B is walked down a column. */
static void matmul_ijk(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
            }
            c[(size_t)i * n + j] = sum;
        }
    }
}

/* Loop interchange: a[i][k] is held in a register while row k of B and row i
 * of C are streamed with unit stride. Same multiply-adds, different order. */
static void matmul_ikj(const double *a, const double *b, double *c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *ci = c + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            const double aik = a[(size_t)i * n + k];
            const double *bk = b + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                ci[j] += aik * bk[j];
            }
        }
    }
}

/* Transposed B: one O(n^2) copy makes the inner product read both operands
 * with unit stride, keeping the i-j-k order and scalar accumulator. */
static void matmul_transpose(const double *a, const double *b, double *bt, double *c, int n) {
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            bt[(size_t)j * n + k] = b[(size_t)k * n + j];
    for (int i = 0; i < n; i++) {
        const double *ai = a + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *btj = bt + (size_t)j * n;
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += ai[k] * btj[k];
            }
            c[(size_t)i * n + j] = sum;
        }
    }
}

/* Cache blocking: the ikj kernel restricted to a bs x bs tile of A, B and C,
 * so the B tile (bs*bs*8 bytes) is reused from cache across bs rows of A. */
static void matmul_blocked(const double *a, const double *b, double *c, int n, int bs) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs) {
        int imax = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int kmax = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++) {
                    double *ci = c + (size_t)i * n;
                    for (int k = kk; k < kmax; k++) {
                        const double aik = a[(size_t)i * n + k];
                        const double *bk = b + (size_t)k * n;
                        for (int j = jj; j < jmax; j++) {
                            ci[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }
}

/* Cache blocking plus register blocking: inside each tile, rows i..i+3 of C
 * are updated together, so every element of B loaded from cache is used four
 * times and the four a[i+r][k] factors stay in registers (unroll-and-jam). */
static void matmul_blocked_unroll(const double *a, const double *b, double *c, int n, int bs) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs) {
        int imax = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int kmax = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jmax = jj + bs < n ? jj + bs : n;
                int i = ii;
                for (; i + 3 < imax; i += 4) {
                    double *c0 = c + (size_t)i * n, *c1 = c0 + n, *c2 = c1 + n, *c3 = c2 + n;
                    for (int k = kk; k < kmax; k++) {
                        const double a0 = a[(size_t)i * n + k], a1 = a[(size_t)(i + 1) * n + k];
                        const double a2 = a[(size_t)(i + 2) * n + k], a3 = a[(size_t)(i + 3) * n + k];
                        const double *bk = b + (size_t)k * n;
                        for (int j = jj; j < jmax; j++) {
                            const double bkj = bk[j];
                            c0[j] += a0 * bkj;
                            c1[j] += a1 * bkj;
                            c2[j] += a2 * bkj;
                            c3[j] += a3 * bkj;
                        }
                    }
                }
                for (; i < imax; i++) {  /* leftover rows when the tile height is not a multiple of 4 */
                    double *ci = c + (size_t)i * n;
                    for (int k = kk; k < kmax; k++) {
                        const double aik = a[(size_t)i * n + k];
                        const double *bk = b + (size_t)k * n;
                        for (int j = jj; j < jmax; j++) {
                            ci[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }
}

/* Peak working set (peak resident memory) of this process in KiB,
 * from GetProcessMemoryInfo. */
static long long peak_rss_kb(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return -1;
    return (long long)(pmc.PeakWorkingSetSize / 1024);
}

/* Monotonic clock in nanoseconds (QueryPerformanceCounter). */
static long long now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (long long)((double)t.QuadPart * 1e9 / (double)freq.QuadPart);
}

static double checksum(const double *c, size_t count) {
    double s = 0.0;
    for (size_t i = 0; i < count; i++) s += c[i];
    return s;
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s method A.bin B.bin n warmup reps batch [block] [C_out.bin]\n",
                argv[0]);
        return 1;
    }
    const char *method = argv[1];
    int n = atoi(argv[4]);
    int warmup = atoi(argv[5]);
    int reps = atoi(argv[6]);
    int batch = atoi(argv[7]);
    int bs = argc > 8 && atoi(argv[8]) > 0 ? atoi(argv[8]) : 256;
    size_t count = (size_t)n * n;
    int m;
    if (!strcmp(method, "ijk")) m = 0;
    else if (!strcmp(method, "ikj")) m = 1;
    else if (!strcmp(method, "transpose")) m = 2;
    else if (!strcmp(method, "blocked")) m = 3;
    else if (!strcmp(method, "blocked_unroll")) m = 4;
    else { fprintf(stderr, "unknown method %s\n", method); return 1; }

    double *a = read_matrix(argv[2], count);
    double *b = read_matrix(argv[3], count);
    double *c = calloc(count, sizeof(double));
    double *bt = m == 2 ? calloc(count, sizeof(double)) : NULL;
    if (!c || (m == 2 && !bt)) { fprintf(stderr, "out of memory\n"); return 2; }

    for (int r = 0; r < warmup + reps; r++) {
        long long t0 = now_ns();
        for (int t = 0; t < batch; t++) {
            switch (m) {
            case 0: matmul_ijk(a, b, c, n); break;
            case 1: matmul_ikj(a, b, c, n); break;
            case 2: matmul_transpose(a, b, bt, c, n); break;
            case 3: matmul_blocked(a, b, c, n, bs); break;
            default: matmul_blocked_unroll(a, b, c, n, bs); break;
            }
        }
        long long t1 = now_ns();
        printf("%s,%d,%lld,%.17g\n", r < warmup ? "warmup" : "measure",
               r < warmup ? r : r - warmup, t1 - t0, checksum(c, count));
        fflush(stdout);
    }

    if (argc > 9) {
        FILE *f = fopen(argv[9], "wb");
        if (!f) { perror(argv[9]); return 1; }
        fwrite(c, sizeof(double), count, f);
        fclose(f);
    }
    printf("peak_rss_kb,%lld\n", peak_rss_kb());
    free(a); free(b); free(c); free(bt);
    return 0;
}
