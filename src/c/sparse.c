/*
 * Sparse matrix multiplication C = A x B in CSR format (Gustavson's row-by-row
 * algorithm with a dense accumulator), written for this assignment; no
 * library is used.
 *
 * Usage: sparse <mode> <A> <B> <n> <warmup> <reps> [C_out.bin]
 *
 *   mode csr    A and B are CSR files (format below). Only the multiplication
 *               is timed.
 *   mode dense  A and B are dense float64 files as in Assignment 1. Each sample
 *               also times, separately, the conversion of A and B to CSR
 *               (convert_in) and the expansion of the CSR product into a
 *               preallocated dense n x n array (convert_out).
 *
 * CSR file (little-endian): int64 n, int64 nnz, int64 row_ptr[n+1],
 * int32 col_idx[nnz], float64 values[nnz]; columns sorted within each row.
 *
 * Timed kernel: symbolic pass (exact nnz of every row of C), allocation of C's
 * col_idx/values arrays, numeric pass. C is freed after each sample, outside
 * the timed region. Column indices inside a row of C are left in first-touch
 * order (not sorted). The k-loop visits A's row in increasing k, so every
 * C_ij is the same left-to-right sum of products as in the dense i-j-k
 * kernel (skipped terms are exact zeros).
 *
 * One line per sample:
 *     <phase>,<index>,<kernel_ns>,<checksum>,<nnz_c>,<convert_in_ns>,<convert_out_ns>
 * (conversion fields are -1 in csr mode), then
 *     nnz,<nnz_a>,<nnz_b>,<nnz_c>
 *     csr_bytes,<bytes of A, B and C in CSR plus the O(n) workspace>
 *     peak_rss_kb,<peak working set in KiB>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>

typedef struct {
    int64_t n, nnz;
    int64_t *row_ptr;
    int32_t *col;
    double *val;
} csr;

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "out of memory (%llu bytes)\n", (unsigned long long)bytes); exit(2); }
    return p;
}

static void csr_free(csr *m) {
    free(m->row_ptr); free(m->col); free(m->val);
    m->row_ptr = NULL; m->col = NULL; m->val = NULL;
}

static size_t csr_bytes(const csr *m) {
    return (size_t)(m->n + 1) * sizeof(int64_t) + (size_t)m->nnz * (sizeof(int32_t) + sizeof(double));
}

static void read_exact(FILE *f, void *dst, size_t size, size_t count, const char *path) {
    if (fread(dst, size, count, f) != count) {
        fprintf(stderr, "%s: short read\n", path);
        exit(1);
    }
}

static csr read_csr(const char *path, int64_t n_expected) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    csr m;
    read_exact(f, &m.n, sizeof(int64_t), 1, path);
    read_exact(f, &m.nnz, sizeof(int64_t), 1, path);
    if (m.n != n_expected) { fprintf(stderr, "%s: n=%lld, expected %lld\n", path,
                                     (long long)m.n, (long long)n_expected); exit(1); }
    m.row_ptr = xmalloc((size_t)(m.n + 1) * sizeof(int64_t));
    m.col = xmalloc((size_t)m.nnz * sizeof(int32_t));
    m.val = xmalloc((size_t)m.nnz * sizeof(double));
    read_exact(f, m.row_ptr, sizeof(int64_t), (size_t)m.n + 1, path);
    read_exact(f, m.col, sizeof(int32_t), (size_t)m.nnz, path);
    read_exact(f, m.val, sizeof(double), (size_t)m.nnz, path);
    fclose(f);
    return m;
}

static double *read_dense(const char *path, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    double *m = xmalloc(count * sizeof(double));
    read_exact(f, m, sizeof(double), count, path);
    fclose(f);
    return m;
}

/* Dense row-major -> CSR: one counting pass, allocation, one filling pass. */
static csr dense_to_csr(const double *d, int64_t n) {
    csr m;
    m.n = n;
    m.row_ptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));
    m.row_ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t cnt = 0;
        for (int64_t j = 0; j < n; j++) cnt += d[i * n + j] != 0.0;
        m.row_ptr[i + 1] = m.row_ptr[i] + cnt;
    }
    m.nnz = m.row_ptr[n];
    m.col = xmalloc((size_t)m.nnz * sizeof(int32_t));
    m.val = xmalloc((size_t)m.nnz * sizeof(double));
    int64_t p = 0;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            if (d[i * n + j] != 0.0) { m.col[p] = (int32_t)j; m.val[p] = d[i * n + j]; p++; }
    return m;
}

/* CSR -> dense into a caller-provided n*n array (zeroed here). */
static void csr_to_dense(const csr *m, double *d) {
    memset(d, 0, (size_t)m->n * m->n * sizeof(double));
    for (int64_t i = 0; i < m->n; i++)
        for (int64_t p = m->row_ptr[i]; p < m->row_ptr[i + 1]; p++)
            d[i * m->n + m->col[p]] = m->val[p];
}

/* Gustavson SpGEMM. `marker` (n int64) and `acc` (n double) are workspace. */
static csr spgemm(const csr *a, const csr *b, int64_t *marker, double *acc) {
    const int64_t n = a->n;
    csr c;
    c.n = n;
    c.row_ptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));

    /* symbolic pass: count distinct columns of every row of C */
    for (int64_t j = 0; j < n; j++) marker[j] = -1;
    c.row_ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t cnt = 0;
        for (int64_t pa = a->row_ptr[i]; pa < a->row_ptr[i + 1]; pa++) {
            const int32_t k = a->col[pa];
            for (int64_t pb = b->row_ptr[k]; pb < b->row_ptr[k + 1]; pb++) {
                const int32_t j = b->col[pb];
                if (marker[j] != i) { marker[j] = i; cnt++; }
            }
        }
        c.row_ptr[i + 1] = c.row_ptr[i] + cnt;
    }
    c.nnz = c.row_ptr[n];
    c.col = xmalloc((size_t)c.nnz * sizeof(int32_t));
    c.val = xmalloc((size_t)c.nnz * sizeof(double));

    /* numeric pass: accumulate row i of C in acc, remember touched columns */
    for (int64_t j = 0; j < n; j++) marker[j] = -1;
    for (int64_t i = 0; i < n; i++) {
        int64_t start = c.row_ptr[i], p = start;
        for (int64_t pa = a->row_ptr[i]; pa < a->row_ptr[i + 1]; pa++) {
            const int32_t k = a->col[pa];
            const double aik = a->val[pa];
            for (int64_t pb = b->row_ptr[k]; pb < b->row_ptr[k + 1]; pb++) {
                const int32_t j = b->col[pb];
                if (marker[j] != i) { marker[j] = i; c.col[p++] = j; acc[j] = aik * b->val[pb]; }
                else acc[j] += aik * b->val[pb];
            }
        }
        for (int64_t q = start; q < p; q++) c.val[q] = acc[c.col[q]];
    }
    return c;
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

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s csr|dense A B n warmup reps [C_out.bin]\n", argv[0]);
        return 1;
    }
    const int dense_mode = !strcmp(argv[1], "dense");
    if (!dense_mode && strcmp(argv[1], "csr")) { fprintf(stderr, "unknown mode %s\n", argv[1]); return 1; }
    const int64_t n = atoll(argv[4]);
    const int warmup = atoi(argv[5]);
    const int reps = atoi(argv[6]);
    const size_t count = (size_t)n * n;

    csr a = {0}, b = {0}, c = {0};
    double *da = NULL, *db = NULL, *dc = NULL;
    if (dense_mode) {
        da = read_dense(argv[2], count);
        db = read_dense(argv[3], count);
        dc = xmalloc(count * sizeof(double));
        memset(dc, 0, count * sizeof(double));
    } else {
        a = read_csr(argv[2], n);
        b = read_csr(argv[3], n);
    }
    int64_t *marker = xmalloc((size_t)n * sizeof(int64_t));
    double *acc = xmalloc((size_t)n * sizeof(double));

    size_t bytes = 0;
    for (int r = 0; r < warmup + reps; r++) {
        long long conv_in = -1, conv_out = -1;
        if (dense_mode) {
            if (r > 0) { csr_free(&a); csr_free(&b); }
            long long t0 = now_ns();
            a = dense_to_csr(da, n);
            b = dense_to_csr(db, n);
            conv_in = now_ns() - t0;
        }
        if (r > 0) csr_free(&c);
        long long t0 = now_ns();
        c = spgemm(&a, &b, marker, acc);
        long long kernel = now_ns() - t0;
        if (dense_mode) {
            t0 = now_ns();
            csr_to_dense(&c, dc);
            conv_out = now_ns() - t0;
        }
        double sum = 0.0;
        for (int64_t q = 0; q < c.nnz; q++) sum += c.val[q];
        printf("%s,%d,%lld,%.17g,%lld,%lld,%lld\n", r < warmup ? "warmup" : "measure",
               r < warmup ? r : r - warmup, kernel, sum, (long long)c.nnz, conv_in, conv_out);
        fflush(stdout);
    }
    bytes = csr_bytes(&a) + csr_bytes(&b) + csr_bytes(&c) + (size_t)n * (sizeof(int64_t) + sizeof(double));

    if (argc > 7) {
        if (!dc) { dc = xmalloc(count * sizeof(double)); }
        csr_to_dense(&c, dc);
        FILE *f = fopen(argv[7], "wb");
        if (!f) { perror(argv[7]); return 1; }
        fwrite(dc, sizeof(double), count, f);
        fclose(f);
    }
    printf("nnz,%lld,%lld,%lld\n", (long long)a.nnz, (long long)b.nnz, (long long)c.nnz);
    printf("csr_bytes,%llu\n", (unsigned long long)bytes);
    printf("peak_rss_kb,%lld\n", peak_rss_kb());
    csr_free(&a); csr_free(&b); csr_free(&c);
    free(da); free(db); free(dc); free(marker); free(acc);
    return 0;
}
