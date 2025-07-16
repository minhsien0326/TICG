#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h> 
#include <errno.h>
#include <sys/stat.h> 
#include <stdbool.h>
#include <omp.h>
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define NDIV (1+IMM1/NTAB)
#define STR_LEN 1024
#define NUM_STR 8
#define MAX_DU 700.0 // Threshold for Metropolis test to avoid exp overflow
#define MAX_CAND 64 // Maximum number of candidates for swap move 
#define VERSION "4.10" 
#define DIMENSION 2
/*
================================================================================
----------------------------------- CHANGE LOG --------------------------------- 
Version 4.8 
for this subversion, the following changes have been made: 
- Use coloring (checkerboard) scheme to avoid racing condition in the polymer segment move loop. 
----------------------------------- HISTORY ------------------------------------ 
Version 4.7 
for this subversion, the following changes have been made:
- Use `refEcell` and `affected_grid_idx` to track the local field energy change that happens for at most
  `n_affected` grid points. 
Version 4.4 
for this subversion, the following changes have been made: 
- Modify the swap move loop which calls attemptSwapMove() to loop over junctions instead of free ends. 
  so that we search for candidate free ends for a given junction, aligning with the function design. 
- Fix bug for the counting of acceptance ratio for swap move. 
Version 4.3 
for this subversion, the following changes have been made: 
- Fix bug: the bond energy calculation was not accounting for the dimensionality of the system for swap move. 
- Change bond energy calculation to local for swap move
- Use listed-based cell list for swap move to speed up the search for candidates. 
- Keep track of the acceptance ratio for each type of move. (polymer bead move, junction move, and swapping move) 
- Print out coordinates of junctions and free ends to check their distributions. 
Version 4.2 
For this subversion, the following changes have been made:
- Add temperature (beta) dependence in Metropolis tests 
- Add a flag for constraint-release 
    - 0 = no constraint release (normal MC) 
    - 1 = constraint-released 
- Add command line arguments for setting parameters:
    - argv[1] = chiN
    - argv[2] = beta: 1/(kT)
    - argv[3] = release_flag: 0 = constrained (normal MC), 1 = released 
Version 4.1 
This version keeps the loop-it-all feature from version 4, but adds a few modification from latest version 8: 
- Add param_ file to store the tunable parameters for the simulation. 
- Modify prefactor for bond energy to account for the dimensionality of the system. 
- Fix bug in gridPointInterpolator where indices were not updated correctly. 
Version 4 
This version aims to optimize some loops in the previous versions to make the code more efficient. 
The following changes have been made: 
- Add version number for easier tracking 
- Add outputPath for saving output files 
- Add `strs` array to store output file names 
- Add safeguard in wrap1D() to avoid roundoff error 
- Add safeguard in gridPointInterpolator() to ensure positive indices 
- Add diagnostic output in depositPhi() to make sure indices are valid 
- Modify the calculation for local field energy change to account for overlapping grid points and the net weights. 
Version 3 
- Some bugs that result in Phi's drifting out of bounds. 
Version 2 
- Define makeStencil, depositPhi, and bondEnergy functions for better organization. 
- Still some bugs regarding energy calculation. 
Version 1 
- Modified from 1D code to 2D code. Lots of repetitive blocks. 
================================================================================  
*/

// 1-D wrap into [0,L)

static long ran1_idum2 = 123456789;
static long ran1_iy = 0;
static long ran1_iv[NTAB] = {0};
#pragma omp threadprivate(ran1_idum2, ran1_iy, ran1_iv)

static int gasdev_iset = 0;
static float gasdev_gset = 0.0f;
#pragma omp threadprivate(gasdev_iset, gasdev_gset)


static double wrap1D(double x, double L)
{
    x -= L * floor (x / L); // wrap 
    if (x >= L - EPS) x = 0.0; // guard against roundoff 
    return x; 
}

// 1-D minimum-image separation in [-L/2, L/2]
static double dmin1D(double dx, double L)
{
    if      (dx >  0.5 * L) dx -= L;
    else if (dx < -0.5 * L) dx += L;
    return dx;
}

// 2-D wrap a point into [0,Lx) and [0,Ly)
static void wrap2D(double *x, double *y, double Lx, double Ly)
{
    *x = wrap1D(*x, Lx);
    *y = wrap1D(*y, Ly);
}

// 2-D minimum-image displacement vector
static void dmin2D(double x1, double y1,
                   double x2, double y2,
                   double Lx, double Ly,
                   double *dx, double *dy)
{
    *dx = dmin1D(x1 - x2, Lx);
    *dy = dmin1D(y1 - y2, Ly);
}

// Initialize random number generator
long initRan()
{
    unsigned long a = clock();
    unsigned long b = time(NULL);
    unsigned long c = getpid();
    a = a - b;  a = a - c;  a = a ^ (c >> 13);
    b = b - c;  b = b - a;  b = b ^ (a << 8);
    c = c - a;  c = c - b;  c = c ^ (b >> 13);
    a = a - b;  a = a - c;  a = a ^ (c >> 12);
    b = b - c;  b = b - a;  b = b ^ (a << 16);
    c = c - a;  c = c - b;  c = c ^ (b >> 5);
    a = a - b;  a = a - c;  a = a ^ (c >> 3);
    b = b - c;  b = b - a;  b = b ^ (a << 10);
    c = c - a;  c = c - b;  c = c ^ (b >> 15);
    return c % 1000000000;
}

// Generate random number between 0 and 1
float ran1(long *idum)
{
    int j;
    long k;
    float temp;
    
    if (*idum <= 0)
    {
        if (-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        ran1_idum2 = (*idum);
        for (j = NTAB + 7; j >= 0; --j)
        {
            k = (*idum) / IQ1;
            *idum = IA1 * (*idum - k * IQ1) - k * IR1;
            if (*idum < 0) *idum += IM1;
            if (j < NTAB) ran1_iv[j] = *idum;
        }
        ran1_iy = ran1_iv[0];
    }
    k = (*idum) / IQ1;
    *idum = IA1 * (*idum - k * IQ1) - k * IR1;
    if (*idum < 0) *idum += IM1;
    k = ran1_idum2 / IQ2;
    ran1_idum2 = IA2 * (ran1_idum2 - k * IQ2) - k * IR2;
    if (ran1_idum2 < 0) ran1_idum2 += IM2;
    j = ran1_iy / NDIV;
    ran1_iy = ran1_iv[j] - ran1_idum2;
    ran1_iv[j] = *idum;
    if (ran1_iy < 1) ran1_iy += IMM1;
    if ((temp = AM * ran1_iy) > RNMX) return RNMX;
    else return temp;
}

// Generate normally distributed random number
float gasdev(long *idum)
{
    
    float fac, rsq, v1, v2;
    
    if (*idum < 0) gasdev_iset = 0;
    if (gasdev_iset == 0)
    {
        do
        {
            v1 = 2.0 * ran1(idum) - 1.0;
            v2 = 2.0 * ran1(idum) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        }
        while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gasdev_gset = v1 * fac;
        gasdev_iset = 1;
        return v2 * fac;
    }
    else
    {
        gasdev_iset = 0;
        return gasdev_gset;
    }
}

// Shuffle a tiny int array (Fisher-Yates) 
static inline void shuffle4(int *a, long *rng)
{
    for (int i = 3; i > 0; --i) {
        int j = (int)(ran1(rng) * (i + 1));
        int tmp = a[i];  a[i] = a[j];  a[j] = tmp;
    }
}

void create_directory(const char* path) {
    if (mkdir(path, 0777) == -1) {
        if (errno != EEXIST) {
            printf("Error creating directory: %s\n", path);
            exit(1);
        }
    }
    printf("Directory created or already exists: %s\n", path);
}

typedef struct {
    int idx[4];
    double w[4];
} gridPointInterpolator;

// Build gridPointInterpolator for point (x,y) in a grid of size Mx*My with grid spacing dL
static gridPointInterpolator makeInterpolator(double x, double y, double dL, int Mx, int My)
{
    // Convert the bead's real position to "grid units" 
    double gx = x / dL, gy = y / dL;
    // Find the index of the cell's lower-left corner 
    int ix0 = (int)floor(gx);
    int iy0 = (int)floor(gy);
    // Extract the fraction offset inside the cell 
    double fx = gx - ix0, fy = gy - iy0;
    // Wrap indices for PBC 
    ix0 = (ix0 % Mx + Mx) % Mx; 
    iy0 = (iy0 % My + My) % My; 
    int ix1 = ((ix0 + 1) % Mx + Mx) % Mx;  
    int iy1 = ((iy0 + 1) % My + My) % My; 
    
    gridPointInterpolator s;
    // bilinear indices and weights 
    s.idx[0] = ix0 + iy0 * Mx; s.w[0] = (1.0 - fx) * (1.0 - fy);
    s.idx[1] = ix0 + iy1 * Mx; s.w[1] = (1.0 - fx) * fy;
    s.idx[2] = ix1 + iy0 * Mx; s.w[2] = fx * (1.0 - fy);
    s.idx[3] = ix1 + iy1 * Mx; s.w[3] = fx * fy;
    return s;
}

// Deposit or remove dphi onto four grid points (stencil) in the gridPointInterpolator
static void depositPhi(double *Phi, const gridPointInterpolator *s, double dphi, int Mg, int sign)
{
    for (int m = 0; m < 4; ++m) {
        if (s->idx[m] < 0 || s->idx[m] >= Mg) {
            fprintf(stderr, "Invalid grid index: %d at m=%d\n", s->idx[m], m);
        }
        Phi[s->idx[m]] += sign * dphi * s->w[m];
    }
}

// Local field-energy density
static double localEnergy(double phiA, double phiB, double chiN, double kappaN)
{
    double incompr = phiA + phiB - 1.0;
    return chiN * phiA * phiB + kappaN * incompr * incompr;
}

// Bond energy for bead i
static double bondEnergy(int i, int N, const double *beadX, const double *beadY, 
                        const double *juncX, const double *juncY, const int *Jbond, 
                        int nj, double b2, double b2a, double Lx, double Ly, int dim)
{
    double E = 0.0, dx, dy, r2;
    // Intra-chain bonds
    if (i % N > 0) { // previous bead (if not the first) 
        dmin2D(beadX[i], beadY[i], beadX[i-1], beadY[i-1], Lx, Ly, &dx, &dy);
        r2 = dx * dx + dy * dy;
        E += 0.5 * b2 * r2 * (double) dim;
    }
    if (i % N < N-1) { // next bead (if not the last) 
        dmin2D(beadX[i], beadY[i], beadX[i+1], beadY[i+1], Lx, Ly, &dx, &dy);
        r2 = dx * dx + dy * dy;
        E += 0.5 * b2 * r2 * (double) dim;
    }
    // Junction bond (only if bead i is bound to a junction)
    for (int j = 0; j < 3 * nj; ++j) {
        if (Jbond[j] == i) {
            int J = j / 3;
            dmin2D(beadX[i], beadY[i], juncX[J], juncY[J], Lx, Ly, &dx, &dy);
            r2 = dx * dx + dy * dy;
            E += 0.5 * b2a * r2 * (double) dim;
        }
    }
    return E;
}

// Junction bond energy for junction i
static double junctionBondEnergy(int i, const double *beadX, const double *beadY, 
                                const double *juncX, const double *juncY, const int *Jbond, 
                                double b2a, double Lx, double Ly, int dim)
{
    double E = 0.0, dx, dy, r2;
    for (int j = 3 * i; j < 3 * i + 3; ++j) {
        int b = Jbond[j];
        dmin2D(beadX[b], beadY[b], juncX[i], juncY[i], Lx, Ly, &dx, &dy);
        r2 = dx * dx + dy * dy;
        E += 0.5 * b2a * r2 * (double) dim;
    }
    return E;
}

static inline int cell_index(double x, double y,
                             double Lx, double Ly,
                             double Lcell, 
                             int Nx, int Ny)
{
    int cx = (int)floor(x / Lcell);
    int cy = (int)floor(y / Lcell);
    // wrap to [0,Nx) and [0,Ny) 
    cx = (cx % Nx + Nx) % Nx;
    cy = (cy % Ny + Ny) % Ny;
    return cx + cy * Nx;
}

static int gather_candidates(double x, double y,
                             const double *beadX, const double *beadY,
                             const int *FEnd,
                             const int *cellHead, const int *cellNext,
                             double r_swap,
                             int Nx, int Ny, double Lx, double Ly, double Lcell, 
                             int *cand)
{
    // centre cell of the junction 
    int cx0 = (int)floor(x / Lcell);
    int cy0 = (int)floor(y / Lcell);
    int nCand = 0;

    // search the 3×3 surrounding cells (periodic) 
    for (int dcx = -1; dcx <= 1; ++dcx) {
        for (int dcy = -1; dcy <= 1; ++dcy) {
            int cx = (cx0 + dcx + Nx) % Nx;
            int cy = (cy0 + dcy + Ny) % Ny;
            int c  = cx + cy * Nx;

            // iterate over this cell’s linked list 
            for (int fe = cellHead[c]; fe != -1; fe = cellNext[fe]) {
                int b = FEnd[fe];   // global bead index 
                double dx, dy;
                dmin2D(beadX[b], beadY[b], x, y, Lx, Ly, &dx, &dy);
                double r2 = dx * dx + dy * dy;
                if (r2 < r_swap * r_swap) {
                    cand[nCand++] = fe; // store free-end slot index 
                    if (nCand == MAX_CAND) return nCand;
                }
            }
        }
    }
    return nCand;
}

static int attemptSwapMove(int           J, // the chosen junction index 
                           long         *idum,
                           double        beta,
                           int           nj,
                           int           nfe,
                           double        r_swap,
                           double        Lx, double Ly, int dim,
                           double        b2a,
                           double        Lcell, int Nxcell, int Nycell,
                           double       *beadX, double *beadY,
                           double       *juncX, double *juncY,
                           int          *Jbond, int *FEnd,
                           int          *cellHead, int *cellNext)
{
    // collect nearby free ends 
    int cand[MAX_CAND];
    int nCand = gather_candidates(juncX[J], juncY[J],
                                  beadX, beadY,
                                  FEnd,
                                  cellHead, cellNext, r_swap,
                                  Nxcell, Nycell, Lx, Ly, Lcell,
                                  cand);
    if (nCand == 0) return -1;

    // choose one free end from the candidate list 
    int feSlot = cand[(int)(ran1(idum) * nCand)];
    int b_fe   = FEnd[feSlot];
    // choose one random arm of junction J 
    int randArm    = (int)(ran1(idum) * 3); 
    int slot   = 3*J + randArm;
    int b_j    = Jbond[slot]; // global bead index of the bound bead to the junction 

    // Local energy change for the swap move 
    double dU;
    {
        double dx, dy, r2_old, r2_new;

        // r^2 of the currently bound bead to the junction 
        dmin2D(beadX[b_j],  beadY[b_j],  juncX[J], juncY[J], Lx, Ly, &dx, &dy);
        r2_old  = dx*dx + dy*dy;
        // r^2 of the free end to the junction 
        dmin2D(beadX[b_fe], beadY[b_fe], juncX[J], juncY[J], Lx, Ly, &dx, &dy);
        r2_new  = dx*dx + dy*dy;

        dU = 0.5 * b2a * (r2_new - r2_old) * dim;
    }

    // Metropolis 
    int accept = (dU <= 0.0) || (dU < MAX_DU && ran1(idum) < exp(-beta * dU));
    if (!accept) return 0;

    // commit swap 
    Jbond[slot]   = b_fe;
    FEnd[feSlot]  = b_j;
    return 1;
}

static void rebuild_cell_list(int nfe, int Ncell,
                              double *beadX, double *beadY,
                              int *FEnd, int *cellHead, int *cellNext,
                              double Lx, double Ly,
                              double Lcell, int Nxcell, int Nycell)
{
    for (int c = 0; c < Ncell; ++c) cellHead[c] = -1;
    for (int i = 0; i < nfe;   ++i) cellNext[i] = -1;

    for (int i = 0; i < nfe; ++i) {
        int b    = FEnd[i];
        int cIdx = cell_index(beadX[b], beadY[b],
                              Lx, Ly, Lcell, Nxcell, Nycell);

        cellNext[i]    = cellHead[cIdx];   // link to previous head 
        cellHead[cIdx] = i;                // new head is this end 
    }
}



int main(int argc, const char *argv[])
{

    int nthreads = 8;           
    omp_set_num_threads(nthreads);
    int i, j, k, t, N, Ntot, Mx, My, Mg, n, bondcount, nj, nfe, count, index, Jindex;
    int chain, bead, dim=2; 
    bool accept; 
    double dx, dy, r2, kappaN, Lx, Ly, sqN, dL, b2, b2a, Uo, U1, dU, dx_trial, dy_trial;
    double dphi, p, bondave, U_field_total, pref;
    double Lcell; 
    int Nxcell, Nycell, Ncell, t_rebuild=5; 

    // Initialize random number generator
    long *idum = malloc(sizeof(long));
    if (!idum) { fprintf(stderr, "Memory allocation failed for idum\n"); exit(1); }
    *idum = initRan();

    // Command line arguments 
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s chiN beta [release_flag 0|1]\n", argv[0]);
        return 1;
    }
    double chiN = atof(argv[1]); // Segregation strength 
    double beta = atof(argv[2]); // Inverse temperature (1/kT)
    double r_swap = atof(argv[3]); 
    int release_flag = (argc == 5) ? atoi(argv[4]) : 0; // Constraint release flag: 0 = constrained, 1 = released


    // Simulation parameters
    long long tmax = (long long)1e8; 
    kappaN = 200.0;
    sqN = 128.0; // Invariant degree of polymerization
    /*
    \sqrt{\bar{N}} is the invariant degree of polymerization, defined as 
            \sqrt{\bar{N}} = \rho_0 R_e^3 / N
    where \rho_0 is the bead number density, R_e is the end-to-end distance of a chain, and N is the degree of polymerization. 
    Given R_e = \sqrt{N} b for ideal chain (melt condition), we can obtain 
            \rho_0 = \sqrt{\bar{N}} / (b^3 \sqrt{N})
    with \sqrt{\bar{N}} = 128 and N = 10, we have 
            \rho_0 = 40.5 b^-3 
    */
    Lx = 20.0;   // In units of Re
    Ly = 20.0;
    dL = 0.25;
    Mx = (int)(Lx / dL);
    My = (int)(Ly / dL);
    Mg = Mx * My;
    N = 10; // Discretization of polymer
    b2 = (double)(N - 1); // Spring constant between beads
    b2a = 1.0 * b2 * 9.0; // Spring constant between chains, junctions
    // b2a = 0.0; 
    n = (int)(Lx * Ly * sqN); // Assuming 1Re in z direction. 
    Ntot = n * N; // Total number of beads 
    dphi = (double)Mg / (double)Ntot; // The amount of phi each bead contributes 
    nj = (int)(0.95 * (double)n * 2.0 / 3.0); // The number of junctions, which has 5% less binding sites total than chain ends so that there are some free ends
    nfe = 2 * n - 3 * nj; // Number of free ends 
    // r_swap = 0.8; // In units of Re, can be adjusted later. This determines the reaction region for swap move. 
    Lcell = r_swap; // Cell size for swap move, in units of Re
    Nxcell = (int)ceil(Lx / Lcell);
    Nycell = (int)ceil(Ly / Lcell);
    Ncell  = Nxcell * Nycell;

    printf("%d %d %d %d %lf\n", n, Ntot, nj, nfe, dphi);

    // Allocate memory with checks
    // Coordinates of Ntot beads 
    double *beadX = calloc(Ntot, sizeof(double));
    double *beadY = calloc(Ntot, sizeof(double));
    // Coordinates of nj junctions 
    double *juncX = calloc(nj, sizeof(double));
    double *juncY = calloc(nj, sizeof(double));
    int *JID = calloc(nj, sizeof(int)); // Number of A-type chains per junction 
    int *JIDist = calloc(nj * 4, sizeof(int)); // Distribution of junction types 
    int *CONNECT = calloc(2 * n, sizeof(int)); // Distribution of link types 
    int *CIJ = calloc(10, sizeof(int)); // Counts of chain connection types 
    int *Jbond = calloc(3 * nj, sizeof(int)); // Global bead indices of beads connected to junctions (3 beads per junction) 
    int *FEnd = calloc(nfe, sizeof(int)); // Global bead indices of free-end beads 
    double *PhiA = calloc(Mg, sizeof(double));
    double *PhiB = calloc(Mg, sizeof(double));
    double *PhiJ = calloc(Mg, sizeof(double));
    double *PhiAave = calloc(Mg, sizeof(double));
    double *PhiBave = calloc(Mg, sizeof(double));
    double *PhiJave = calloc(Mg, sizeof(double));
    int *phicount = calloc(Mg, sizeof(int));
    int *cellHead = calloc(Ncell, sizeof(int)); // Head index for each cell 
    int *cellNext = calloc(nfe, sizeof(int)); // linked-list "next" pointers 
    double *Ufield = calloc(Mg, sizeof(double));

    if (!beadX || !beadY || !juncX || !juncY || !JID || !JIDist || !CONNECT ||
        !CIJ || !Jbond || !FEnd || !PhiA || !PhiB || !PhiJ || !Ufield ||
        !PhiAave || !PhiBave || !PhiJave || !phicount || !cellHead || !cellNext) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    char outputPath[STR_LEN];
    snprintf(outputPath, sizeof(outputPath), "./data_v%s_test/", VERSION);  

    // Create output directory
    create_directory(outputPath);    

    const char* prefixes[] = {
        "PHI_", // 0
        "Coord_", // 1 
        "Misc_", // 2 
        "juncDistr_", // 3 
        "CIJ_", // 4
        "PHIavg_", // 5 
        "param_", // 6 
        "accRatio_", // 7 
    };

    FILE* files[NUM_STR];
    char* strs[NUM_STR];
   // int len = snprintf(NULL, 0, "%s%sXN%.1f_rSwap%.2f_rel%d_ver%s.txt",
   //                outputPath, prefixes[i], chiN, r_swap, release_flag, VERSION);
    for(int i = 0; i < NUM_STR; i++){
        int len = snprintf(NULL, 0, "%s%sXN%.1f_rSwap%.2f_rel%d_ver%s.txt",
                   outputPath, prefixes[i], chiN, r_swap, release_flag, VERSION);
       // strs[i] = malloc(sizeof(char) * STR_LEN);
        strs[i] = malloc(len+1);
        snprintf(strs[i],len+1, "%s%sXN%.1f_rSwap%.2f_rel%d_ver%s.txt", 
                outputPath, 
                prefixes[i], 
                chiN, 
                r_swap, 
                release_flag,
                VERSION);
    }    
        
    files[6] = fopen(strs[6], "w");
    // prefix "param_"
    if (files[6] != NULL) {
        fprintf(files[6], "tmax: %lld\n", tmax);
        fprintf(files[6], "chiN: %lf\n", chiN);
        fprintf(files[6], "kappaN: %lf\n", kappaN);
        fprintf(files[6], "sqN: %lf\n", sqN);
        fprintf(files[6], "Lx: %lf\n", Lx);
        fprintf(files[6], "Ly: %lf\n", Ly);
        fprintf(files[6], "dL: %lf\n", dL);
        fprintf(files[6], "Mx: %d\n", Mx);
        fprintf(files[6], "My: %d\n", My);
        fprintf(files[6], "N: %d\n", N);
        fprintf(files[6], "b2: %lf\n", b2);
        fprintf(files[6], "b2a: %lf\n", b2a);
        fprintf(files[6], "n: %d\n", n);
        fprintf(files[6], "Ntot: %d\n", Ntot);
        fprintf(files[6], "dphi: %lf\n", dphi);
        fprintf(files[6], "nj: %d\n", nj);
        fprintf(files[6], "nfe: %d\n", nfe);
        fprintf(files[6], "beta: %lf\n", beta); 
        fprintf(files[6], "r_swap: %lf\n", r_swap);
        fprintf(files[6], "Lcell: %lf\n", Lcell);
        fprintf(files[6], "Nxcell: %d\n", Nxcell);
        fprintf(files[6], "Nycell: %d\n", Nycell);
        fprintf(files[6], "Ncell: %d\n", Ncell);
        fprintf(files[6], "release_flag: %d\n", release_flag);
        fclose(files[6]);
    }  

    // Initialize CIJ matrix
    for (i = 0; i < 10; ++i) {
        CIJ[i] = 0;
    }

    // Initialize chain locations
    for (i = 0; i < n; ++i) {
        // Place the first bead of chain `i` at a random position in the box 
        beadX[i * N] = Lx * ran1(idum);
        beadY[i * N] = Ly * ran1(idum);
        for (j = 1; j < N; ++j) {
            // Subsequent beads are placed at a distance dL from the previous one  
            beadX[i * N + j] = beadX[i * N + j - 1] + dL * gasdev(idum);
            beadY[i * N + j] = beadY[i * N + j - 1] + dL * gasdev(idum);
            wrap2D(&beadX[i * N + j], &beadY[i * N + j], Lx, Ly);
        }
    }

    // Initialize junction locations
    for (i = 0; i < nj; ++i) {
        juncX[i] = ran1(idum) * Lx;
        juncY[i] = ran1(idum) * Ly;
    }

    // Assign chain ends to junctions or free ends
    count = 0;
    for (i = 0; i < n; ++i) {
        if (count < 3 * nj) {
            // i*N is the index of the first monomer in chain i 
            Jbond[count] = i * N;
            count++;
        } else {
            FEnd[count - 3 * nj] = i * N;
            count++;
        }
        if (count < 3 * nj) {
            // i*N+N-1 is the index of the last monomer in chain i 
            Jbond[count] = i * N + N - 1;
            count++;
        } else {
            FEnd[count - 3 * nj] = i * N + N - 1;
            count++;
        }
    }

    // Initialize grid arrays
    for (i = 0; i < Mg; ++i) {
        PhiAave[i] = 0.0;
        PhiBave[i] = 0.0;
        PhiJave[i] = 0.0;
        phicount[i] = 0;
        PhiA[i] = 0.0;
        PhiB[i] = 0.0;
        PhiJ[i] = 0.0;
    }

    // Assign monomers to grid
    for (chain = 0; chain < n; ++chain) {
        for (bead = 0; bead < N; ++bead) {
            int i = chain * N + bead; // Global bead index           
            gridPointInterpolator s = makeInterpolator(beadX[i], beadY[i], dL, Mx, My);
            // Assign initial density contributions to PhiA or PhiB 
            if ((chain % 2) == 0) {
                depositPhi(PhiA, &s, dphi, Mg, 1);
            } else {
                depositPhi(PhiB, &s, dphi, Mg, 1);
            }
        }
    }

    // // Compute initial field energy
    // U_field_total = 0.0;
    pref = dL * dL * sqN;
    // for (j = 0; j < Mg; ++j) {
    //     Ufield[j] = localEnergy(PhiA[j], PhiB[j], chiN, kappaN);
    //     U_field_total += Ufield[j] * pref;
    // }

    // Initialize junction type distribution
    for (i = 0; i < 4 * nj; ++i) {
        JIDist[i] = 0;
    }

    bondave = 0.0; bondcount = 0;

    long long total_attemptedBead=0, total_acceptedBead=0; 
    long long total_acceptedJunc=0, total_attemptedJunc=0;
    long long prop_swap = 0,  tried_swap = 0,  acc_swap = 0;
    int need_rebuild = 1; 
    for (t = 0; t < tmax; ++t) {
        int colour_order[4] = {0, 1, 2, 3}; 
        shuffle4(colour_order, idum); 
    for (int pass = 0; pass < 4; ++pass) {
        int colour = colour_order[pass]; 
        #pragma omp parallel copyin( ran1_idum2, ran1_iy, ran1_iv, gasdev_iset, gasdev_gset )
        { 
            long  seed = initRan() + omp_get_thread_num();
            long  idum_thread = -seed;
            ran1(&idum_thread);         
            gasdev(&idum_thread);
            // Assigning variables for core-specific phi changes & dU
            double *dPhiA   = calloc(Mg, sizeof(double));
            double *dPhiB   = calloc(Mg, sizeof(double));
            // double *refEcell = calloc(8, sizeof(double)); 
            // int *affected_grid_idx = calloc(8, sizeof(int));
            // double *PhiAtemp = calloc(Mg, sizeof(double));
            // double *PhiBtemp = calloc(Mg, sizeof(double));
            // double  dU_local = 0.0; //set initial local fluctuations at zero
            long long attempted = 0 , accepted = 0; 
            
            // Initialize grid arrays
            for (int p = 0; p < Mg; ++p) {
                dPhiA[p] = 0.0;
                dPhiB[p] = 0.0;
            }

            // // double PhiAtemp[Mg], PhiBtemp[Mg];  
            // for (int grid = 0; grid < Mg; ++grid) {
            //     PhiAtemp[grid] = PhiA[grid];
            //     PhiBtemp[grid] = PhiB[grid];
            // } // Copy current densities to temporary arrays for removal        

            // // double refEcell[8]; 
            // // int affected_grid_idx[8]; 
            // for (int grid = 0; grid < 8; ++grid) {
            //     refEcell[grid] = 0.0;
            //     affected_grid_idx[grid] = -1; 
            // } 

    #pragma omp for schedule(static) //dividing the jobs inside next for loop evenly onto number of cores/threads
            for (int i = 0; i < n; ++i) {
                for(int j = 0; j < N; ++j) {
                //int    chain = i / N;
                int b = i * N + j; 
                int ix = (int)floor(beadX[b] / dL);
                int iy = (int)floor(beadY[b] / dL);
                int myColour = (ix & 1) | ((iy & 1) << 1); 
                if (myColour != colour) continue; 

                attempted++; 

                double U_bond_old = bondEnergy(b, N, beadX, beadY, juncX, juncY,
                                            Jbond, nj, b2, b2a, Lx, Ly, dim);
                gridPointInterpolator oldS = makeInterpolator(beadX[b], beadY[b], dL, Mx, My);

                // // Get reference energy at the old stencil 
                // int n_affected = 0; 
                // for (int l = 0; l < 4; ++l) {
                //     affected_grid_idx[l] = oldS.idx[l];
                //     refEcell[n_affected] = localEnergy(PhiA[oldS.idx[l]], PhiB[oldS.idx[l]], chiN, kappaN); // Old stencil before removal 
                //     n_affected++; 
                // }

                // // Remove old density contribution
                // if (i % 2 == 0) { // Even chains are A 
                //     depositPhi(PhiA, &oldS, dphi, Mg, -1);
                // } else { // Odd chains are B 
                //     depositPhi(PhiB, &oldS, dphi, Mg, -1);
                // } // This density removal action changes the field energy at the old stencil 

    //             double U_field_old = 0.0;
    //             for (int m = 0; m < 4; ++m){
    //                // U_field_old +=Ufield[oldS.idx[m]] * pref;
    //                 U_field_old += localEnergy(PhiA[oldS.idx[m]], PhiB[oldS.idx[m]], chiN, kappaN);
    // //Ufield[oldS.idx[m]] * pref;
    //              } // sum of ( reading cells E density * pref * weight)
    //             double U_old = U_bond_old + U_field_old;

                // trial move
                double x_old = beadX[b], y_old = beadY[b];
                beadX[b] += 0.05 * gasdev(&idum_thread);
                beadY[b] += 0.05 * gasdev(&idum_thread);
                wrap2D(&beadX[b], &beadY[b], Lx, Ly);

                // new bond + new field
                double U_bond_new = bondEnergy(b, N, beadX, beadY, juncX, juncY, // calling bondEnergy with updated positions
                                            Jbond, nj, b2, b2a, Lx, Ly, dim);
                gridPointInterpolator newS = makeInterpolator(beadX[b], beadY[b], dL, Mx, My); //interpolater with new positions

                // // Get reference energy at the new stencil and check if old and new stencils overlap at some grid points 
                // for (int k = 0; k < 4; ++k) {
                //     int seen = 0; 
                //     for (int g = 0; g < n_affected; ++g) {
                //         if (affected_grid_idx[g] == newS.idx[k]) { 
                //             // Here the reference energy is from Phi's at the old stencil before removal 
                //             // Later on for calculating energy difference, we use the Phi's at the new stencil, which incorporates both the weight accociated with old-removal and new-deposit 
                //             seen = 1;
                //             break;
                //         }
                //     }
                //     if (!seen) {
                //         affected_grid_idx[n_affected] = newS.idx[k];
                //         refEcell[n_affected] = localEnergy(PhiAtemp[newS.idx[k]], PhiBtemp[newS.idx[k]], chiN, kappaN); // New stencil before deposit 
                //         n_affected++;
                //     }
                // }

                // // Add new density contribution
                // if (i % 2 == 0) {
                //     depositPhi(PhiA, &newS, dphi, Mg, 1);
                // } else {
                //     depositPhi(PhiB, &newS, dphi, Mg, 1);
                // } // This density deposit action changes the field energy at the new stencil 

                // // Compute delta U_field
                // double delta_U_field = 0.0;
                // for (int k = 0; k < n_affected; ++k) { 
                //     delta_U_field += (localEnergy(PhiAtemp[affected_grid_idx[k]], PhiBtemp[affected_grid_idx[k]], chiN, kappaN) - refEcell[k]) * pref;
                // } 

                // cells affected by removal 
                double delta_U_field = 0.0;
                for (int m = 0; m < 4; ++m) {
                    int q = oldS.idx[m];
                    // freeze
                    double phiA_ref = PhiA[q], phiB_ref = PhiB[q];
                    double phiA_new = phiA_ref, phiB_new = phiB_ref;

                    // // not frozen
                    // double phiA_ref = PhiA[q] + dPhiA[q];
                    // double phiB_ref = PhiB[q] + dPhiB[q];
                    // double phiA_new = phiA_ref, phiB_new = phiB_ref;

                    if ((i % 2) == 0) phiA_new -= dphi * oldS.w[m];
                    else                phiB_new -= dphi * oldS.w[m];

                    delta_U_field += (localEnergy(phiA_new, phiB_new, chiN, kappaN) - localEnergy(phiA_ref, phiB_ref, chiN, kappaN));
                }
                // cells affected by deposit 
                for (int m = 0; m < 4; ++m) {
                    int r = newS.idx[m];
                    // freeze 
                    double phiA_ref = PhiA[r], phiB_ref = PhiB[r];
                    double phiA_new = phiA_ref, phiB_new = phiB_ref;

                    // // not frozen 
                    // double phiA_ref = PhiA[r] + dPhiA[r];
                    // double phiB_ref = PhiB[r] + dPhiB[r];
                    // double phiA_new = phiA_ref, phiB_new = phiB_ref;

                    if ((i % 2) == 0) phiA_new += dphi * newS.w[m];
                    else                phiB_new += dphi * newS.w[m];

                    delta_U_field += (localEnergy(phiA_new, phiB_new, chiN, kappaN) - localEnergy(phiA_ref, phiB_ref, chiN, kappaN));
                }

                delta_U_field *= (double)pref; 


                // double U_field_new = 0.0;
                // for (int m = 0; m < 4; ++m) {
                //    // U_field_new += Ufield[newS.idx[m]] * pref;
                //     U_field_new += localEnergy(PhiA[newS.idx[m]], PhiB[newS.idx[m]], chiN, kappaN);
                //   //  U_field_new += Ufield[m] * pref;
                // } //Same as before, I'm reading each cells density and performing the operations on it, then summing over the four
                double dU = (U_bond_new - U_bond_old) + delta_U_field; //Change in energy is our new energy state minus the previous one

                // Metropolis
                bool accept = (dU <= 0.0)
                        || (ran1(&idum_thread) < exp(-beta * dU)); // drawing random number directly inside this line 
                if (accept) {
                    // double signA = 1 - (i % 2);//need to check whether this is an A chain or B chain so I set even is A, update A not B, and vice versa
                    // double signB = (i % 2);
                    accepted++; 
                
                    // // record density change
                    // for (int m = 0; m < 4; ++m) {
                    //     dPhiA[oldS.idx[m]] -= signA * dphi * oldS.w[m]; //removing old contribution
                    //     dPhiB[oldS.idx[m]] -= signB * dphi * oldS.w[m];
                    //     dPhiA[newS.idx[m]] += signA * dphi * newS.w[m]; //adding new contribution
                    //     dPhiB[newS.idx[m]] += signB * dphi * newS.w[m];
                    // }
                    // dU_local += (U_field_new - U_field_old);

                    for (int m = 0; m < 4; ++m) {
                        int q_old = oldS.idx[m];
                        int q_new = newS.idx[m];

                        if ((i % 2) == 0) { 
                            dPhiA[q_old] -= dphi * oldS.w[m];
                            dPhiA[q_new] += dphi * newS.w[m];
                        } else { 
                            dPhiB[q_old] -= dphi * oldS.w[m];
                            dPhiB[q_new] += dphi * newS.w[m];
                        }
                    }                    
                } else {
                    // undo coords for rejected case
                    beadX[b] = x_old; beadY[b] = y_old; 
               
                }            

                // // freeze the density changes 
                // for(int p = 0; p < n_affected; ++p) {
                //     int g = affected_grid_idx[p]; 
                //     PhiAtemp[g] = PhiA[g]; 
                //     PhiBtemp[g] = PhiB[g]; 
                // }

            }
            //  #pragma omp critical
            //    U_field_total += dU_local;
        //      static volatile double dummy; 
        //       dummy = dU_local; 
            
        //        U_field_total += dU_local;
            } // end omp for

                        // adding it back into globals
                #pragma omp atomic
                        total_attemptedBead += attempted;

                #pragma omp atomic
                        total_acceptedBead  += accepted;

                #pragma omp critical
                        for (int j = 0; j < Mg; ++j) {
                            PhiA[j] += dPhiA[j];
                            PhiB[j] += dPhiB[j];
                        }
                // #pragma omp critical 
                //     for (int j = 0; j < Mg; ++j) {
                //         Ufield[j] = localEnergy(PhiA[j], PhiB[j], chiN, kappaN);
                //     }
        //    } // end omp for
            free(dPhiA);
            free(dPhiB);
            // free(refEcell); 
            // free(affected_grid_idx);
            // free(PhiAtemp);
            // free(PhiBtemp);
        }
    }

        // Junction moves
     

#pragma omp parallel copyin( ran1_idum2, ran1_iy, ran1_iv, gasdev_iset, gasdev_gset )
{ 
        long  seed = initRan() + omp_get_thread_num();
        long  idum_thread = -seed;
        ran1(&idum_thread);         
        gasdev(&idum_thread);
        long long attempted = 0, accepted = 0;
#pragma omp for schedule(static)
        for (i = 0; i < nj; ++i) {
            ++attempted;
            
            double JBond_old = junctionBondEnergy(i, beadX, beadY, juncX, juncY, Jbond, b2a, Lx, Ly, dim);
            double x_old = juncX[i], y_old = juncY[i];
            juncX[i] += 0.05 * gasdev(&idum_thread);
            juncY[i] += 0.05 * gasdev(&idum_thread);
            wrap2D(&juncX[i], &juncY[i], Lx, Ly);
            double JBond_new = junctionBondEnergy(i, beadX, beadY, juncX, juncY, Jbond, b2a, Lx, Ly, dim);
            double dU = JBond_new - JBond_old;

            bool accept = (dU <= 0.0)
            || (ran1(&idum_thread) < exp(-beta * dU));
            if (accept) {
            ++accepted;
               // record density change
            }
            else {
                juncX[i] = x_old;
                juncY[i] = y_old;
            }


        }
#pragma omp atomic
        total_attemptedJunc += attempted;

#pragma omp atomic
        total_acceptedJunc += accepted;
// Compute junction density (PhiJ)
        // This is just for checking where the junctions are; i.e. 
        // Junction itself does not contribute to the overall density, essentially treated volumeless 

    }
        // Free end swaps
        if (t % t_rebuild == 0 || need_rebuild) {
            rebuild_cell_list(nfe, Ncell,
                            beadX, beadY, FEnd,
                            cellHead, cellNext,
                            Lx, Ly, Lcell, Nxcell, Nycell);

            need_rebuild = 0;
        }

        // attempt one swap per free end  
        for (int J = 0; J < nj; ++J) {
            ++prop_swap;
            int status = attemptSwapMove(J, idum, beta, nj, nfe, r_swap,
                                        Lx, Ly, dim, b2a,
                                        Lcell, Nxcell, Nycell,
                                        beadX, beadY, juncX, juncY,
                                        Jbond, FEnd,
                                        cellHead, cellNext);

            if (status >= 0) {
                ++tried_swap; 
                if (status == 1) {
                    ++acc_swap; 
                    need_rebuild = 1;
                }
            }  
        }

        // Compute junction density (PhiJ)
        // This is just for checking where the junctions are; i.e. 
        // Junction itself does not contribute to the overall density, essentially treated volumeless 
        for (i = 0; i < Mg; ++i) {
            PhiJ[i] = 0.0;
        }
        for (j = 0; j < nj; ++j) {
            gridPointInterpolator s = makeInterpolator(juncX[j], juncY[j], dL, Mx, My);
            depositPhi(PhiJ, &s, dphi, Mg, 1);
        }

        // Accumulate average densities
        for (i = 0; i < Mg; ++i) {
            PhiAave[i] += PhiA[i];
            PhiBave[i] += PhiB[i];
            PhiJave[i] += PhiJ[i];
            phicount[i]++;
        }

        // Compute junction IDs
        for (i = 0; i < nj; ++i) {
            JID[i] = 0;
            for (j = 0; j < 3; ++j) {
                int beadIndex = Jbond[i * 3 + j];
                int chainIndex = beadIndex / N;
                JID[i] += chainIndex % 2;
            }
        }

        // Update CONNECT array
        for (i = 0; i < 2 * n; ++i) {
            CONNECT[i] = -1;
        }
        for (i = 0; i < 3 * nj; ++i) {
            index = Jbond[i];
            Jindex = i / 3;
            int chain = index / N;
            if (index % N == 0) {
                CONNECT[2 * chain] = JID[Jindex];
            }
            if (index % N == N - 1) {
                CONNECT[2 * chain + 1] = JID[Jindex];
            }
        }

        // Populate CIJ matrix
        for (i = 0; i < n; ++i) {
            if (i % 2 == 0) {
                int end0 = CONNECT[2 * i];
                int end1 = CONNECT[2 * i + 1];
                if (end0 == 0 && end1 == 0) CIJ[0]++;
                if ((end0 == 0 && end1 == 1) || (end0 == 1 && end1 == 0)) CIJ[1]++;
                if ((end0 == 0 && end1 == 2) || (end0 == 2 && end1 == 0)) CIJ[2]++;
                if ((end0 == 0 && end1 == 3) || (end0 == 3 && end1 == 0)) CIJ[3]++;
                if (end0 == 1 && end1 == 1) CIJ[4]++;
                if ((end0 == 1 && end1 == 2) || (end0 == 2 && end1 == 1)) CIJ[5]++;
                if ((end0 == 1 && end1 == 3) || (end0 == 3 && end1 == 1)) CIJ[6]++;
                if (end0 == 2 && end1 == 2) CIJ[7]++;
                if ((end0 == 2 && end1 == 3) || (end0 == 3 && end1 == 2)) CIJ[8]++;
                if (end0 == 3 && end1 == 3) CIJ[9]++;
            }
        }

        // Update junction type distribution
        for (i = 0; i < nj; ++i) {
            int num_A_bound = JID[i];
            JIDist[4 * i + num_A_bound]++;
        }

        // Compute bond lengths for averaging
        bondave = 0.0; bondcount = 0;
        for (i = 0; i < n; ++i) {
            for (j = 1; j < N; ++j) {
                dmin2D(beadX[i * N + j], beadY[i * N + j], 
                       beadX[i * N + j - 1], beadY[i * N + j - 1], Lx, Ly, &dx, &dy);
                r2 = dx * dx + dy * dy;
                bondave += r2; bondcount++;
            }
        }
        for (j = 0; j < 3 * nj; ++j) {
            int b = Jbond[j];
            int J = j / 3;
            dmin2D(beadX[b], beadY[b], juncX[J], juncY[J], Lx, Ly, &dx, &dy);
            r2 = dx * dx + dy * dy;
            bondave += r2; bondcount++;
        }

        // Output every 10 timesteps
        if (t % 50 == 0) {
            files[0] = fopen(strs[0], "w");
            // prefix "PHI_" 
            if (files[0] != NULL) {
                fprintf(files[0], "%d\n", t);
                for (i = 0; i < Mx; ++i) {
                    for (j = 0; j < My; ++j) {
                        int q = j * Mx + i;
                        fprintf(files[0], "%lf %lf %lf %lf ", 
                                PhiA[q], PhiB[q], PhiA[q] + PhiB[q], PhiJ[q]);
                    }
                    fprintf(files[0], "\n");
                }
                fprintf(files[0], "\n\n");
                fclose(files[0]);
            }
            files[1] = fopen(strs[1], "w");
            // prefix "Coord_" 
            if (files[1] != NULL) {
                fprintf(files[1], "%d\n", t);
                for (i = 0; i < Ntot; ++i) {
                    fprintf(files[1], "%lf %lf\n", beadX[i], beadY[i]);
                }
                fprintf(files[1], "\n\n");
                fclose(files[1]);
            }
            files[2] = fopen(strs[2], "w");
            // prefix "Misc_" 
            if (files[2] != NULL) {
                fprintf(files[2], "%d\n", t);
                // bead coordinates of the first chain 
                for (i = 0; i < N; ++i) {
                    fprintf(files[2], "%lf %lf ", beadX[i], beadY[i]);
                }
                fprintf(files[2], "\n");
                fprintf(files[2], "%lf\n", bondcount > 0 ? bondave / (double)bondcount : 0.0);
                // for (i = 0; i < 3 * nj; ++i) {
                //     fprintf(files[2], "%d ", Jbond[i]);
                // }
                // fprintf(files[2], "\n");
                // for (i = 0; i < nfe; ++i) {
                //     fprintf(files[2], "%d ", FEnd[i]);
                // }
                // junction coordinates
                fprintf(files[2], "\n");
                for (i = 0; i < 3 * nj; ++i) {
                    fprintf(files[2], "%lf %lf ", beadX[Jbond[i]], beadY[Jbond[i]]);
                }
                fprintf(files[2], "\n");
                // free end coordinates
                for (i = 0; i < nfe; ++i) {
                    fprintf(files[2], "%lf %lf ", beadX[FEnd[i]], beadY[FEnd[i]]);
                }
                fprintf(files[2], "\n");
                fclose(files[2]);
            }
            files[3] = fopen(strs[3], "w"); 
            // prefix "juncDistr_" 
            if (files[3] != NULL) {
                for (i = 0; i < 4; ++i) {
                    for (j = 0; j < nj; ++j) {
                        fprintf(files[3], "%d ", JIDist[4 * j + i]);
                    }
                    fprintf(files[3], "\n");
                }
                fprintf(files[3], "\n\n");
                fclose(files[3]);
            }
            files[4] = fopen(strs[4], "w"); 
            // prefix "CIJ_" 
            if (files[4] != NULL) {
                for (i = 0; i < 10; ++i) {
                    fprintf(files[4], "%d\n", CIJ[i]);
                }
                fclose(files[4]);
            }
        }

        // Output average densities every 1000 timesteps
        if (t % 1000 == 0) {
            files[5] = fopen(strs[5], "w"); 
            // prefix "PHIavg_" 
            if (files[5] != NULL) {
                fprintf(files[5], "%d\n", t);
                for (i = 0; i < Mx; ++i) {
                    for (j = 0; j < My; ++j) {
                        int q = j * Mx + i;
                        double norm = (phicount[q] > 0) ? (double)phicount[q] : 1.0;
                        fprintf(files[5], "%lf %lf %lf %lf ", 
                                PhiAave[q] / norm, PhiBave[q] / norm, 
                                (PhiAave[q] + PhiBave[q]) / norm, PhiJave[q] / norm);
                    }
                    fprintf(files[5], "\n");
                }
                fprintf(files[5], "\n\n");
                fclose(files[5]);
            }
            for (i = 0; i < Mg; ++i) {
                PhiAave[i] = 0.0;
                PhiBave[i] = 0.0;
                PhiJave[i] = 0.0;
                phicount[i] = 0;
            }         
        }

        if (t % 50 == 0) {
            double acc_ratio_bead = total_attemptedBead ? (double)total_acceptedBead / total_attemptedBead : 0.0;
            double acc_ratio_junc = total_attemptedJunc ? (double)total_acceptedJunc / total_attemptedJunc : 0.0;
            double neighbor_avail = prop_swap ? (double)tried_swap / prop_swap : 0.0;
            double acc_ratio_swap = tried_swap ? (double)acc_swap / tried_swap : 0.0;

            FILE *f = fopen(strs[7], "a");
            if (f) {
                fprintf(f, "%d %lf %lf %lf %lf\n", 
                    t, acc_ratio_bead, acc_ratio_junc, neighbor_avail, acc_ratio_swap);
                fclose(f);
            }

            /* zero for the next window */
            total_attemptedBead = total_acceptedBead = 0;
            total_attemptedJunc = total_acceptedJunc = 0;
            prop_swap = acc_swap = 0;
            tried_swap = 0;
        }

        // Reset CIJ every 10000 timesteps
        if (t % 10000 == 0) {
            for (i = 0; i < 10; ++i) {
                CIJ[i] = 0;
            }
        }
    }



    // Free memory
    for(int i = 0; i < NUM_STR; i++) 
    {
        free(strs[i]);
    }    
    free(beadX);
    free(beadY);
    free(juncX);
    free(juncY);
    free(JID);
    free(JIDist);
    free(CONNECT);
    free(CIJ);
    free(Jbond);
    free(FEnd);
    free(PhiA);
    free(PhiB);
    free(PhiJ);
    free(PhiAave);
    free(PhiBave);
    free(PhiJave);
    free(phicount);
    free(idum);
    free(cellHead);
    free(cellNext);

    return 0;
}
