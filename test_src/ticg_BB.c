#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h> 
#include <errno.h>
#include <sys/stat.h> 
#include <stdbool.h>

// ---------- constants ---------------------------------------------------------------------------- 
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
#define STR_LEN 256
#define NUM_STR 4
#define MAX_DU 700.0 // Threshold for Metropolis test to avoid exp overflow
#define SYSTEM "BB"
#define DIM 3 // dimensionality (1, 2, or 3) 
#define VERSION "1.0" 
// ---------- model parameters --------------------------------------------------------------------- 
#define N_TYPE       2                      // number of species (A and B) 
#define N_CHAIN      300                    // number of chains for each type 
#define N_BEAD       32                     // degree of polymerisation (number beads per chain) 
#define BOX_LENGTH   10.0                   // simulation box length (in unit of Re) 
#define B2           1.0                    // bond length squared (reference length scale) 
#define DELTA_L      0.20                   // density cloud size ( = a_int in paper) 
#define KAPPA_N      50.0                   // \kappa N for compressibility
#define CHI_0_N      0.5                    // “bare” \chi_0 N 
#define BETA         1.0                    // inverse temperature (1/k_BT) 
#define F_DISP       0.05                   // prefactor for trial displacement 
#define MG           16                     // number of grid points along each dimension (for visualizing density fields) 
#define NGRID        (MG*MG*MG)             // total number of grid points 
// ---------- derived constants ------------------------------------------------
#define N_CHAIN_TOT  (N_CHAIN * N_TYPE)                                         // total number of chains 
static const double  INV_CLOUD_VOL = 1 / (DELTA_L * DELTA_L * DELTA_L);         // cache the normalization factor for density cloud volume
static const double  sqrt_N_bar = (double) (N_CHAIN_TOT * N_BEAD) / (BOX_LENGTH * BOX_LENGTH * BOX_LENGTH) / (double) N_BEAD; 
/*
\sqrt{\bar{N}} is the invariant degree of polymerization, defined as 
        \sqrt{\bar{N}} = \rho_0 R_e^3 / N
where \rho_0 is the bead number density, R_e is the end-to-end distance of a chain, and N is the degree of polymerization. 

\rho_0 = nN / V; V = L^3 
\sqrt{\bar{N}} = \rho_0 R_e^3 / N
               = (nN / V) (R_e^3 / N)
               = nN / (L^3 / R_e^3) / N 
               = n / (L^3 / R_e^3)
               = n / BOX_LENGTH^3
where BOX_LENGTH is the length of the simulation box in units of R_e. 
*/
// ---------- data structures ---------------------------------------------------------------------- 
typedef struct {
    double x, y, z;
} vec3;

typedef struct {
    vec3   r[N_BEAD]; // bead positions
    int    type[N_BEAD]; // 0 = A, 1 = B
} chain;

static chain *poly; // global pointer to the array of chains 
static double box = BOX_LENGTH; 

// indices and weights for particle in mesh interpolation (only for outputting density fields) 
typedef struct {
    int    idx[8]; 
    double w  [8]; 
} gridPointInterpolator;

// ---------- helper functions --------------------------------------------------------------------- 
// build gridPointInterpolator for point (x,y,z) in a grid of size MG*MG*MG with grid spacing BOX_LENGTH/MG 
static gridPointInterpolator mkInterp(double x, double y, double z) {
    double h = BOX_LENGTH / MG; 
    double gx = x / h, gy = y / h, gz = z / h; 
    int ix0 = (int) floor(gx);
    int iy0 = (int) floor(gy);
    int iz0 = (int) floor(gz);
    // fractions 
    double fx = gx - ix0, fy = gy - iy0, fz = gz - iz0; 
    // wrap 
    ix0 = (ix0 % MG + MG) % MG;
    iy0 = (iy0 % MG + MG) % MG;
    iz0 = (iz0 % MG + MG) % MG;
    int ix1 = (ix0 + 1) % MG;
    int iy1 = (iy0 + 1) % MG;
    int iz1 = (iz0 + 1) % MG;

    gridPointInterpolator s;
    // 3D bilinear indices and weights
    s.idx[0] = ix0 + iy0 * MG + iz0 * MG * MG; s.w[0] = (1.0 - fx) * (1.0 - fy) * (1.0 - fz);
    s.idx[1] = ix0 + iy1 * MG + iz0 * MG * MG; s.w[1] = (1.0 - fx) * fy * (1.0 - fz);
    s.idx[2] = ix1 + iy0 * MG + iz0 * MG * MG; s.w[2] = fx * (1.0 - fy) * (1.0 - fz);
    s.idx[3] = ix1 + iy1 * MG + iz0 * MG * MG; s.w[3] = fx * fy * (1.0 - fz);
    s.idx[4] = ix0 + iy0 * MG + iz1 * MG * MG; s.w[4] = (1.0 - fx) * (1.0 - fy) * fz;
    s.idx[5] = ix0 + iy1 * MG + iz1 * MG * MG; s.w[5] = (1.0 - fx) * fy * fz;
    s.idx[6] = ix1 + iy0 * MG + iz1 * MG * MG; s.w[6] = fx * (1.0 - fy) * fz;
    s.idx[7] = ix1 + iy1 * MG + iz1 * MG * MG; s.w[7] = fx * fy * fz;
    // Ensure indices are non-negative
    for (int i = 0; i < 8; ++i) {
        if (s.idx[i] < 0 || s.idx[i] >= NGRID) {
            fprintf(stderr, "Invalid grid index: %d at i=%d\n", s.idx[i], i);
            exit(EXIT_FAILURE);
        }
    }
    return s;
}

static void depositPhi(double *Phi, const gridPointInterpolator *s, double dphi, int sign) {
    // Deposit or remove dphi onto eight grid points (stencil) in the gridPointInterpolator
    for (int i = 0; i < 8; ++i) {
        Phi[s->idx[i]] += sign * dphi * s->w[i];
    }
}

// minimum‑image convention for periodic boundary conditions 
static double mic(double d) {
    if (d >  0.5*box) d -= box;
    if (d < -0.5*box) d += box;
    return d;
}

// squared distance under PBC
static double rsq(const vec3 *a, const vec3 *b, vec3 *dr_out) {
    vec3 dr = { mic(a->x - b->x), mic(a->y - b->y), mic(a->z - b->z) };
    if (dr_out) *dr_out = dr;
    return dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
}

// bonded energy for one bead 
static double bonded_energy_one_bead(int c, int i) {
    const chain *ch = &poly[c]; 
    double e = 0.0; 
    if (i > 0) {
        vec3 dr; double r2 = rsq(&ch->r[i], &ch->r[i-1], &dr);
        e += 0.5 * DIM * r2 / B2; // bond to previous bead 
    }
    if (i < N_BEAD-1) {
        vec3 dr; double r2 = rsq(&ch->r[i], &ch->r[i+1], &dr);
        e += 0.5 * DIM * r2 / B2; // bond to next bead 
    }
    return e;
}

// To be packed into src/MANYBODY/pair_ticg.cpp for LAMMPS ===================== 
// overlap of two cubic clouds along one dimension 
static inline double overlap_1d(double d) {
    d = fabs(d);
    if (d >= DELTA_L) return 0.0;
    return DELTA_L - d; // linear overlap length 
}

// pair potential energy 
static double pair_energy(const vec3 *ri, const vec3 *rj,
                          int type_i, int type_j) {

    vec3 dr; // minimum-image vector 
    rsq(ri, rj, &dr); 

    // volume overlap for cubic clouds 
    double v_overlap = overlap_1d(dr.x) *
                       overlap_1d(dr.y) *
                       overlap_1d(dr.z); 
    v_overlap *= INV_CLOUD_VOL; // normalize to unit volume
    // convert overlap to local‐density contribution 
    double e = 0.0;
    // incompatibility (chi term) 
    if (type_i != type_j) {
        e += CHI_0_N * v_overlap; 
    } else {
        e += 0; // no penalty for same type 
    }
    // incompressibility (kappa term) 
    e += KAPPA_N * v_overlap; 
    return e;
}
// ============================================================================= 

// non-bonded energy for a given bead 
static double nonbonded_energy_one_bead(int c_idx, int i_idx)
{
    const vec3 *ri = &poly[c_idx].r[i_idx];
    int  ti =  poly[c_idx].type[i_idx];

    double e = 0.0;
    for (int c = 0; c < N_CHAIN_TOT; ++c)
        for (int j = 0; j < N_BEAD; ++j) {
            if (c == c_idx && j == i_idx) continue; // skip self
            const vec3 *rj = &poly[c].r[j];
            int tj = poly[c].type[j];
            e += pair_energy(ri, rj, ti, tj);
        }
    return e;
}

// wrap coordinates back into the box 
static void wrap(vec3 *r) {
    if (r->x < 0) r->x += box; if (r->x >= box) r->x -= box;
    if (r->y < 0) r->y += box; if (r->y >= box) r->y -= box;
    if (r->z < 0) r->z += box; if (r->z >= box) r->z -= box;
}

// initialize random number generator
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

// generate random number between 0 and 1 
float ran1(long *idum)
{
    int j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    float temp;
    
    if (*idum <= 0)
    {
        if (-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for (j = NTAB + 7; j >= 0; --j)
        {
            k = (*idum) / IQ1;
            *idum = IA1 * (*idum - k * IQ1) - k * IR1;
            if (*idum < 0) *idum += IM1;
            if (j < NTAB) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1;
    *idum = IA1 * (*idum - k * IQ1) - k * IR1;
    if (*idum < 0) *idum += IM1;
    k = idum2 / IQ2;
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
    if (idum2 < 0) idum2 += IM2;
    j = iy / NDIV;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if (iy < 1) iy += IMM1;
    if ((temp = AM * iy) > RNMX) return RNMX;
    else return temp;
}

// generate normally distributed random number
float gasdev(long *idum)
{
    static int iset = 0;
    static float gset;
    float fac, rsq, v1, v2;
    
    if (*idum < 0) iset = 0;
    if (iset == 0)
    {
        do
        {
            v1 = 2.0 * ran1(idum) - 1.0;
            v2 = 2.0 * ran1(idum) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        }
        while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;
        return v2 * fac;
    }
    else
    {
        iset = 0;
        return gset;
    }
}

// function for creating a directory if it does not exist 
void create_directory(const char* path) {
    if (mkdir(path, 0777) == -1) {
        if (errno != EEXIST) {
            printf("Error creating directory: %s\n", path);
            exit(1);
        }
    }
    printf("Directory created or already exists: %s\n", path);
}

// function to compute the greatest common divisor (GCD) of two integers 
static int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// function to initialize the system 
static void init_sys(void) {
    poly = calloc(N_CHAIN_TOT, sizeof(chain)); 
    if (!poly) {
        fprintf(stderr, "Memory allocation failure\n");
        exit(EXIT_FAILURE);
    } 

    int c = 0; // counter for total number of chains  
    for (int type = 0; type < N_TYPE; ++type) {
        // initialize random number generator for each type
        long seed = initRan();
        for (int k = 0; k < N_CHAIN; ++k, ++c) { // total chain counter c increments here 
            poly[c].type[0] = type; // set type for first bead 
            // place first bead at a random position in the box 
            poly[c].r[0].x = ran1(&seed) * box;
            poly[c].r[0].y = ran1(&seed) * box;
            poly[c].r[0].z = ran1(&seed) * box;
            // place the subsequent beads at a distance DELTA_L from the previous one 
            for (int i = 1; i < N_BEAD; ++i) {
                poly[c].type[i] = type; // set type for subsequent beads 
                poly[c].r[i].x = poly[c].r[i-1].x + DELTA_L * gasdev(&seed);
                poly[c].r[i].y = poly[c].r[i-1].y + DELTA_L * gasdev(&seed);
                poly[c].r[i].z = poly[c].r[i-1].z + DELTA_L * gasdev(&seed);
                // wrap coordinates back into the box
                wrap(&poly[c].r[i]);
            }
        }
    }
}

void computePHI(double *PHIA, double *PHIB) {     
    double dphi = MG / (double)(N_CHAIN_TOT * N_BEAD); // density increment per bead 

    for (int c = 0; c < N_CHAIN_TOT; ++c) {
        for (int i = 0; i < N_BEAD; ++i) {
            gridPointInterpolator s = mkInterp(poly[c].r[i].x, poly[c].r[i].y, poly[c].r[i].z);
            if (poly[c].type[i] == 0) { 
                depositPhi(PHIA, &s, dphi, 1); 
            } else { 
                depositPhi(PHIB, &s, dphi, 1); 
            } 
        }
    }
}

// ------------------------------------------------------------------------------------------------- 

// ---------- main function ------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    // allocate memories -------------------------------------------------------
    double *PHIA = calloc(NGRID, sizeof(double));
    double *PHIB = calloc(NGRID, sizeof(double)); 
    // file IO ----------------------------------------------------------------- 
    char outputPath[STR_LEN];
    snprintf(outputPath, sizeof(outputPath), "./data_v%s/", VERSION);     

    create_directory(outputPath);

    static const char *prefixes[NUM_STR] = {
        "param_", // 0
        "Phi_", // 1
        "Coord_", // 2 
        "accRatio_", // 3 
    }; 

    FILE *files[NUM_STR];
    char *strs[NUM_STR];

    for (int i = 0; i < NUM_STR; i++) {
        int len = snprintf(NULL, 0, "%s%sX0N%.1f_v%s.txt", 
                           outputPath, prefixes[i], CHI_0_N, VERSION);
        strs[i] = malloc((len + 1)); 
        snprintf(strs[i], len + 1, "%s%sX0N%.1f_v%s.txt", 
                 outputPath, 
                 prefixes[i], 
                 CHI_0_N, 
                 VERSION);
    }
    // ------------------------------------------------------------------------- 
    files[0] = fopen(strs[0], "w");
    // prefix "param_"
    if (files[0]) {
        fprintf(files[0], "SYSTEM: %s\n", SYSTEM);
        fprintf(files[0], "DIM: %d\n", DIM);
        fprintf(files[0], "VERSION: %s\n", VERSION);
        fprintf(files[0], "N_TYPE: %d\n", N_TYPE);
        fprintf(files[0], "N_CHAIN: %d\n", N_CHAIN);
        fprintf(files[0], "N_BEAD: %d\n", N_BEAD);
        fprintf(files[0], "N_CHAIN_TOT: %d\n", N_CHAIN_TOT);
        fprintf(files[0], "BOX_LENGTH: %lf\n", BOX_LENGTH);
        fprintf(files[0], "B2: %lf\n", B2);
        fprintf(files[0], "DELTA_L: %lf\n", DELTA_L);
        fprintf(files[0], "KAPPA_N: %lf\n", KAPPA_N);
        fprintf(files[0], "CHI_0_N: %lf\n", CHI_0_N);
        fprintf(files[0], "BETA: %lf\n", BETA);
        fprintf(files[0], "F_DISP: %lf\n", F_DISP);
        fprintf(files[0], "MG: %d\n", MG);
        fprintf(files[0], "NGRID: %d\n", NGRID);
        fprintf(files[0], "INV_CLOUD_VOL: %lf\n", INV_CLOUD_VOL);
        fprintf(files[0], "sqrt_N_bar: %lf\n", sqrt_N_bar);
        fclose(files[0]);
    }

    // initialize random number generator
    long rng = initRan(); 
    // initialize system
    init_sys();

    long long prop_bead = 0, acc_bead = 0;
    long long tmax = (long long) 1e8; 
    int stride, c0; 
    for (int t = 0; t < tmax; ++t) {
        // stride method to randomly visit different chains -------------------- 
        c0 = (int) (ran1(&rng) * N_CHAIN_TOT); // random offset
        do{
            stride = 1 + (int) (ran1(&rng) * (N_CHAIN_TOT - 1)); // random stride in [1, N_CHAIN_TOT)
        } while (
            gcd(stride, N_CHAIN_TOT) != 1 // ensure that stride and N_CHAIN_TOT are coprime 
        ); 
        // ---------------------------------------------------------------------
        for(int k = 0, c = c0; k < N_CHAIN_TOT; ++k, c = (c + stride) % N_CHAIN_TOT) {
            int chain = c; 
            for (int bead = 0; bead < N_BEAD; ++bead) {
                // Bonded energy for the bead 
                double U_b = bonded_energy_one_bead(chain, bead); 
                double U_nb = nonbonded_energy_one_bead(chain, bead);

                // propose a move for the bead 
                prop_bead++; // increment proposal count
                vec3 old_pos = poly[chain].r[bead]; // store old position 
                // trial displacement 
                poly[chain].r[bead].x += F_DISP * gasdev(&rng); 
                poly[chain].r[bead].y += F_DISP * gasdev(&rng);
                poly[chain].r[bead].z += F_DISP * gasdev(&rng);
                // wrap coordinates back into the box 
                wrap(&poly[chain].r[bead]);

                // compute new energies
                double U_b_new = bonded_energy_one_bead(chain, bead);
                double U_nb_new = nonbonded_energy_one_bead(chain, bead);
                
                // Metropolis test
                double dU = (U_b_new + U_nb_new) - (U_b + U_nb); 
                bool accept = false;
                accept = (dU <= 0.0) || 
                         (dU < MAX_DU && ran1(&rng) < exp(-BETA * dU));
                if (accept) {
                    acc_bead++; 
                } else {
                    poly[chain].r[bead] = old_pos; 
                }
            }
        }


        if (t % 1 == 0) {
            // Compute PhiA and PhiB
            memset(PHIA, 0, NGRID * sizeof(double));
            memset(PHIB, 0, NGRID * sizeof(double));
            computePHI(PHIA, PHIB);
            // Write density fields to files
            files[1] = fopen(strs[1], "w");
            if (files[1]) {
                fprintf(files[1], "%d\n", t);
                for (int i = 0; i < MG; ++i) {
                    for (int j = 0; j < MG; ++j) {
                        for (int k = 0; k < MG; ++k) {
                            int idx = i + MG * (j + MG * k);
                            fprintf(files[1], "%lf %lf %lf ", 
                                    PHIA[idx], PHIB[idx], PHIA[idx] + PHIB[idx]);
                        }
                        fprintf(files[1], "\n");
                    }
                }
                fclose(files[1]);
            }
        }

        // Output acceptance ratio every 1000 steps
        if (t % 1 == 0) {
            double acc_ratio_bead = prop_bead ? (double) acc_bead / prop_bead : 0.0; 

            files[3] = fopen(strs[3], "a");
            if (files[3]) {
                fprintf(files[3], "%d %lf\n", t, acc_ratio_bead);
                fclose(files[3]);
            }

            // zero for the next window 
            prop_bead = acc_bead = 0;
        } 
    }
    free(poly); 
    for (int i = 0; i < NUM_STR; ++i) {
        free(strs[i]);
    }
    free(PHIA);
    free(PHIB);
}