#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 
------------- CHANGE LOG --------------------------------------------------------------------------- 
Version 2.0: 
In this version, the following changes have been made:
- Define dphi such that the sum of phi_A and phi_B equals 1. 
------------- HISTORY ------------------------------------------------------------------------------
Version 1.0: 
This script reads a LAMMPS dump file, extracts polymer chain information,
and computes the density fields PhiA and PhiB for two types of chains. 
*/ 

// ---------- Constants and Model Parameters -----------------------------------
#define VERSION      "2.0"                  // version of the code
#define MAX_MOL      400000                 // maximum number of molecules (chains) in the system 
#define DL           2.36                   // discretization length 
// ---------- Data Structures -------------------------------------------------- 
// chain bookkeeping 
typedef enum { NONE, CHAIN_A, CHAIN_B } ChainType; 

typedef struct {
    ChainType kind; 
    int       len; 
} ChainInfo;

// 8-point stencil container 
typedef struct {
    int    idx[8]; 
    double w  [8]; 
} gridPointInterpolator;
// ---------- Helper Functions -------------------------------------------------
static void die(const char *msg){ perror(msg); exit(EXIT_FAILURE); }

static void read_token(FILE *fp, char buf[], size_t n)
{
    if (fscanf(fp, "%s", buf) != 1) {
        fprintf(stderr, "unexpected end-of-file\n");
        exit(EXIT_FAILURE);
    }
}

// read the header of a single dump frame, return #atoms 
static size_t read_frame_header(FILE *fp,
                                long long *step,
                                double *xlo, double *xhi,
                                double *ylo, double *yhi,
                                double *zlo, double *zhi)
{
    char tok[64];

    // ITEM: TIMESTEP 
    read_token(fp, tok, sizeof tok); // ITEM: 
    fgets(tok, sizeof tok, fp); // TIMESTEP   
    fscanf(fp, "%lld", step);

    // ITEM: NUMBER OF ATOMS 
    read_token(fp, tok, sizeof tok); // ITEM:
    fgets(tok, sizeof tok, fp); // NUMBER OF ATOMS 
    size_t natoms;  fscanf(fp, "%zu", &natoms);

    //ITEM: BOX BOUNDS pp pp pp 
    read_token(fp, tok, sizeof tok); // ITEM:
    fgets(tok, sizeof tok, fp); // the rest 
    fscanf(fp, "%lf%lf %lf%lf %lf%lf",
           xlo, xhi, ylo, yhi, zlo, zhi);

    // ITEM: ATOMS id mol type mass q x y z
    read_token(fp, tok, sizeof tok); // ITEM:
    fgets(tok, sizeof tok, fp); // the rest 

    return natoms;
}

// first pass: discover chain types and lengths 
static void getMolInfo(FILE *fp,
                            ChainInfo info[MAX_MOL+1],
                            int *nA, int *nB)
{
    long long step;
    double xlo,xhi,ylo,yhi,zlo,zhi;
    size_t natoms = read_frame_header(fp, &step,
                                      &xlo,&xhi,&ylo,&yhi,&zlo,&zhi);

    for (size_t i = 0; i < natoms; ++i) {
        int id, mol, type;
        double dummy;

        // ITEM: atom_ID molecule_ID atom_type mass q x y z 
        fscanf(fp, "%d %d %d %lf %lf %lf %lf %lf",
               &id, &mol, &type, 
               &dummy, &dummy, &dummy, &dummy, &dummy); // ignore mass, q, x, y, z now 

        if (mol > MAX_MOL) {
            fprintf(stderr,"mol id %d > MAX_MOL\n",mol);
            exit(EXIT_FAILURE);
        }

        if (info[mol].kind == NONE) { // first time we see it 
            if (type == 3 || type == 4) { info[mol].kind = CHAIN_A;  ++*nA; }
            else if (type == 5 || type == 6) { info[mol].kind = CHAIN_B;  ++*nB; }
        }

        info[mol].len++; // count the number of atoms in this molecule 
    }
}

// build gridPointInterpolator for point (x,y,z) in a grid of size MG*MG*MG with grid spacing BOX_LENGTH/MG 
static gridPointInterpolator
mkInterp(double x, double y, double z,          // particle coordinates         
         double xlo, double ylo, double zlo,    // box lower edges              
         double Lx,  double Ly,  double Lz,     // box lengths                  
         int    Mx,  int    My,  int    Mz)     // # cells along each direction 
{
    // convert particle position to fractional grid coordinates 
    // shift so the box runs from 0 to L instead of xlo..xhi 
    double gx = (x - xlo) * Mx / Lx;   // gx in [0, Mx) 
    double gy = (y - ylo) * My / Ly;   // gy in [0, My) 
    double gz = (z - zlo) * Mz / Lz;   // gz in [0, Mz) 

    // integer part gives the “lower‑left‑front” lattice point 
    int ix0 = (int)floor(gx);
    int iy0 = (int)floor(gy);
    int iz0 = (int)floor(gz);

    // fractions of the grid cell 
    double fx = gx - ix0;
    double fy = gy - iy0;
    double fz = gz - iz0;

    // wrap
    ix0 = (ix0 % Mx + Mx) % Mx;
    iy0 = (iy0 % My + My) % My;
    iz0 = (iz0 % Mz + Mz) % Mz;

    int ix1 = (ix0 + 1) % Mx;
    int iy1 = (iy0 + 1) % My;
    int iz1 = (iz0 + 1) % Mz;

    // 3D bilinear indices and weights 
    gridPointInterpolator s;
    int strideY = Mx;
    int strideZ = Mx * My;

    s.idx[0] = ix0 + iy0*strideY + iz0*strideZ;  s.w[0] = (1-fx)*(1-fy)*(1-fz);
    s.idx[1] = ix0 + iy1*strideY + iz0*strideZ;  s.w[1] = (1-fx)*   fy *(1-fz);
    s.idx[2] = ix1 + iy0*strideY + iz0*strideZ;  s.w[2] =    fx *(1-fy)*(1-fz);
    s.idx[3] = ix1 + iy1*strideY + iz0*strideZ;  s.w[3] =    fx *   fy *(1-fz);

    s.idx[4] = ix0 + iy0*strideY + iz1*strideZ;  s.w[4] = (1-fx)*(1-fy)*   fz ;
    s.idx[5] = ix0 + iy1*strideY + iz1*strideZ;  s.w[5] = (1-fx)*   fy *   fz ;
    s.idx[6] = ix1 + iy0*strideY + iz1*strideZ;  s.w[6] =    fx *(1-fy)*   fz ;
    s.idx[7] = ix1 + iy1*strideY + iz1*strideZ;  s.w[7] =    fx *   fy *   fz ;

    return s;
}

static void depositPhi(double *Phi, const gridPointInterpolator *s, double dphi, int sign) {
    // Deposit or remove dphi onto eight grid points (stencil) in the gridPointInterpolator
    for (int i = 0; i < 8; ++i) {
        Phi[s->idx[i]] += sign * dphi * s->w[i];
    }
}

// ---------- Main Function ----------------------------------------------------

int main(int argc, char *argv[])
{
// ---------- File I/O --------------------------------------------------------- 
    // char inputPath[256] = "./lammps-dump/test-dump.txt";
    // snprintf(outputPath, sizeof(outputPath), "./lammps-grid/out_v%s.txt", VERSION);    

    char inputPath[256] = "./lammps-dump/bigBox_NVT_XN10_BB_new.txt"; 
    char outputPath[256] = "./lammps-grid/bigBox_NVT_XN10_BB_new.txt";     
    
    // char inputPath[256] = "./lammps-dump/bigBox_NVT_XN10_BV_Lambda1p3.txt"; 
    // char outputPath[256] = "./lammps-grid/bigBox_NVT_XN10_BV_Lambda1p3.txt"; 

    // char inputPath[256] = "./lammps-dump/bigBox_NVT_XN10_BV_Lambda1p0.txt"; 
    // char outputPath[256] = "./lammps-grid/bigBox_NVT_XN10_BV_Lambda1p0.txt"; 

    // char inputPath[256] = "./lammps-dump/bigBox_NVT_XN0_BV_new.txt"; 
    // char outputPath[256] = "./lammps-grid/bigBox_NVT_XN0_BV_new.txt";             

    FILE *in  = fopen(inputPath, "r");
    FILE *out = fopen(outputPath, "a"); 

    if (!in || !out) {die("Error opening files");}
// ---------- First pass: discover the system ---------------------------------- 
    ChainInfo info[MAX_MOL + 1] = {0}; 
    for (int i = 0; i <= MAX_MOL; ++i) {
        info[i].kind = NONE; 
        info[i].len = 0; 
    }
    int nA = 0, nB = 0; 
    getMolInfo(in, info, &nA, &nB);

    // Compute average chain lengths 
    long long totLenA = 0, totLenB = 0;

    for (int mol = 1; mol <= MAX_MOL; ++mol) {
        if (info[mol].kind == CHAIN_A) {totLenA += info[mol].len;}
        else if (info[mol].kind == CHAIN_B) {totLenB += info[mol].len;} 
    }
    int NA = nA ? (double)totLenA / nA : 0; 
    int NB = nB ? (double)totLenB / nB : 0; 

    printf("nA = %d, NA = %d; nB = %d, NB = %d\n", nA, NA, nB, NB);
    int Ntot = nA * NA + nB * NB; // total number of polymer beads 
    printf("Total number of polymer beads: %d\n", Ntot); 
// ---------- Second pass ------------------------------------------------------ 
    rewind(in); // rewind to the beginning of the file 

    // dynamic scratch arrays 
    size_t capAtoms=0, capGrid=0;
    double *x=NULL,*y=NULL,*z=NULL; int *atype=NULL;
    double *PhiA=NULL,*PhiB=NULL;

    long long step; 
    double xlo, xhi, ylo, yhi, zlo, zhi; 
    char token[64];

    while (!feof(in)) {
        size_t natoms = read_frame_header(in, &step,
                                          &xlo,&xhi,&ylo,&yhi,&zlo,&zhi);     
        if (natoms == 0) {
            printf("End of file reached or no atoms in frame.\n"); 
            break;
        }

        // grow atom arrays if needed
        if (natoms > capAtoms) {
            capAtoms = natoms;
            x = realloc(x, capAtoms * sizeof *x);
            y = realloc(y, capAtoms * sizeof *y);
            z = realloc(z, capAtoms * sizeof *z);
            atype = realloc(atype, capAtoms * sizeof *atype);
            if (!x || !y || !z || !atype) die("realloc"); 
        } 

        
        // read atom data 
        // atom_ID molecule_ID atom_type mass q x y z 
        for (size_t i = 0; i < natoms; ++i) {
            /* 
                junction: type 1 and 2 
                A chain 3444...4443: type 3 and 4 
                B chain 5666...6665: type 5 and 6 
            */ 

            int type; 

            if (fscanf(in,
                        "%*d %*d %d %*lf %*lf %lf %lf %lf", 
                        // use %*d to skip atom_ID and molecule_ID
                        // %*lf to skip mass and charge 
                        &type, &x[i], &y[i], &z[i]) != 4) {
                fprintf(stderr, "Error reading atom data at line %zu\n", i+1);
                exit(EXIT_FAILURE);
            }

            atype[i] = type; // store atom type
        }

        // mesh size for this frame 
        double  Lx = xhi - xlo, 
                Ly = yhi - ylo, 
                Lz = zhi - zlo;

        int     Mx = (int) (Lx / DL), 
                My = (int) (Ly / DL), 
                Mz = (int) (Lz / DL);

        int  NGRID = Mx * My * Mz; 

        if (NGRID > capGrid) {
            capGrid = NGRID;
            PhiA = realloc(PhiA, capGrid * sizeof *PhiA);
            PhiB = realloc(PhiB, capGrid * sizeof *PhiB);
            if (!PhiA || !PhiB) die("realloc for Phi arrays");
        } 
        memset(PhiA, 0, NGRID * sizeof *PhiA);
        memset(PhiB, 0, NGRID * sizeof *PhiB);  


        // double dphi = (double) NGRID / (double) Ntot; 

        printf("Step: %lld\n", step);
        printf("Lx = %.6f, Ly = %.6f, Lz = %.6f\n", 
               Lx, Ly, Lz);
        printf("Mx = %d, My = %d, Mz = %d\n",
                Mx, My, Mz);

        // printf("NGRID = %d, Ntot = %d, dphi = %.6f\n", 
        //        NGRID, Ntot, dphi); 
        double dphi = 1.0; 
        double normFactor = (double) Ntot / (double) NGRID; 

        // deposit density 
        for (int i = 0; i < natoms; ++i) {
            // create gridPointInterpolator for each atom 
            gridPointInterpolator s = mkInterp(x[i], y[i], z[i], 
                                                xlo, ylo, zlo, 
                                                Lx, Ly, Lz, 
                                                Mx, My, Mz);
            // deposit density increment dphi into the grid 
            if (atype[i] == 3 || atype[i] == 4) { 
                depositPhi(PhiA, &s, dphi, 1);
            } else if (atype[i] == 5 || atype[i] == 6) { 
                depositPhi(PhiB, &s, dphi, 1);
            }
        }         

        for (int i = 0; i < NGRID; ++i) {
            // normalize PhiA and PhiB to ensure they sum to 1
            PhiA[i] /= normFactor;
            PhiB[i] /= normFactor;
        }

        // TEST ---------------------------------------------------------------- 
        // // output atom positions to check the parsing 
        // for (size_t i = 0; i < natoms; ++i) {
        //     // print: timestep  atomID   x   y   z 
        //     fprintf(out, "%lld  %zu  %.8f  %.8f  %.8f\n",
        //             step, i+1, x[i], y[i], z[i]);
        // }

        // TEST ---------------------------------------------------------------- 
        // // output PhiA and PhiB to file
        // fprintf(out, "Timestep: %lld\n", step);
        // fprintf(out, "Grid size: %d x %d x %d\n", Mx, My, Mz); 
        // fprintf(out, "PhiA:\n");
        // for (size_t i = 0; i < Mx; ++i) {
        //     for (size_t j = 0; j < My; ++j) {
        //         for (size_t k = 0; k < Mz; ++k) {
        //             size_t idx = i + j * Mx + k * Mx * My;
        //             fprintf(out, "%.6f ", PhiA[idx]);
        //         }
        //         fprintf(out, "\n");
        //     }
        //     fprintf(out, "\n");
        // }
        // fprintf(out, "PhiB:\n");
        // for (size_t i = 0; i < Mx; ++i) {
        //     for (size_t j = 0; j < My; ++j) {
        //         for (size_t k = 0; k < Mz; ++k) {
        //             size_t idx = i + j * Mx + k * Mx * My;
        //             fprintf(out, "%.6f ", PhiB[idx]);
        //         }
        //         fprintf(out, "\n");
        //     }
        //     fprintf(out, "\n");
        // }
        // fprintf(out, "PhiA + PhiB:\n");
        // for (size_t i = 0; i < Mx; ++i) {
        //     for (size_t j = 0; j < My; ++j) {
        //         for (size_t k = 0; k < Mz; ++k) {
        //             size_t idx = i + j * Mx + k * Mx * My;
        //             fprintf(out, "%.6f ", PhiA[idx] + PhiB[idx]); 
        //         }
        //         fprintf(out, "\n");
        //     }
        //     fprintf(out, "\n");
        // }
        // fprintf(out, "\n");

        // TEST ---------------------------------------------------------------- 
        // printf("Global average of PhiA+PhiB: "); 
        // double globalAvg = 0.0; 
        // for (size_t i = 0; i < NGRID; ++i) {
        //     globalAvg += PhiA[i] + PhiB[i];
        // }
        // globalAvg /= NGRID;
        // printf("%.6f\n", globalAvg); 

        double sumA = 0.0, sumB = 0.0, sumAB = 0.0;
        for (int i = 0; i < NGRID; ++i) {
            sumA  += PhiA[i];
            sumB  += PhiB[i];
            sumAB += PhiA[i] * PhiB[i];
        }
        double avgA  = sumA  / NGRID;
        double avgB  = sumB  / NGRID;
        double avgAB = sumAB / NGRID;

        double Psi = 1.0 - (avgAB / (avgA * avgB));

        // print Psi to file
        fprintf(out, "%lld %lf\n", step, Psi);  
    }

    free(x); 
    free(y); 
    free(z);     
    free(atype);
    free(PhiA);
    free(PhiB);
    fclose(in);
    fclose(out);
    return EXIT_SUCCESS;
}
