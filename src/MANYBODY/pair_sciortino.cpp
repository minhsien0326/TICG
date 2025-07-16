/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Sarit Dutta (UIUC)
------------------------------------------------------------------------- */

#include "pair_sciortino.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define DELTA 4

/* ---------------------------------------------------------------------- */

PairSciortino::PairSciortino(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  nelements = 0;
  elements = nullptr;
  nparams = maxparam = 0;
  params = nullptr;
  elem2param = nullptr;
  map = nullptr;

  maxshort = 10;
  neighshort = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairSciortino::~PairSciortino()
{
  if (copymode) return;

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  memory->destroy(params);
  memory->destroy(elem2param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairSciortino::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum,jnumm1;
  int itype,jtype,ktype,ijparam,ikparam,ijkparam;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair,evj,evk;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fj[3],fk[3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  evj = 0.0;
  evk = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      if (rsq >= params[ijparam].cutsq) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }

      twobody(&params[ijparam],rsq,fpair,eflag,evdwl);

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;
      f[j][0] -= delx*fpair;
      f[j][1] -= dely*fpair;
      f[j][2] -= delz*fpair;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delx,dely,delz);
    }

    jnumm1 = numshort - 1;

    for (jj = 0; jj < jnumm1; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      double fjxtmp,fjytmp,fjztmp;
      fjxtmp = fjytmp = fjztmp = 0.0;

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        if (jtype != ktype) continue;
        if (itype == jtype) continue;
        ikparam = elem2param[itype][ktype][ktype];
        ijkparam = elem2param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        threebody(&params[ijparam],&params[ikparam],&params[ijkparam],
                  rsq1,rsq2,delr1,delr2,fj,fk,eflag,evdwl,evj,evk);

        fxtmp -= fj[0] + fk[0];
        fytmp -= fj[1] + fk[1];
        fztmp -= fj[2] + fk[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        if (evflag) ev_tally3_1(i,j,k,evdwl,0.0,fj,fk,delr1,delr2,evj,evk);
      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

//  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairSciortino::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSciortino::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSciortino::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // ensure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if "NULL"
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = nullptr;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSciortino::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Sciortino requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Sciortino requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->set_half(0);
  neighbor->requests[irequest]->set_full(1);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSciortino::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairSciortino::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "sciortino", unit_convert_flag);
    char * line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,
                                                            unit_convert);

    while((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;
        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                              "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }

        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].kelement = kelement;
        params[nparams].epsilon  = values.next_double();
        params[nparams].sigma    = values.next_double();
        params[nparams].powerm   = values.next_double();
        params[nparams].cut      = values.next_double();
        params[nparams].lambda   = values.next_double();
      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      if (unit_convert) {
        params[nparams].epsilon *= conversion_factor;
      }

      if (params[nparams].epsilon < 0.0 || params[nparams].sigma < 0.0 ||
          params[nparams].powerm < 0.0  || params[nparams].cut < 0.0   ||
          params[nparams].lambda < 0.0)
        error->one(FLERR,"Illegal Sciortino parameter");

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if(comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairSciortino::setup_params()
{
  int i,j,k,m,n;
  double rtmp;

  // set elem2param for all triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem2param);
  memory->create(elem2param,nelements,nelements,nelements,"pair:elem2param");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem2param[i][j][k] = n;
      }

  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter.

  for (m = 0; m < nparams; m++) {
    params[m].cutsq = params[m].cut * params[m].cut;
    params[m].lambda_epsilon = params[m].lambda*params[m].epsilon;
    params[m].epsiloninv = 1.0/params[m].epsilon;

    if (params[m].jelement == params[m].kelement){
      params[m].c1 = 4.0*params[m].epsilon * 2.0*params[m].powerm *
                    pow(params[m].sigma,(2.0*params[m].powerm));
      params[m].c2 = 4.0*params[m].epsilon * params[m].powerm *
                    pow(params[m].sigma,params[m].powerm);
      params[m].c3 = 4.0*params[m].epsilon * 
                    pow(params[m].sigma,(2.0*params[m].powerm));
      params[m].c4 = 4.0*params[m].epsilon * 
                    pow(params[m].sigma,params[m].powerm);
      rtmp = pow(2.0, (1.0/params[m].powerm)) * params[m].sigma;
      params[m].rminsq = rtmp*rtmp;

      if (params[m].ielement == params[m].jelement){
        params[m].cutsq = params[m].rminsq;
        params[m].offset = -params[m].epsilon;
      }
      else
        params[m].offset = 0.0;
    }
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++) {
    rtmp = sqrt(params[m].cutsq);
    if (rtmp > cutmax) cutmax = rtmp;
  }
}

/* ---------------------------------------------------------------------- */

void PairSciortino::twobody(Param *param, double rsq, double &fforce,
                     int eflag, double &eng)
{
  double rinvsq, rinvm, rinv2m;

  rinvsq = 1.0/rsq;
  rinvm = pow(rinvsq,(param->powerm/2.0));
  rinv2m = rinvm*rinvm;
  fforce = (param->c1*rinv2m - param->c2*rinvm) * rinvsq;

  if (eflag) eng = (param->c3*rinv2m - param->c4*rinvm) - param->offset;
}

/* ---------------------------------------------------------------------- */

void PairSciortino::threebody(Param *paramij, Param *paramik, Param *paramijk,
                       double rsq1, double rsq2,
                       double *delr1, double *delr2,
                       double *fj, double *fk, int eflag, double &eng, double &eng1, double &eng2)
{
  double fpair1, fpair2;
  double flev;

  if (rsq1 <= paramij->rminsq){
    eng1 = 1.0; fpair1 = 0.0;
  }
  else {
    twobody(paramij,rsq1,fpair1,1,eng1);
    eng1   *= -paramij->epsiloninv;
    fpair1 *= -paramij->epsiloninv;
  }

  if (rsq2 <= paramik->rminsq){
    eng2 = 1.0; fpair2 = 0.0;
  }
  else {
    twobody(paramik,rsq2,fpair2,1,eng2);
    eng2   *= -paramik->epsiloninv;
    fpair2 *= -paramik->epsiloninv;
  }

  flev = -fpair1*paramijk->lambda_epsilon*eng2;
  fj[0] = -delr1[0]*flev;
  fj[1] = -delr1[1]*flev;
  fj[2] = -delr1[2]*flev;

  flev = -fpair2*paramijk->lambda_epsilon*eng1;
  fk[0] = -delr2[0]*flev;
  fk[1] = -delr2[1]*flev;
  fk[2] = -delr2[2]*flev;

  if (eflag) eng = paramijk->lambda_epsilon * eng1 * eng2;
}
