/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */

#include "main.h"

#if defined(GCRODR) || defined(POLYPREC)


/*
#ifdef GCRODR
void blacs_get_(int*, int*, int*);
void blacs_pinfo_(int*, int*);
void blacs_gridinit_(int*, char*, int*, int*);
void blacs_gridinfo_(int*, int*, int*, int*, int*);
void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
void pgeqr2_PRECISION( int*, int*, MKL_PRECISION*, int*, int*, int*, MKL_PRECISION*, MKL_PRECISION*, int*, int* );
void pung2r_PRECISION( int*, int*, int*, MKL_PRECISION*, int*, int*, int*, MKL_PRECISION*, MKL_PRECISION*, int*, int* );
void blacs_gridexit_(int*);
int numroc_(int*, int*, int*, int*, int*);
int indxg2p_(int*, int*, int*, int*, int*);
#endif
*/


// inversion of a triangular matrix
void inv_tri_PRECISION(eigslvr_PRECISION_struct* eigen_struct)
{

  //#define trtri_double LAPACKE_ztrtri
  trtri_PRECISION( LAPACK_COL_MAJOR, 'U', 'N', eigen_struct->qr_k, eigen_struct->qr_R[0],
                   eigen_struct->qr_k);
}

void qr_PRECISION(eigslvr_PRECISION_struct* eigen_struct)
{

  //#define geqr2_double LAPACKE_zgeqr2
  geqr2_PRECISION( LAPACK_COL_MAJOR, eigen_struct->qr_m, eigen_struct->qr_n, eigen_struct->qr_QR[0],
                   eigen_struct->qr_lda, eigen_struct->qr_tau );

}

void q_from_qr_PRECISION(eigslvr_PRECISION_struct* eigen_struct)
{

  //#define ungqr_double LAPACKE_zungqr
  ungqr_PRECISION( LAPACK_COL_MAJOR, eigen_struct->qr_m, eigen_struct->qr_n, eigen_struct->qr_k,
                   eigen_struct->qr_QR[0], eigen_struct->qr_lda, eigen_struct->qr_tau );

}

void gen_eigslvr_PRECISION(eigslvr_PRECISION_struct* eigen_struct)
{

  eigen_struct->info = ggev_PRECISION( LAPACK_COL_MAJOR, eigen_struct->jobvl, eigen_struct->jobvr,
                                       eigen_struct->N, eigen_struct->A, eigen_struct->lda,
                                       eigen_struct->B, eigen_struct->ldb, eigen_struct->w, eigen_struct->beta,
                                       eigen_struct->vl, eigen_struct->ldvl, eigen_struct->vr, eigen_struct->ldvr );
}

void eigslvr_PRECISION(eigslvr_PRECISION_struct* eigen_struct)
{

  eigen_struct->info = geev_PRECISION( LAPACK_ROW_MAJOR, eigen_struct->jobvl, eigen_struct->jobvr,
                                      eigen_struct->N, eigen_struct->A, eigen_struct->lda, eigen_struct->w,
                                      eigen_struct->vl, eigen_struct->ldvl, eigen_struct->vr, eigen_struct->ldvr );       
}

void dirctslvr_PRECISION(dirctslvr_PRECISION_struct* dirctslvr)
{

    memcpy( dirctslvr->x, dirctslvr->b, sizeof(complex_PRECISION)*(dirctslvr->N) );  

    dirctslvr->info = gesv_PRECISION( LAPACK_COL_MAJOR, dirctslvr->N, dirctslvr->nrhs, dirctslvr->Hcc, 
                                    dirctslvr->lda, dirctslvr->ipiv, dirctslvr->x, dirctslvr->ldb ); 
}

#ifdef GCRODR

// c is the solution, b is ultimately the output
void gels_via_givens_PRECISION( int ida, int idb, complex_PRECISION* a, int lda, complex_PRECISION* b, int ldb, complex_PRECISION* c, int k, int m, gmres_PRECISION_struct *p ) {

  int i,j;

  //if ( g.on_solve==0 ) {
  //  gels_PRECISION( LAPACK_COL_MAJOR, 'N', ida, idb, 1, a, lda, b, ldb );
  //  return;
  //}

  // solve via Givens rotations

  complex_PRECISION *Gii = p->gcrodr_PRECISION.lsp_diag_G;

  complex_PRECISION **Hc = p->gcrodr_PRECISION.eigslvr.Hc;
  complex_PRECISION **H  = p->gcrodr_PRECISION.lsp_H;
  for ( j=0;j<m;j++ ) {
    memcpy( H[j], Hc[j], sizeof(complex_PRECISION)*(m+1) );
  }

  complex_PRECISION **B  = p->gcrodr_PRECISION.ort_B;
  complex_PRECISION *x   = c;

  // apply Givens over the Hessenberg part to turn it into triangular
  complex_PRECISION si, ci, denom, bbuff;
  complex_PRECISION *bH = b+k;
  complex_PRECISION Hloc[2];

  for ( i=0;i<m;i++ ) {

    // compute Givens coefficients
    denom = sqrt( cabs_PRECISION(H[i][i+1])*cabs_PRECISION(H[i][i+1]) + cabs_PRECISION(H[i][i])*cabs_PRECISION(H[i][i]) );
    si = H[i][i+1] / denom;
    ci = H[i][i] / denom;

    // modify rhs
    bbuff  = bH[i];
    bH[i]   = conj_PRECISION(ci)*bbuff;
    bH[i+1] = -si*bbuff;

    // modify Hessenberg matrix
    for ( j=i;j<m;j++ ) {
      Hloc[0] = H[j][i];
      Hloc[1] = H[j][i+1];
      // modification of H
      H[j][i]   =  conj_PRECISION(ci)*Hloc[0] + conj_PRECISION(si)*Hloc[1];
      H[j][i+1] = -si*Hloc[0] + ci*Hloc[1];
    }

  }

  // do triangular solve, first due to the Hessenberg matrix
  complex_PRECISION *xH = x+k;
  for ( i=(m-1);i>-1;--i ) {
    complex_PRECISION part_sum = 0.0;
    for ( j=(i+1);j<m;j++ ) {
      part_sum -= H[j][i]*xH[j];
    }
    part_sum += bH[i];
    xH[i] = part_sum/H[i][i];
  }

  // then, continue the triangular solve with the B and diagonal parts
  for ( i=(k-1);i>-1;--i ) {
    complex_PRECISION part_sum = 0.0;
    for ( j=0;j<m;j++ ) {
      part_sum -= B[j][i]*xH[j];
    }
    part_sum += b[i];
    x[i] = part_sum/Gii[i];
  }

  memcpy( b, x, sizeof(complex_PRECISION)*ldb );
}

// IMPORTANT NOTE : not a general call to p?geqr2(...)
void pqr_PRECISION( int mx, int nx, complex_PRECISION **Ax, complex_PRECISION **R, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading )
{

  int j, k = p->gcrodr_PRECISION.k;

  int start, end;
  compute_core_start_end( p->v_start, p->v_end, &start, &end, l, threading );

  // perform QR via CGS

  START_MASTER(threading)
  memset( R[0], 0.0, sizeof(complex_PRECISION)*k*k );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  for( j=0;j<k;j++ ) {

    if( j>0 ) {
      complex_PRECISION *bf = p->gcrodr_PRECISION.R[j];
      complex_PRECISION *buffer = p->gcrodr_PRECISION.Bbuff[0];
      complex_PRECISION tmpx[j];
      process_multi_inner_product_PRECISION( j, tmpx, Ax, Ax[j], p->v_start, p->v_end, l, threading );
      START_MASTER(threading)
      for ( int ww=0; ww<j; ww++ )
        buffer[ww] = tmpx[ww];
      if ( g.num_processes > 1 ) {
        PROF_PRECISION_START( _ALLR );
        MPI_Allreduce( buffer, bf, j, MPI_COMPLEX_PRECISION, MPI_SUM, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
        PROF_PRECISION_STOP( _ALLR, 1 );
      } else {
        for( int ww=0; ww<j; ww++ )
          bf[ww] = buffer[ww];
      }
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      for( int ww=0; ww<j; ww++ ) {
        vector_PRECISION_saxpy( Ax[j], Ax[j], Ax[ww], -R[j][ww], start, end, l );
      }
      SYNC_MASTER_TO_ALL(threading)

    }

    {
      PRECISION nrm_Ai = global_norm_PRECISION( Ax[j], p->v_start, p->v_end, l, threading );
      START_MASTER(threading)
      R[j][j] = nrm_Ai;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
      vector_PRECISION_real_scale( Ax[j], Ax[j], 1/creal(R[j][j]), start, end, l );
      SYNC_MASTER_TO_ALL(threading)
    }

  }

  // --------------------------------------------------------------------------------------------------------------

  // ALTERNATIVE ----> use ScaLAPACK

  /*
  // mostly taken from :
  // https://gist.github.com/leopoldcambier/be8e68906ecfd7f03edf0d809db37cc1
  // https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/scalapack-routines/scalapack-array-descriptors.html

  complex_PRECISION *a = A[0];

  int izero=0;
  int ione=1;
  int myrank_mpi, nprocs_mpi;

  MPI_Comm_rank( (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm, &myrank_mpi );
  MPI_Comm_size( (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm, &nprocs_mpi );

  int m = mx * nprocs_mpi;
  int n = nx;       // (Global) Matrix size
  int nprow = nprocs_mpi;   // Number of row procs
  int npcol = 1;   // Number of column procs
  int nb = 1;      // (Global) Block size
  char layout='C'; // Block cyclic, Column major processor mapping

  // Initialize BLACS
  int iam=0, nprocs=0;
  int zero = 0;
  int ictxt, myrow, mycol;
  blacs_pinfo_(&iam, &nprocs) ; // BLACS rank and world size
  blacs_get_(&zero, &zero, &ictxt ); // -> Create context
  blacs_gridinit_(&ictxt, &layout, &nprow, &npcol ); // Context -> Initialize the grid
  blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol ); // Context -> Context grid info (# procs row/col, current procs row/col)

  int ia=0, ja=0;
  int g_start=0;
  {
    int *ll=l->local_lattice, loc_vol=ll[X]*ll[Y]*ll[Z]*ll[T], *pg=l->num_processes_dir;
    
    g_start += g.my_coords[X] * loc_vol;
    g_start += g.my_coords[Y] * loc_vol * pg[X];
    g_start += g.my_coords[Z] * loc_vol * pg[X]*pg[Y];
    g_start += g.my_coords[T] * loc_vol * pg[X]*pg[Y]*pg[Z];
  }
  // divided by 2 due to oddeven
  ia = (g.odd_even)?(g_start*l->num_lattice_site_var/2):(g_start*l->num_lattice_site_var);
  ia++;
  ja = 1;
  
  // Compute the size of the local matrices
  int mpA = numroc_( &m, &nb, &myrow, &izero, &nprow ); // My proc -> row of local A

  //printf("myrow = %d, g_start = %d (%d), nprow = %d, rank = %d, mpA = %d, mx = %d\n", myrow, g_start, g_start*l->num_lattice_site_var/2, nprow, g.my_rank, mpA, mx);

  // Create descriptor
  int descA[9];
  int info;
  int lddA = mpA > 1 ? mpA : 1;

  descinit_( descA, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &lddA, &info);

  if(info != 0) {
    error0("Error in descinit, info = %d\n", info);
  }

  complex_PRECISION *taux=NULL;
  int lwork=0;
  complex_PRECISION *work=NULL;

  {
    int iroff = (ia-1)%(nb);
    int icoff = (ja-1)%(nb);

    int iarow = indxg2p_(&ia, &nb, &myrow, descA+6, &nprow);
    int iacol = indxg2p_(&ja, &nb, &mycol, descA+7, &npcol);

    int n_lw=nx+icoff, m_lw=mx+iroff;
    int nq0 = numroc_(&n_lw, &nb, &mycol, &iacol, &npcol);
    int mp0 = numroc_(&m_lw, &nb, &myrow, &iarow, &nprow);

    lwork = mp0 + MAX(1,nq0);
  }

  // FIXME : move these malloc/free to a more appropriate place
  // FIXME : switch to lwork different from -1

  MALLOC( taux, complex_PRECISION, nx );
  MALLOC( work, complex_PRECISION, lwork );

  MPI_Barrier(MPI_COMM_WORLD);

  // compute QR in parallel
  lwork=-1;
  pgeqr2_PRECISION( (MKL_INT*) &mx, (MKL_INT*) &nx, (MKL_PRECISION*)a, (MKL_INT*) &ia,
                    (MKL_INT*) &ja, (MKL_INT*) descA, (MKL_PRECISION*)taux, (MKL_PRECISION*)work,
                    (MKL_INT*) &lwork, (MKL_INT*) &info );
  if (info != 0) {
    error0("Error in pgeqrf_PRECISION, info = %d\n", info);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if ( g.my_rank==0 ) {
    // copy/extract R
    memset( R[0], 0, sizeof(complex_PRECISION)*nx*nx );
    // compute R^{-1}
    for ( int j=0; j<nx; j++ ) {
      for ( int i=0; i<(j+1); i++ ) {
        R[j][i] = A[j][i];
      }
    }
    
    MPI_Bcast( R[0], nx*nx, MPI_COMPLEX_PRECISION, g.my_rank, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
  } else {
    MPI_Bcast( R[0], nx*nx, MPI_COMPLEX_PRECISION, 0, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
  }

  //int nr_prnts = 5;
  //printf("R (proc=%d) ---> \n", g.my_rank);
  //for (int i=0; i<nr_prnts; i++) printf("%f+i%f   ", creal(R[0][i]), cimag(R[0][i]));
  //printf("\n");

  MPI_Barrier(MPI_COMM_WORLD);

  // construct Q from QR
  lwork=-1;
  pung2r_PRECISION( (MKL_INT*) &mx, (MKL_INT*) &nx, (MKL_INT*) &nx, (MKL_PRECISION*)a, (MKL_INT*) &ia,
                    (MKL_INT*) &ja, (MKL_INT*) descA, (MKL_PRECISION*)taux, (MKL_PRECISION*)work,
                    (MKL_INT*) &lwork, (MKL_INT*) &info );
  if (info != 0) {
    error0("Error in pungqr_PRECISION, info = %d\n", info);
  }

  FREE( taux, complex_PRECISION, nx );
  FREE( work, complex_PRECISION, lwork );
  if (info != 0) {
    error0("Error in pgeqrf_PRECISION, info = %d\n", info);
  }

  blacs_gridexit_(&ictxt);
  */
}
#endif

// least squares
//void least_sq_PRECISION( eigslvr_PRECISION_struct* eigen_struct )
//{

//  eigen_struct->info = ggev_PRECISION( LAPACK_COL_MAJOR, eigen_struct->jobvl, eigen_struct->jobvr,
//                                       eigen_struct->N, eigen_struct->A, eigen_struct->lda,
//                                       eigen_struct->B, eigen_struct->ldb, eigen_struct->w, eigen_struct->beta,
//                                       eigen_struct->vl, eigen_struct->ldvl, eigen_struct->vr, eigen_struct->ldvr );
//}

#endif
