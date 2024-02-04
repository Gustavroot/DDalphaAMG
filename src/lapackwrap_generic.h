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

#ifndef LAPACKWRAP_PRECISION_HEADER
  #define LAPACKWRAP_PRECISION_HEADER

  // eigensolvers
  void gen_eigslvr_PRECISION(eigslvr_PRECISION_struct* eigen_struct);
  void eigslvr_PRECISION(eigslvr_PRECISION_struct* eigen_struct);
  void dirctslvr_PRECISION(dirctslvr_PRECISION_struct* dirctslvr);

  // QR functions
  void qr_PRECISION(eigslvr_PRECISION_struct* eigen_struct);
  void q_from_qr_PRECISION(eigslvr_PRECISION_struct* eigen_struct);

  // explicit inversion of triangular matrix
  void inv_tri_PRECISION(eigslvr_PRECISION_struct* eigen_struct);

  // parallel QR
  void pqr_PRECISION( int mx, int nx, complex_PRECISION **Ax, complex_PRECISION **R, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );

  // LSP
  void gels_via_givens_PRECISION( int ida, int idb, complex_PRECISION* a, int lda, complex_PRECISION* b, int ldb, complex_PRECISION* c, int k, int m, gmres_PRECISION_struct *p );

#endif
