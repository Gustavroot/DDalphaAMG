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

#ifndef LOCAL_POLYPREC_PRECISION_HEADER
  #define LOCAL_POLYPREC_PRECISION_HEADER


  void local_set_ghost_PRECISION( vector_PRECISION phi, const int mu, const int dir,
                                  comm_PRECISION_struct *c, const int amount, level_struct *l );

  void coarse_local_n_hopping_term_PRECISION( vector_PRECISION out, vector_PRECISION in, operator_PRECISION_struct *op,
                                              const int amount, level_struct *l, struct Thread *threading );

  void coarse_local_hopping_term_PRECISION( vector_PRECISION out, vector_PRECISION in, operator_PRECISION_struct *op,
                                            const int amount, level_struct *l, struct Thread *threading );

  void coarse_local_apply_schur_complement_PRECISION( vector_PRECISION out, vector_PRECISION in,
                                                      operator_PRECISION_struct *op, level_struct *l,
                                                      struct Thread *threading );

  void local_apply_polyprec_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                                       int res, level_struct *l, struct Thread *threading );

  void local_re_construct_lejas_PRECISION( level_struct *l, struct Thread *threading );

  void local_fgmres_PRECISION_struct_init( local_gmres_PRECISION_struct *p );

  void local_fgmres_PRECISION_struct_alloc( int m, int n, long int vl, PRECISION tol, const int type, const int prec_kind,
                                            void (*precond)(), void (*eval_op)(), local_gmres_PRECISION_struct *p, level_struct *l );

  void local_fgmres_PRECISION_struct_free( local_gmres_PRECISION_struct *p, level_struct *l );

  int local_fgmres_PRECISION( local_gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );

  int local_arnoldi_step_PRECISION( vector_PRECISION *V, vector_PRECISION *Z, vector_PRECISION w,
                                    complex_PRECISION **H, complex_PRECISION* buffer, int j, void (*prec)(),
                                    local_gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );

  void local_process_multi_inner_product_PRECISION( int count, complex_PRECISION *results, vector_PRECISION *phi, vector_PRECISION psi,
                                                    int start, int end, level_struct *l, struct Thread *threading );

  void local_qr_update_PRECISION( complex_PRECISION **H, complex_PRECISION *s,
                                  complex_PRECISION *c, complex_PRECISION *gamma, int j,
                                  level_struct *l, struct Thread *threading );

  void coarse_local_diag_oo_inv_PRECISION( vector_PRECISION y, vector_PRECISION x, operator_PRECISION_struct *op, 
                                           level_struct *l, struct Thread *threading );

  void coarse_local_diag_ee_PRECISION( vector_PRECISION y, vector_PRECISION x, operator_PRECISION_struct *op, level_struct *l, struct Thread *threading );

  void local_harmonic_ritz_PRECISION( local_gmres_PRECISION_struct *p );
  void local_leja_ordering_PRECISION( local_gmres_PRECISION_struct *p );
  void local_update_lejas_PRECISION( local_gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );
  void local_apply_polyprec_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                                       int res, level_struct *l, struct Thread *threading );

#endif
