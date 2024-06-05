#ifndef LINALG_PROXY_PRECISION_H
#define LINALG_PROXY_PRECISION_H


complex_PRECISION global_inner_product_PRECISION( vector_PRECISION* V, vector_PRECISION psi, complex_PRECISION *result,
                  int n, int start, int end, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );

#endif  // LINALG_PROXY_PRECISION_H
