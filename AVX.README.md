[toc]

## I. coarse_apply_schur_complement_PRECISION  (coarse_oddeven_generic.c)   

### 1. **`[DONE]`** coarse_diag_ee_PRECISION( out, in, op, l, threading ); 
#### 1.1. **`[DONE]`** vectorized_coarse_self_couplings_PRECISION(y, x, op->clover_vectorized, start, end, l );
##### 1.1.1 **`[DONE]`** vectorized_cgemv   
defined in  *vectorized_blas.h / vectorized_blas_avx.h*

### 2. **`[no vectorized]`** vector_PRECISION_define( tmp[0], 0, start, end, l );
  
### 3. **`[no vectorized]`** coarse_hopping_term_PRECISION( tmp[0], in, op, _ODD_SITES, l, threading );
#### 3.1. **`[no vectorized]`** coarse_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l ); 
     
### 4. **`[DONE]`** coarse_diag_oo_inv_PRECISION( tmp[1], tmp[0], op, l, threading ); 
#### 4.1. **`[DONE]`** vectorized_coarse_self_couplings_PRECISION()
```c
#if !defined(VECTORIZE_COARSE_OPERATOR_PRECISION) && !defined(AVX_COARSE_OPERATOR_PRECISION)
...
#if defined(AVX2)
   vectorized_coarse_self_couplings_PRECISION(y, x, op->clover_vectorized, start, end, l );
#elif defined(SSE)
    coarse_self_couplings_PRECISION_vectorized( y, x, op->clover_vectorized, start, end, l );
#else
    error0("defined(VECTORIZED_PRECISION), but neither defined(AVX2) nor defined(SSE)");
#endif
```

### 5.**`[DONE]`** coarse_n_hopping_term_PRECISION( out, tmp[1], op, _EVEN_SITES, l, threading );
```C++
void (*coarse_hopp)(vector_PRECISION eta, vector_PRECISION phi, OPERATOR_TYPE_PRECISION *D, level_struct *l);
  if(sign == +1)
    coarse_hopp = coarse_hopp_PRECISION_vectorized;
  else
    coarse_hopp = coarse_n_hopp_PRECISION_vectorized;
```
