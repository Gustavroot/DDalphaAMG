
> ## NOTICE:
> - **The latest news put on the top of this file, and the older follows it**
> - **More plans will updates in [AVX.README.md](AVX.README.md)** 

---
---

### Scalar，SSE-only and SSE+AVX2 tests
- output see [./AVX.RUNS/](./AVX.RUNS)


---
### AVX control --- see: [src/vectorization_control.h](src/vectorization_control.h)
- `SIMD_LENGTH_PRECISION` is different between SSE and AVX. So `SIMD_LENGTH_PRECISION` is defined only when SSE allowed; `AVX_LENGTH_PRECISON` is deined when AVX enabled at the same time. That's to avoid ambiguity and conflict between SSE and AVX.
- Now AVX2 requires SSE support, please set **SSE_ENABLER = yes**.  
- Now to focus on  AVX_COARSE_HOPING_OPERATOR;
  AVX_COARSE_SELF_OPERATOR also finished. You can turn it off/on in [src/vectorization_control.h](src/vectorization_control.h)
- ~~`AVX_COARSE_OPERATOR_PRECISION`is~~ defined as the **GLOBAL AVX MACRO SWITCH** to control COARSE Operator, and **it shuold not be enabled and take effect by now.**
```C
#if defined(AVX2)
#if !defined(SSE)
#error(SSE Not Defined! Now AVX2 requires SSE support, please set SSE_ENABLER = yes.)
#endif

#define AVX_LENGTH_float  8
#define AVX_LENGTH_double 4

// #define AVX_COARSE_OPERATOR_float // to control all COASE_OPERATOR support, but not now.
// #define AVX_COARSE_SELF_OPERATOR_float     // Self-coupling
#define AVX_COARSE_HOPPING_OPERATOR_float  // Hopping term

#define AVX_BLAS_float
```