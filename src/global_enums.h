#ifndef CONSTANTS_H
#define CONSTANTS_H

// enumerations
enum { T, Z, Y, X };
#ifdef __cplusplus
// Scoped version of the previous enum.
enum class LatticeAxis {T=T, Z=Z, Y=Y, X=X};
#endif
enum
{
    _EVEN,
    _ODD
};
enum
{
    _NO_DEFAULT_SET,
    _DEFAULT_SET
};
enum
{
    _NO_REORDERING,
    _REORDER
};
enum
{
    _ADD,
    _COPY
};
enum
{
    _ORDINARY,
    _SCHWARZ
};
enum
{
    _RES,
    _NO_RES
};
enum
{
    _STANDARD,
    _LIME,
    _MULTI
}; // formats
enum
{
    _READ,
    _WRITE
};
enum
{
    _NO_SHIFT
};
enum
{
    _BTWN_ORTH = 20
};
enum
{
    _GLOBAL_FGMRES,
    _K_CYCLE,
    _COARSE_GMRES,
    _SMOOTHER
};
enum
{
    _COARSE_GLOBAL
};
enum
{
    _FULL_SYSTEM,
    _EVEN_SITES,
    _ODD_SITES
};
enum
{
    _LEFT,
    _RIGHT,
    _NOTHING
};
enum
{
    _GIP,
    _PIP,
    _LA2,
    _LA6,
    _LA8,
    _LA,
    _CPY,
    _SET,
    _PR,
    _SC,
    _NC,
    _SM,
    _OP_COMM,
    _OP_IDLE,
    _ALLR,
    _GD_COMM,
    _GD_IDLE,
    _GRAM_SCHMIDT,
    _GRAM_SCHMIDT_ON_AGGREGATES,
    _SM1,
    _SM2,
    _SM3,
    _SM4,
    _SMALL1,
    _SMALL2,
    _HOPPING,
    _NHOPPING,
    _SOlV_NC,
    _SM_OE,
    _NUM_PROF
}; // _NUM_PROF has always to be the last constant!
enum
{
    _VTS = 20
};
enum
{
    _TRCKD_VAL,
    _STP_TIME,
    _SLV_ITER,
    _SLV_TIME,
    _CRS_ITER,
    _CRS_TIME,
    _SLV_ERR,
    _CGNR_ERR,
    _NUM_OPTB
};

#ifdef CUDA_OPT
enum
{
    _H2D,
    _D2H,
    _D2D
};
enum
{
    _CUDA_ASYNC,
    _CUDA_SYNC
};
enum
{
    _SOFT_CHECK,
    _HARD_CHECK
};
#endif
#endif // CONSTANTS_H
