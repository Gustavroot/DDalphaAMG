#ifndef UTIL_MACROS_H
#define UTIL_MACROS_H

#define IMPLIES( A, B ) !( A ) || ( B )
#define XOR( A, B ) (( A ) && !( B )) || (!( A ) && ( B ))
#define NAND( A, B ) !( (A) && (B) )
#define DIVIDES( A, B ) A == 0 || ((double)(B)/(double)(A) - (double)((int)(B)/(int)(A))) == 0 
#define ASCENDING( A, B, C ) ( (A)<=(B) ) && ( (B)<=(C) )
#define MAX( A, B ) ( (A > B) ? A : B )
#define MIN( A, B ) ( (A < B) ? A : B )

#endif // UTIL_MACROS_H
