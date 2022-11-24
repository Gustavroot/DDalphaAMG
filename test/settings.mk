# Settings to be set during compiling to adjust
# the compilation process to the compiling system.

CRITERION_INCLUDE	:= $(or $(CRITERION_INCLUDE),/usr/include)
CRITERION_LIB		:= $(or $(CRITERION_LIB),/usr/lib)