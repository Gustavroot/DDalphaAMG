# --- USER CHANGES SHALL BE MADE IN MakeSettings.mk ---
-include MakeSettings.mk

# --- DO NOT CHANGE -----------------------------------
SRCDIR = src
SRC = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.c,$(shell find $(SRCDIR) -name '*.c')))
SRC_CUDA = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.cu,$(shell find $(SRCDIR) -name '*.cu')))
BUILDDIR = build
GSRCDIR = $(BUILDDIR)/gsrc
SRCGEN = $(patsubst $(SRCDIR)/%,%,$(shell find $(SRCDIR) -name '*_generic.c'))
SRCGEN_CUDA = $(patsubst $(SRCDIR)/%,%,$(shell find $(SRCDIR) -name '*_generic.cu'))
GSRCFLT = $(patsubst %_generic.c,$(GSRCDIR)/%_float.c,$(SRCGEN))
GSRCFLT_CUDA = $(patsubst %_generic.cu,$(GSRCDIR)/%_float.cu,$(SRCGEN_CUDA))
GSRCDBL = $(patsubst %_generic.c,$(GSRCDIR)/%_double.c,$(SRCGEN))
GSRCDBL_CUDA = $(patsubst %_generic.cu,$(GSRCDIR)/%_double.cu,$(SRCGEN_CUDA))
GSRC = $(patsubst %,$(GSRCDIR)/%,$(SRC)) $(GSRCFLT) $(GSRCDBL)
GSRC_CUDA = $(patsubst %,$(GSRCDIR)/%,$(SRC_CUDA)) $(GSRCFLT_CUDA) $(GSRCDBL_CUDA)
HEA = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.h,$(shell find $(SRCDIR) -name '*.h')))
HEAGEN = $(patsubst $(SRCDIR)/%,%,$(shell find $(SRCDIR) -name '*_generic.h'))
GHEAFLT = $(patsubst %_generic.h,$(GSRCDIR)/%_float.h,$(HEAGEN))
GHEADBL = $(patsubst %_generic.h,$(GSRCDIR)/%_double.h,$(HEAGEN))
GHEA = $(patsubst %,$(GSRCDIR)/%,$(HEA))
GHEA += $(GHEAFLT) $(GHEADBL)
OBJ = $(patsubst $(GSRCDIR)/%.c,$(BUILDDIR)/%.o,$(GSRC))
OBJ_NO_MAIN = $(filter-out %/main.o,$(OBJ))
OBJDB = $(patsubst %.o,%_db.o,$(OBJ))
OBJ_NO_MAINDB = $(filter-out %/main_db.o,$(OBJDB))
OBJ_CUDA = $(patsubst $(GSRCDIR)/%.cu,$(BUILDDIR)/%.o,$(GSRC_CUDA))
OBJ_CUDADB = $(patsubst %.o,%_db.o,$(OBJ_CUDA))
OBJ_CUDA_DLINK = $(BUILDDIR)/dd_alpha_amg.dlink.o
OBJ_CUDA_DLINKDB = $(BUILDDIR)/dd_alpha_amg_db.dlink.o

# --- FLAGS -------------------------------------------

## Includes
COMMON_COMPILE_FLAGS = -I$(GSRCDIR)
# COMMON_COMPILE_FLAGS = -DPROFILING
ifdef MPI_INCLUDE
	COMMON_COMPILE_FLAGS += -I$(MPI_INCLUDE)
endif
ifdef CUDA_INCLUDE
	COMMON_COMPILE_FLAGS += -I$(CUDA_INCLUDE)
endif

# with SSE on, this flag forces (CPU) GMRES as a smoother to run without vectorization
#COMMON_COMPILE_FLAGS += -DGMRES_ON_GPUS

# GCR as a smoother
#COMMON_COMPILE_FLAGS += -DGCR_SMOOTHER
# Richardson as a smoother
#COMMON_COMPILE_FLAGS += -DRICHARDSON_SMOOTHER

# include twisted mass term at the coarsest level
COMMON_COMPILE_FLAGS += -DTM_COARSEST

# LAPACK is needed for coarsest-level improvements
#LAPACK_DIR = dependencies/lapack-3.9.0
#LAPACKE_DIR = $(LAPACK_DIR)/LAPACKE
#LAPACKE_INCLUDE = $(LAPACKE_DIR)/include
#BLASLIB      = $(LAPACK_DIR)/librefblas.a
#LAPACKLIB    = $(LAPACK_DIR)/liblapack.a
#LAPACKELIB   = $(LAPACK_DIR)/liblapacke.a
#LAPACK_LIBRARIES = $(LAPACKELIB) $(LAPACKLIB) $(BLASLIB)
#COMMON_COMPILE_FLAGS += -I$(LAPACKE_INCLUDE)

# coarsest-level improvements
#COMMON_COMPILE_FLAGS += -DGCRODR
#COMMON_COMPILE_FLAGS += -DPOLYPREC

## Defines
COMMON_COMPILE_FLAGS += -DCUDA_ERROR_CHECK -DPROFILING $(NVTX_DISABLE) #-DGPU2GPU_COMMS_VIA_CPUS
ifeq ($(SSE_ENABLER),yes)
	COMMON_COMPILE_FLAGS += -DSSE
ifeq ($(AVX512_ENABLER),yes)
	COMMON_COMPILE_FLAGS += -DAVX512
else 
ifeq ($(AVX_ENABLER),yes)
	COMMON_COMPILE_FLAGS += -DAVX2 -DAVX
endif
endif
endif

ifeq ($(CUDA_ENABLER),yes)
	COMMON_COMPILE_FLAGS += -DCUDA_OPT
endif

COMPILE_FLAGS = $(COMMON_COMPILE_FLAGS) -DPARAMOUTPUT -DTRACK_RES -DFGMRES_RESTEST

ifeq ($(CUDA_ENABLER),yes)
COMPILE_FLAGS += -fopenmp -DOPENMP
else
COMPILE_FLAGS += -fopenmp -DOPENMP
endif

ifeq ($(SSE_ENABLER),yes)
	COMPILE_FLAGS += -msse4.2 -msse
ifeq ($(AVX512_ENABLER),yes)
	COMPILE_FLAGS += -mavx512vl -mavx512f -mfma
else
ifeq ($(AVX_ENABLER),yes)
	COMPILE_FLAGS += -mavx -mavx2 -mfma
endif
endif
endif

# Extra Warnings that developers should fix but don't.
# This is a C only flag as implicit function declaration is forbidden in C++ anyways.
COMPILE_FLAGS += -Wall -Werror-implicit-function-declaration
LINK_FLAGS = -lgomp -lm -ldl

#LINK_FLAGS += -DHALF_PREC_STORAGE

# -DSINGLE_ALLREDUCE_ARNOLDI
# -DCOARSE_RES -DSCHWARZ_RES -DTESTVECTOR_ANALYSIS
OPT_VERSION_FLAGS = -O3 -ffast-math
DEBUG_VERSION_FLAGS = 

## CUDA-only flags
NVCC_ARCHITECTURE_FLAGS = -arch=$(CUDA_ARCH)
ifdef CUDA_CODE
	NVCC_ARCHITECTURE_FLAGS += -code=$(CUDA_CODE)
endif

COMPILE_FLAGS_CUDA = $(NVCC_ARCHITECTURE_FLAGS) -rdc=true $(COMMON_COMPILE_FLAGS)

ifeq ($(CUDA_ENABLER),yes)
COMPILE_FLAGS_CUDA += -DOPENMP -Xcompiler "-fopenmp -Wall"
else
COMPILE_FLAGS_CUDA += -DOPENMP -Xcompiler "-fopenmp -Wall"
endif

ifeq ($(SSE_ENABLER),yes)
	COMPILE_FLAGS_CUDA += -Xcompiler "-msse4.2"
endif
OPT_VERSION_FLAGS_CUDA = -O3 -Xcompiler "-ffast-math"
DEBUG_VERSION_FLAGS_CUDA = 

NVCC_LINK_FLAGS = $(NVCC_ARCHITECTURE_FLAGS) -lmpi -lgomp -lm
ifdef MPI_LIB
	NVCC_LINK_FLAGS += -L$(MPI_LIB)
endif

#-include test/gtest/Makefile

all: wilson library library_db #documentation gtest

.PHONY: all wilson library library_db #documentation gtest
.SUFFIXES:
.SECONDARY:


# Linking of dd_alpha_amg application
wilson: dd_alpha_amg dd_alpha_amg_db

ifeq ($(CUDA_ENABLER),yes)
dd_alpha_amg : $(OBJ) $(OBJ_CUDA)
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $(OBJ) $(OBJ_CUDA) $(LAPACK_LIBRARIES) -lgfortran
else
dd_alpha_amg : $(OBJ)
#	$(CC) -o $@ $(OBJ) $(LINK_FLAGS)
	$(CC) $(LINK_FLAGS) -o $@ $(OBJ) -lmpi -lgomp -lm $(LAPACK_LIBRARIES) -lgfortran
endif

ifeq ($(CUDA_ENABLER),yes)
dd_alpha_amg_db : $(OBJDB) $(OBJ_CUDADB)
	$(NVCC) -g $(NVCC_LINK_FLAGS) -o $@ $(OBJDB) $(OBJ_CUDADB) $(LAPACK_LIBRARIES) -lgfortran
else
dd_alpha_amg_db : $(OBJDB)
#	$(CC) -g -o $@ $(OBJDB) $(LINK_FLAGS)
	$(CC) -g $(LINK_FLAGS) -o $@ $(OBJDB) -lmpi -lgomp -lm $(LAPACK_LIBRARIES) -lgfortran
endif

######
#TODO: FIX LIBRARY CREATION WITH CUDA ENABLED
######
# Linking of dd_alpha_amg library
library: lib/libdd_alpha_amg.a include/dd_alpha_amg_parameters.h include/dd_alpha_amg.h
library_db: lib/libdd_alpha_amg_db.a include/dd_alpha_amg_parameters.h include/dd_alpha_amg.h

ifeq ($(CUDA_ENABLER),yes)
$(OBJ_CUDA_DLINK): $(OBJ_NO_MAIN) $(OBJ_CUDA)
# to actually use the objects created by NVCC we need an object file
# on which NVCC did perform device code linking
# see also: https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
	$(NVCC) $(NVCC_LINK_FLAGS) -dlink -lcudadevrt -o $@ $(OBJ_NO_MAIN) $(OBJ_CUDA)

lib/libdd_alpha_amg.a: $(OBJ_CUDA_DLINK) $(OBJ_NO_MAIN) $(OBJ_CUDA)
# see also https://stackoverflow.com/questions/26893588/creating-a-static-cuda-library-to-be-linked-with-a-c-program
	ar rc $@ $(OBJ_CUDA_DLINK) $(OBJ_NO_MAIN) $(OBJ_CUDA)
	ranlib $@

$(OBJ_CUDA_DLINKDB): $(OBJ_NO_MAINDB) $(OBJ_CUDADB)
# to actually use the objects created by NVCC we need an object file
# on which NVCC did perform device code linking
# see also: https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
	$(NVCC) $(NVCC_LINK_FLAGS) -dlink -lcudadevrt -o $@ $(OBJ_NO_MAINDB) $(OBJ_CUDADB)

lib/libdd_alpha_amg_db.a: $(OBJ_CUDA_DLINKDB) $(OBJ_NO_MAINDB) $(OBJ_CUDADB)
# see also https://stackoverflow.com/questions/26893588/creating-a-static-cuda-library-to-be-linked-with-a-c-program
	ar rc $@ $(OBJ_CUDA_DLINKDB) $(OBJ_NO_MAINDB) $(OBJ_CUDADB)
	ranlib $@

else
lib/libdd_alpha_amg.a: $(OBJ_NO_MAIN)
	ar rc $@ $(OBJ_NO_MAIN)
	ranlib $@

lib/libdd_alpha_amg_db.a: $(OBJ_NO_MAINDB)
	ar rc $@ $(OBJ_NO_MAINDB)
	ranlib $@
endif

# Header files for library
include/dd_alpha_amg.h: src/dd_alpha_amg.h
	cp src/dd_alpha_amg.h $@

include/dd_alpha_amg_parameters.h: src/dd_alpha_amg_parameters.h
	cp src/dd_alpha_amg_parameters.h $@

# Documentation
documentation: doc/user_doc.pdf doc/doxygen

doc/user_doc.pdf: doc/user_doc.tex doc/user_doc.bib
	( cd doc; pdflatex user_doc; bibtex user_doc; pdflatex user_doc; pdflatex user_doc; )

doc/doxygen: src/* src/gpu/* doxygen.conf
	doxygen doxygen.conf

# Object compilation (host)
$(BUILDDIR)/%.o: $(GSRCDIR)/%.c $(GHEA)
	@mkdir -p $(@D)
	$(CC) $(COMPILE_FLAGS) $(OPT_VERSION_FLAGS) -c $< -o $@ $(LINK_FLAGS)

$(BUILDDIR)/%_db.o: $(GSRCDIR)/%.c $(GHEA)
	@mkdir -p $(@D)
	$(CC) -g $(COMPILE_FLAGS) $(DEBUG_VERSION_FLAGS) -DDEBUG -c $< -o $@

# Object compilation (CUDA)
ifeq ($(CUDA_ENABLER),yes)
$(BUILDDIR)/%.o: $(GSRCDIR)/%.cu $(GHEA)
	$(NVCC) $(COMPILE_FLAGS_CUDA) $(OPT_VERSION_FLAGS_CUDA) -dc -c $< -o $@
endif

ifeq ($(CUDA_ENABLER),yes)
$(BUILDDIR)/%_db.o: $(GSRCDIR)/%.cu $(GHEA)
	$(NVCC) -g $(COMPILE_FLAGS_CUDA) $(DEBUG_VERSION_FLAGS_CUDA) -dc -DDEBUG -c $< -o $@
endif


# Source file copies ./gsrc
$(GSRCDIR)/%.h: $(SRCDIR)/%.h $(firstword $(MAKEFILE_LIST))
	@mkdir -p $(@D)
	cp $< $@

$(GSRCDIR)/%_float.h: $(SRCDIR)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_double.h: $(SRCDIR)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%.cu: $(SRCDIR)/%.cu $(firstword $(MAKEFILE_LIST))
	@mkdir -p $(@D)
	cp $< $@

$(GSRCDIR)/%.c: $(SRCDIR)/%.c $(firstword $(MAKEFILE_LIST))
	@mkdir -p $(@D)
	cp $< $@

$(GSRCDIR)/%_float.cu: $(SRCDIR)/%_generic.cu $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_float.c: $(SRCDIR)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_double.cu: $(SRCDIR)/%_generic.cu $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%_double.c: $(SRCDIR)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

# Cleaning
clean:
	rm -rf $(BUILDDIR)
	rm -f dd_alpha_amg
	rm -f dd_alpha_amg_db
	rm -f lib/*
	rm -f doc/user_doc.pdf
	rm -rf doc/doxygen
