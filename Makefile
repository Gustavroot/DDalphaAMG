# --- COMPILER ----------------------------------------

# if using -std different than gnu11, some changes are needed
CC = mpicc -std=gnu11 -Wall -pedantic
#MPI_INCLUDE = /home/ramirez/installs/openmpi/dir/include/
#MPI_LIB = /home/ramirez/installs/openmpi/dir/lib64/

CPP = cpp
MAKEDEP = $(CPP) -MM

# if using -std different than c++11, some changes are needed
NVCC = nvcc
CUDA_INCLUDE = /usr/local/cuda/include/
CUDA_LIB = /usr/local/cuda/lib64/

# --- DO NOT CHANGE -----------------------------------
SRCDIR = src
SRCDIR_CUDA = src/gpu
SRC = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.c,$(wildcard $(SRCDIR)/*.c)))
SRC += $(patsubst $(SRCDIR_CUDA)/%,%,$(filter-out %_generic.c,$(wildcard $(SRCDIR_CUDA)/*.c)))
SRC_CUDA = $(patsubst $(SRCDIR_CUDA)/%,%,$(filter-out %_generic.cu,$(wildcard $(SRCDIR_CUDA)/*.cu)))
BUILDDIR = build
GSRCDIR = $(BUILDDIR)/gsrc
SRCGEN = $(patsubst $(SRCDIR)/%,%,$(wildcard $(SRCDIR)/*_generic.c))
SRCGEN += $(patsubst $(SRCDIR_CUDA)/%,%,$(wildcard $(SRCDIR_CUDA)/*_generic.c))
SRCGEN_CUDA = $(patsubst $(SRCDIR_CUDA)/%,%,$(wildcard $(SRCDIR_CUDA)/*_generic.cu))
GSRCFLT = $(patsubst %_generic.c,$(GSRCDIR)/%_float.c,$(SRCGEN))
GSRCFLT_CUDA = $(patsubst %_generic.cu,$(GSRCDIR)/%_float.cu,$(SRCGEN_CUDA))
GSRCDBL = $(patsubst %_generic.c,$(GSRCDIR)/%_double.c,$(SRCGEN))
GSRCDBL_CUDA = $(patsubst %_generic.cu,$(GSRCDIR)/%_double.cu,$(SRCGEN_CUDA))
GSRC = $(patsubst %,$(GSRCDIR)/%,$(SRC)) $(GSRCFLT) $(GSRCDBL)
GSRC_CUDA = $(patsubst %,$(GSRCDIR)/%,$(SRC_CUDA)) $(GSRCFLT_CUDA) $(GSRCDBL_CUDA)
HEA = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.h,$(wildcard $(SRCDIR)/*.h)))
HEA_CUDA = $(patsubst $(SRCDIR_CUDA)/%,%,$(filter-out %_generic.h,$(wildcard $(SRCDIR_CUDA)/*.h)))
HEAGEN = $(patsubst $(SRCDIR)/%,%,$(wildcard $(SRCDIR)/*_generic.h))
HEAGEN_CUDA = $(patsubst $(SRCDIR_CUDA)/%,%,$(wildcard $(SRCDIR_CUDA)/*_generic.h))
GHEAFLT = $(patsubst %_generic.h,$(GSRCDIR)/%_float.h,$(HEAGEN))
GHEAFLT_CUDA = $(patsubst %_generic.h,$(GSRCDIR)/%_float.h,$(HEAGEN_CUDA))
GHEADBL = $(patsubst %_generic.h,$(GSRCDIR)/%_double.h,$(HEAGEN))
GHEADBL_CUDA = $(patsubst %_generic.h,$(GSRCDIR)/%_double.h,$(HEAGEN_CUDA))
GHEA = $(patsubst %,$(GSRCDIR)/%,$(HEA))
GHEA += $(patsubst %,$(GSRCDIR)/%,$(HEA_CUDA))
GHEA += $(GHEAFLT) $(GHEAFLT_CUDA) $(GHEADBL) $(GHEADBL_CUDA)
OBJ = $(patsubst $(GSRCDIR)/%.c,$(BUILDDIR)/%.o,$(GSRC))
OBJDB = $(patsubst %.o,%_db.o,$(OBJ))
OBJ_CUDA = $(patsubst $(GSRCDIR)/%.cu,$(BUILDDIR)/%.o,$(GSRC_CUDA))
OBJ_CUDADB = $(patsubst %.o,%_db.o,$(OBJ_CUDA))
DEP = $(patsubst %.c,%.dep,$(GSRC)) $(patsubst %.cu,%.dep,$(GSRC_CUDA))

# --- FLAGS -------------------------------------------
CUDA_ENABLER = -DCUDA_OPT
COMMON_FLAGS = -DCUDA_ERROR_CHECK -DPROFILING $(CUDA_ENABLER)
#COMMON_FLAGS = -DPROFILING

OPT_FLAGS = -fopenmp -DOPENMP -DSSE -msse4.2 -I$(CUDA_INCLUDE)
CFLAGS = -DPARAMOUTPUT -DTRACK_RES -DFGMRES_RESTEST $(COMMON_FLAGS)
# -DSINGLE_ALLREDUCE_ARNOLDI
# -DCOARSE_RES -DSCHWARZ_RES -DTESTVECTOR_ANALYSIS
OPT_VERSION_FLAGS =$(OPT_FLAGS) -O3 -ffast-math
DEBUG_VERSION_FLAGS = $(OPT_FLAGS)

OPT_FLAGS_CUDA = 
CFLAGS_CUDA = $(COMMON_FLAGS)
OPT_VERSION_FLAGS_CUDA = $(OPT_FLAGS_CUDA) -O3 # what about --ffast-math ?
DEBUG_VERSION_FLAGS_CUDA = $(OPT_FLAGS_CUDA)

# --- FLAGS FOR CUDA ---------------------------------
NVCC_EXTRA_COMP_FLAGS = -I$(MPI_INCLUDE) -L$(MPI_LIB)
NVCC_EXTRA_COMP_FLAGS += -lmpi
NVCC_EXTRA_COMP_FLAGS += -arch=sm_70 -rdc=true -lcudadevrt
NVCC_EXTRA_COMP_FLAGS += -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
NVCC_EXTRA_COMP_FLAGS += -lcudart -L$(CUDA_LIB)

# --- FLAGS FOR HDF5 ---------------------------------
# H5HEADERS=-DHAVE_HDF5 /usr/include
# H5LIB=-lhdf5 -lz

# --- FLAGS FOR LIME ---------------------------------
# LIMEH=-DHAVE_LIME -I$(LIMEDIR)/include
# LIMELIB= -L$(LIMEDIR)/lib -llime

all: wilson library documentation
wilson: dd_alpha_amg dd_alpha_amg_db
library: lib/libdd_alpha_amg.a include/dd_alpha_amg_parameters.h include/dd_alpha_amg.h
documentation: doc/user_doc.pdf

.PHONY: all wilson library
.SUFFIXES:
.SECONDARY:

dd_alpha_amg : $(OBJ) $(OBJ_CUDA)
ifeq ($(CUDA_ENABLER),-DCUDA_OPT)
	$(NVCC) --compiler-options='$(OPT_VERSION_FLAGS)' $(NVCC_EXTRA_COMP_FLAGS) $(LIMEH) -o $@ $(OBJ) $(OBJ_CUDA) $(H5LIB) $(LIMELIB) -lm
else
	$(CC) $(OPT_VERSION_FLAGS) $(LIMEH) -o $@ $(OBJ) $(H5LIB) $(LIMELIB) -lm
endif
dd_alpha_amg_db : $(OBJDB) $(OBJ_CUDADB)
ifeq ($(CUDA_ENABLER),-DCUDA_OPT)
	$(NVCC) -g --compiler-options='$(DEBUG_VERSION_FLAGS)' $(NVCC_EXTRA_COMP_FLAGS) $(LIMEH) -o $@ $(OBJDB) $(OBJ_CUDADB) $(H5LIB) $(LIMELIB) -lm
else
	$(CC) -g $(DEBUG_VERSION_FLAGS) $(LIMEH) -o $@ $(OBJDB) $(H5LIB) $(LIMELIB) -lm
endif

lib/libdd_alpha_amg.a: $(OBJ)
	ar rc $@ $(OBJ)
	ar d $@ main.o
	ranlib $@

doc/user_doc.pdf: doc/user_doc.tex doc/user_doc.bib
	( cd doc; pdflatex user_doc; bibtex user_doc; pdflatex user_doc; pdflatex user_doc; )

include/dd_alpha_amg.h: src/dd_alpha_amg.h
	cp src/dd_alpha_amg.h $@

include/dd_alpha_amg_parameters.h: src/dd_alpha_amg_parameters.h
	cp src/dd_alpha_amg_parameters.h $@

$(BUILDDIR)/%.o: $(GSRCDIR)/%.c $(SRCDIR)/*.h $(SRCDIR_CUDA)/*.h
	$(CC) $(CFLAGS) $(OPT_VERSION_FLAGS) $(H5HEADERS) $(LIMEH) -c $< -o $@ -lm

$(BUILDDIR)/%_db.o: $(GSRCDIR)/%.c $(SRCDIR)/*.h $(SRCDIR_CUDA)/*.h
	$(CC) -g $(CFLAGS) $(DEBUG_VERSION_FLAGS) $(H5HEADERS) $(LIMEH) -DDEBUG -c $< -o $@ -lm

ifeq ($(CUDA_ENABLER),-DCUDA_OPT)
$(BUILDDIR)/%.o: $(GSRCDIR)/%.cu $(SRCDIR)/*.h $(SRCDIR_CUDA)/*.h
	$(NVCC) $(CFLAGS_CUDA) $(OPT_VERSION_FLAGS_CUDA) $(NVCC_EXTRA_COMP_FLAGS) -dc -L$(CUDA_LIB) -c $< -o $@ -lm
endif

ifeq ($(CUDA_ENABLER),-DCUDA_OPT)
$(BUILDDIR)/%_db.o: $(GSRCDIR)/%.cu $(SRCDIR)/*.h $(SRCDIR_CUDA)/*.h
	$(NVCC) -g $(CFLAGS_CUDA) $(DEBUG_VERSION_FLAGS_CUDA) $(NVCC_EXTRA_COMP_FLAGS) -dc -L$(CUDA_LIB) -DDEBUG -c $< -o $@ -lm
endif

$(GSRCDIR)/%.h: $(SRCDIR)/%.h $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%.h: $(SRCDIR_CUDA)/%.h $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%_float.h: $(SRCDIR)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_float.h: $(SRCDIR_CUDA)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_double.h: $(SRCDIR)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%_double.h: $(SRCDIR_CUDA)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%.cu: $(SRCDIR_CUDA)/%.cu $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%.c: $(SRCDIR)/%.c $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%.c: $(SRCDIR_CUDA)/%.c $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%_float.cu: $(SRCDIR_CUDA)/%_generic.cu $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_float.c: $(SRCDIR)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_float.c: $(SRCDIR_CUDA)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_double.cu: $(SRCDIR_CUDA)/%_generic.cu $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%_double.c: $(SRCDIR)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%_double.c: $(SRCDIR_CUDA)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

%.dep: %.c $(GHEA)
	$(MAKEDEP) $< | sed 's,\(.*\)\.o[ :]*,$(BUILDDIR)/\1.o $@ : ,g' > $@
	$(MAKEDEP) $< | sed 's,\(.*\)\.o[ :]*,$(BUILDDIR)/\1_db.o $@ : ,g' >> $@

%.dep: %.cu $(GHEA)
	$(MAKEDEP) $< | sed 's,\(.*\)\.o[ :]*,$(BUILDDIR)/\1.o $@ : ,g' > $@
	$(MAKEDEP) $< | sed 's,\(.*\)\.o[ :]*,$(BUILDDIR)/\1_db.o $@ : ,g' >> $@

clean:
	rm -f $(BUILDDIR)/*.o
	rm -f $(GSRCDIR)/*
	rm -f dd_alpha_amg
	rm -f dd_alpha_amg_db

-include $(DEP)
