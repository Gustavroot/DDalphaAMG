# --- COMPILER ----------------------------------------
CC = mpicc -std=gnu11 -Wall -pedantic
CPP = cpp
MAKEDEP = $(CPP) -MM
NVCC=/usr/local/cuda/bin/nvcc

# --- DO NOT CHANGE -----------------------------------
SRCDIR = src
BUILDDIR = build
GSRCDIR = $(BUILDDIR)/gsrc
SRC = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.c,$(wildcard $(SRCDIR)/*.c)))
SRC_CUDA = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.cu,$(wildcard $(SRCDIR)/*.cu)))
SRCGEN = $(patsubst $(SRCDIR)/%,%,$(wildcard $(SRCDIR)/*_generic.c))
SRCGEN_CUDA = $(patsubst $(SRCDIR)/%,%,$(wildcard $(SRCDIR)/*_generic.cu))
GSRCFLT = $(patsubst %_generic.c,$(GSRCDIR)/%_float.c,$(SRCGEN))
GSRCFLT_CUDA = $(patsubst %_generic.cu,$(GSRCDIR)/%_float.cu,$(SRCGEN_CUDA))
GSRCDBL = $(patsubst %_generic.c,$(GSRCDIR)/%_double.c,$(SRCGEN))
GSRCDBL_CUDA = $(patsubst %_generic.cu,$(GSRCDIR)/%_double.cu,$(SRCGEN_CUDA))
GSRC = $(patsubst %,$(GSRCDIR)/%,$(SRC)) $(GSRCFLT) $(GSRCDBL)
GSRC_CUDA = $(patsubst %,$(GSRCDIR)/%,$(SRC_CUDA)) $(GSRCFLT_CUDA) $(GSRCDBL_CUDA)
HEA = $(patsubst $(SRCDIR)/%,%,$(filter-out %_generic.h,$(wildcard $(SRCDIR)/*.h)))
HEAGEN = $(patsubst $(SRCDIR)/%,%,$(wildcard $(SRCDIR)/*_generic.h))
GHEAFLT = $(patsubst %_generic.h,$(GSRCDIR)/%_float.h,$(HEAGEN))
GHEADBL = $(patsubst %_generic.h,$(GSRCDIR)/%_double.h,$(HEAGEN))
GHEA = $(patsubst %,$(GSRCDIR)/%,$(HEA)) $(GHEAFLT) $(GHEADBL)
OBJ = $(patsubst $(GSRCDIR)/%.c,$(BUILDDIR)/%.o,$(GSRC))
OBJDB = $(patsubst %.o,%_db.o,$(OBJ))
OBJ_CUDA = $(patsubst $(GSRCDIR)/%.cu,$(BUILDDIR)/%.o,$(GSRC_CUDA))
OBJ_CUDADB = $(patsubst %.o,%_db.o,$(OBJ_CUDA))
DEP = $(patsubst %.c,%.dep,$(GSRC)) $(patsubst %.cu,%.dep,$(GSRC_CUDA))

# --- FLAGS -------------------------------------------
COMMON_FLAGS = -DCUDA_OPT -DCUDA_ERROR_CHECK -DPROFILING
#COMMON_FLAGS = -DPROFILING

OPT_FLAGS = -fopenmp -DOPENMP -DSSE -msse4.2 -I/usr/local/cuda/include/
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
NVCC_EXTRA_COMP_FLAGS = -I/home/ramirez/installs/openmpi/include/ -L/home/ramirez/installs/openmpi/lib64/
NVCC_EXTRA_COMP_FLAGS += -lmpi
NVCC_EXTRA_COMP_FLAGS += -arch=sm_50
NVCC_EXTRA_COMP_FLAGS += -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
NVCC_EXTRA_COMP_FLAGS += -lcudart -L/usr/local/cuda/lib64/

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
	$(NVCC) --compiler-options='$(OPT_VERSION_FLAGS)' $(NVCC_EXTRA_COMP_FLAGS) $(LIMEH) -o $@ $(OBJ) $(OBJ_CUDA) $(H5LIB) $(LIMELIB) -lm

dd_alpha_amg_db : $(OBJDB) $(OBJ_CUDADB)
	$(NVCC) -g --compiler-options='$(DEBUG_VERSION_FLAGS)' $(NVCC_EXTRA_COMP_FLAGS) $(LIMEH) -o $@ $(OBJDB) $(OBJ_CUDADB) $(H5LIB) $(LIMELIB) -lm

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

$(BUILDDIR)/%.o: $(GSRCDIR)/%.c $(SRCDIR)/*.h
	$(CC) $(CFLAGS) $(OPT_VERSION_FLAGS) $(H5HEADERS) $(LIMEH) -c $< -o $@

$(BUILDDIR)/%_db.o: $(GSRCDIR)/%.c $(SRCDIR)/*.h
	$(CC) -g $(CFLAGS) $(DEBUG_VERSION_FLAGS) $(H5HEADERS) $(LIMEH) -DDEBUG -c $< -o $@

$(BUILDDIR)/%.o: $(GSRCDIR)/%.cu $(SRCDIR)/*.h
	$(NVCC) $(CFLAGS_CUDA) $(OPT_VERSION_FLAGS_CUDA) $(NVCC_EXTRA_COMP_FLAGS) -rdc=true -lcudadevrt -L/usr/local/cuda/lib64/ -c $< -o $@

$(BUILDDIR)/%_db.o: $(GSRCDIR)/%.cu $(SRCDIR)/*.h
	$(NVCC) -g $(CFLAGS_CUDA) $(DEBUG_VERSION_FLAGS_CUDA) $(NVCC_EXTRA_COMP_FLAGS) -rdc=true -lcudadevrt -L/usr/local/cuda/lib64/ -DDEBUG -c $< -o $@

$(GSRCDIR)/%.h: $(SRCDIR)/%.h $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%_float.h: $(SRCDIR)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_double.h: $(SRCDIR)/%_generic.h $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%.cu: $(SRCDIR)/%.cu $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%.c: $(SRCDIR)/%.c $(firstword $(MAKEFILE_LIST))
	cp $< $@

$(GSRCDIR)/%_float.cu: $(SRCDIR)/%_generic.cu $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_float.c: $(SRCDIR)/%_generic.c $(firstword $(MAKEFILE_LIST))
	sed -f float.sed $< > $@

$(GSRCDIR)/%_double.cu: $(SRCDIR)/%_generic.cu $(firstword $(MAKEFILE_LIST))
	sed -f double.sed $< > $@

$(GSRCDIR)/%_double.c: $(SRCDIR)/%_generic.c $(firstword $(MAKEFILE_LIST))
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
