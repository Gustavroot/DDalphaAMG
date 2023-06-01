ifeq ($(CUDA_ENABLER),yes)

SRCDIR_GTEST = test/gtest
BUILDDIR_GTEST = $(BUILDDIR)/gtest
SRC_GTEST = $(shell find $(SRCDIR_GTEST) -name '*.cu')
HEA_GTEST = $(shell find $(SRCDIR_GTEST) -name '*.h')
OBJ_GTEST = $(patsubst $(SRCDIR_GTEST)/%.cu,$(BUILDDIR_GTEST)/%.o,$(SRC_GTEST))

COMPILE_FLAGS_GTEST = -I $(GSRCDIR)
ifdef GTEST_INCLUDE
	COMPILE_FLAGS_GTEST += -I$(GTEST_INCLUDE)
endif
ifdef RAPIDCHECK_INCLUDE
	COMPILE_FLAGS_GTEST += -I$(RAPIDCHECK_INCLUDE)
endif


LINK_FLAGS_GTEST = -lgtest -lgtest_main -lrapidcheck
ifdef GTEST_LIB
	LINK_FLAGS_GTEST += -L$(GTEST_LIB)
endif
ifdef RAPIDCHECK_LIB
	LINK_FLAGS_GTEST += -L$(RAPIDCHECK_LIB)
endif

$(BUILDDIR_GTEST)/%.o: $(SRCDIR_GTEST)/%.cu $(HEA_GTEST) $(GHEA)
	@mkdir -p $(@D)
	$(NVCC) -g $(COMPILE_FLAGS_GTEST) $(COMPILE_FLAGS_CUDA) $(DEBUG_VERSION_FLAGS_CUDA) -dc -DDEBUG -c $< -o $@

gtest: $(BUILDDIR_GTEST)/dd_alpha_amg_gtest

$(BUILDDIR_GTEST)/dd_alpha_amg_gtest: $(OBJ_NO_MAINDB) $(OBJ_CUDADB) $(OBJ_GTEST)
	@mkdir -p $(@D)
	$(NVCC) $(LINK_FLAGS_GTEST) $(NVCC_LINK_FLAGS) -o $@ $(OBJ_NO_MAINDB) $(OBJ_CUDADB) $(OBJ_GTEST)

endif
