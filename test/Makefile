TEST_SRCDIR = test
TEST_BUILDDIR = $(BUILDDIR)/test

-include settings.mk


# Flags for the C++ compiler during compile time.
TEST_CFLAGS = $(CFLAGS)  -g -I$(GSRCDIR) -I$(CUDA_INCLUDE) -I$(CRITERION_INCLUDE)
# Flags for the C++ compiler during link time.
TEST_EXTRA_LDFLAGS = -g -lmpi -L$(MPI_LIB) -lcriterion -L$(CRITERION_LIB)

ifeq ($(CUDA_ENABLER),-DCUDA_OPT)
	TEST_LDFLAGS = $(TEST_EXTRA_LDFLAGS) -lcudart -L$(CUDA_LIB)
	TEST_OBJ_OUT = $(TEST_DLINK_BLOB)
else
	TEST_LDFLAGS = $(TEST_EXTRA_LDFLAGS)
	TEST_OBJ_OUT = $(OBJDB_NO_MAIN)
endif

TEST_SRC = $(wildcard $(TEST_SRCDIR)/*.c)
TEST_OBJ = $(patsubst $(TEST_SRCDIR)/%.c,$(TEST_BUILDDIR)/%.o,$(TEST_SRC))
TEST_HEA = $(wildcard $(TEST_SRCDIR)/*.h)

# to actually use the objects created by NVCC we need an object file
# on which NVCC did perform device code linking
# see also: https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
TEST_DLINK_BLOB = $(TEST_BUILDDIR)/dd_alpha_amg_db_blob.dlink.o

.PHONY: test

$(TEST_BUILDDIR):
	# create build dir for tests if it does not already exist
	mkdir -p $(TEST_BUILDDIR)

test: build_test $(TEST_BUILDDIR)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(CRITERION_LIB) $(TEST_BUILDDIR)/test_dd_alpha_amg

build_test: $(TEST_BUILDDIR)/test_dd_alpha_amg

$(TEST_DLINK_BLOB): $(OBJDB_NO_MAIN) $(OBJ_CUDADB)
	$(NVCC) $(NVCC_EXTRA_COMP_FLAGS) -dlink -lcudadevrt -L$(CUDA_LIB) -o $@ $<

$(TEST_BUILDDIR)/test_dd_alpha_amg: $(TEST_OBJ) $(TEST_OBJ_OUT)
	$(CC) $(TEST_LDFLAGS) -o $@ $(TEST_OBJ) $(TEST_OBJ_OUT)

$(TEST_OBJ): $(TEST_BUILDDIR)/%.o: $(TEST_SRCDIR)/%.c $(TEST_HEA) $(GHEA) $(TEST_BUILDDIR)
	$(CC) $(TEST_CFLAGS) -c $< -o $@

clean_test:
	rm -rf $(TEST_BUILDDIR)
