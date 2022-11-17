TEST_SRCDIR = test
TEST_BUILDDIR = $(BUILDDIR)/test

# Flags for the C++ compiler during compile time.
TEST_CXXFLAGS = $(CFLAGS)  -g -I$(GSRCDIR) -I$(CUDA_INCLUDE)
# Flags for the C++ compiler during link time.
TEST_EXTRA_LDFLAGS = -g -lgtest -lmpi -L$(MPI_LIB)

ifeq ($(CUDA_ENABLER),-DCUDA_OPT)
	TEST_LDFLAGS = $(TEST_EXTRA_LDFLAGS) -lcudart -L$(CUDA_LIB)
	TEST_OBJ_OUT = $(TEST_DLINK_BLOB)
else
	TEST_LDFLAGS = $(TEST_EXTRA_LDFLAGS)
	TEST_OBJ_OUT = $(OBJDB_NO_MAIN)
endif

TEST_SRC = $(wildcard $(TEST_SRCDIR)/*.cpp)
TEST_OBJ = $(patsubst $(TEST_SRCDIR)/%.cpp,$(TEST_BUILDDIR)/%.o,$(TEST_SRC))
TEST_HEA = $(wildcard $(TEST_SRCDIR)/*.hpp)

# to actually use the objects created by NVCC we need an object file
# on which NVCC did perform device code linking
# see also: https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
TEST_DLINK_BLOB = $(TEST_BUILDDIR)/dd_alpha_amg_db_blob.dlink.o

$(TEST_BUILDDIR):
	# create build dir for tests if it does not already exist
	mkdir -p $(TEST_BUILDDIR)

test: build_test $(TEST_BUILDDIR)
	$(TEST_BUILDDIR)/test_dd_alpha_amg

build_test: $(TEST_BUILDDIR)/test_dd_alpha_amg

$(TEST_DLINK_BLOB): $(OBJDB_NO_MAIN) $(OBJ_CUDADB)
	$(NVCC) $(NVCC_EXTRA_COMP_FLAGS) -dlink -lcudadevrt -L$(CUDA_LIB) -o $@ $<

$(TEST_BUILDDIR)/test_dd_alpha_amg: $(TEST_OBJ) $(TEST_OBJ_OUT)
	$(CXX) $(TEST_LDFLAGS) -lgtest_main -o $@ $(TEST_OBJ) $(TEST_OBJ_OUT)

$(TEST_OBJ): $(TEST_BUILDDIR)/%.o: $(TEST_SRCDIR)/%.cpp $(TEST_HEA) $(GHEA) $(TEST_BUILDDIR)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@
