-include test/MakeSettings.mk
-include test/mpicases/Makefile

TEST_SRCDIR = test
MPICASES_SRCDIR = $(TEST_SRCDIR)/mpicases
TEST_BUILDDIR = $(BUILDDIR)/test
MPICASES_BUILDDIR = $(TEST_BUILDDIR)/mpicases

# Flags for the C++ compiler during compile time.
TEST_CFLAGS = $(CFLAGS) -g -I$(GSRCDIR) -isystem$(CUDA_INCLUDE) -isystem$(CRITERION_INCLUDE) \
	-fopenmp
# Flags for the C++ compiler during link time.
TEST_LDFLAGS = -g -lmpi -L$(MPI_LIB) -lm -lcudart -lcudadevrt \
	-L$(CUDA_LIB) -Xcompiler "-fopenmp"

TEST_SRC = $(wildcard $(TEST_SRCDIR)/*.c)
TEST_OBJ = $(patsubst $(TEST_SRCDIR)/%.c,$(TEST_BUILDDIR)/%.o,$(TEST_SRC))
TEST_HEA = $(wildcard $(TEST_SRCDIR)/*.h)

MPICASES_SRC = $(wildcard $(MPICASES_SRCDIR)/*.c)
MPICASES_OBJ = $(patsubst $(MPICASES_SRCDIR)/%.c,$(MPICASES_BUILDDIR)/%.o,$(MPICASES_SRC))
MPICASES_EXE = $(patsubst $(MPICASES_SRCDIR)/%.c,$(MPICASES_BUILDDIR)/%,$(MPICASES_SRC))
# This is basically just phony targets
MPICASES_RUNTARGET = $(patsubst $(MPICASES_SRCDIR)/%.c,$(MPICASES_BUILDDIR)/%.runtarget,$(MPICASES_SRC))
MPICASES_HEA = $(wildcard $(MPICASES_SRCDIR)/*.h)

.PHONY: test

$(TEST_BUILDDIR):
	# create build dir for tests if it does not already exist
	mkdir -p $(TEST_BUILDDIR)

$(MPICASES_BUILDDIR):
	mkdir -p $(MPICASES_BUILDDIR)

test: build_test $(TEST_BUILDDIR)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(CRITERION_LIB) $(TEST_BUILDDIR)/test_dd_alpha_amg

build_test: $(TEST_BUILDDIR)/test_dd_alpha_amg

mpicases: $(MPICASES_RUNTARGET)

$(MPICASES_RUNTARGET): $(MPICASES_BUILDDIR)/%.runtarget: $(MPICASES_BUILDDIR)/%
	mpirun -np 2 $<


$(MPICASES_EXE): $(MPICASES_BUILDDIR)/%: $(MPICASES_BUILDDIR)/%.o library_db
	$(NVCC) $(TEST_LDFLAGS) -o $@ $< ./lib/libdd_alpha_amg_db.a

$(TEST_BUILDDIR)/test_dd_alpha_amg: $(TEST_OBJ) library_db
# TODO actually NVCC should not be required as a linker, but otherwise we get linker errors
# relating to relocation of device code.
	$(NVCC) $(TEST_LDFLAGS) -lcriterion -L$(CRITERION_LIB) -o $@ $(TEST_OBJ) ./lib/libdd_alpha_amg_db.a

$(MPICASES_OBJ): $(MPICASES_BUILDDIR)/%.o: $(MPICASES_SRCDIR)/%.c $(GHEA) $(MPICASES_BUILDDIR)
	$(CC) $(TEST_CFLAGS) -c $< -o $@

$(TEST_OBJ): $(TEST_BUILDDIR)/%.o: $(TEST_SRCDIR)/%.c $(TEST_HEA) $(GHEA) $(TEST_BUILDDIR)
	$(CC) $(TEST_CFLAGS) -c $< -o $@

clean_test:
	rm -rf $(TEST_BUILDDIR)
