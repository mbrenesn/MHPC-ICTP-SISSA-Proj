default : full_dist.x

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

CLINKER=mpicxx
CXX=mpicxx
CXXFLAGS= -O3 -DNDEBUG
LD=mpicxx
LDFLAGS= -O3 -DNDEBUG
# Use this for external flags, like the explicit normcheck at each time iteration
EEFLAGS= #-DNORMCHECK

full_dist.x : basis.o hamiltonian.o sparse_hamiltonian.o main.o
	-${CLINKER} $(LDFLAGS) $^ -o full_dist.x ${SLEPC_SYS_LIB}

basis.o : basis.cc
	$(CXX) $(CXXFLAGS) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden $(EEFLAGS) -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include

sparse_hamiltonian.o : sparse_hamiltonian.cc
	$(CXX) $(CXXFLAGS) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden $(EEFLAGS) -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include

main.o : main.cc
	$(CXX) $(CXXFLAGS) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include

%.o : %.cc
	$(CXX) $(CXXFLAGS) -c $<

wipe : 
	rm -r *.o *.x
