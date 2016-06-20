default : parallel.x

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

CLINKER=mpic++
CXX=mpic++
CXXFLAGS= -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
LD=mpic++
LDFLAGS= -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
# Use this for external flags, like the explicit normcheck at each time iteration
EEFLAGS= #-DNORMCHECK

parallel.x : basis.o hamiltonian.o sparse_hamiltonian.o main.o
	-${CLINKER} $^ -o parallel.x ${SLEPC_SYS_LIB}

sparse_hamiltonian.o : sparse_hamiltonian.cc
	$(CXX) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden -g -O3 $(EEFLAGS) -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/x86_64/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/x86_64/include -I$(MPI_DIR)/include

main.o : main.cc
	$(CXX) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden -g -O3 -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/x86_64/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/x86_64/include -I$(MPI_DIR)/include

%.o : %.cc
	$(CXX) $(CXXFLAGS) -c $<

wipe : 
	rm -r *.o
