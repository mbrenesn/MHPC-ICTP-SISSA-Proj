default : dist_comm.x

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

CLINKER=mpicxx
CXX=mpicxx
CXXFLAGS=-g -O3 -mavx -DNDEBUG
LD=mpicxx
LDFLAGS=-g -O3 -mavx -DNDEBUG
# Use this for external flags, like the explicit normcheck at each time iteration
EEFLAGS= #-DNORMCHECK

dist_comm.x : basis.o hamiltonian.o sparse_hamiltonian.o main.o
	-${CLINKER} $(LDFLAGS) $^ -o dist_comm.x ${SLEPC_SYS_LIB}

basis.o : basis.cc
	$(CXX) $(CXXFLAGS) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden $(EEFLAGS) -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(BOOST_DIR)

sparse_hamiltonian.o : sparse_hamiltonian.cc
	$(CXX) $(CXXFLAGS) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden $(EEFLAGS) -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(BOOST_DIR)

main.o : main.cc
	$(CXX) $(CXXFLAGS) -c $< -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(BOOST_DIR)

%.o : %.cc
	$(CXX) $(CXXFLAGS) -c $< -I$(BOOST_DIR)

wipe : 
	rm -r *.o *.x
