CXX=g++
CXXFLAGS= -Wall -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
NXX=nvcc
NXXFLAGS= -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG -arch=sm_35
LD=nvcc
LDFLAGS= -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG -arch=sm_35
LDD= -L $(OPENBLAS_LIB) -lcublas -lcudart
INC= -I $(CUSP_DIR) -I $(OPENBLAS_INC)

all: gpu.x

gpu.x: basis.o sparse_hamiltonian.o main.o
	$(LD) $(LDFLAGS) $^ -o $@ $(LDD)

main.o : main.cu
	$(NXX) $(NXXFLAGS) -c $< $(INC)

%.o : %.cu %.h
	$(NXX) $(NXXFLAGS) -c $< $(INC)

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.x *.o
