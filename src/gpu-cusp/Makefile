CXX=g++
CXXFLAGS= -Wall -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
NXX=nvcc
NXXFLAGS= -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
LD=nvcc
LDFLAGS= -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
INC= -I/home/mbrenesn/cusplibrary-0.5.1

all: gpu.x

gpu.x: basis.o sparse_hamiltonian.o main.o
	$(LD) $(LDFLAGS) $^ -o $@

main.o : main.cu
	$(NXX) $(NXXFLAGS) -c $< $(INC)

%.o : %.cu %.h
	$(NXX) $(NXXFLAGS) -c $< $(INC)

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.x *.o
