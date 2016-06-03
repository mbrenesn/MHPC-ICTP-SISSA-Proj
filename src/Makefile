CXX=g++
CXXFLAGS= -Wall -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
LD=g++
LDFLAGS= -Wall -g -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG

all: main.x

main.x: basis.o hamiltonian.o sparse_hamiltonian.o main.o
	$(LD) $(LDFLAGS) $^ -o $@

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.x *.o
