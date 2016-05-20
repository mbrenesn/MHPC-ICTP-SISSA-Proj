CXX=g++
CXXFLAGS= -DNDEBUG -DBOOST_UBLAS_NDEBUG
LD=g++
LDFLAGS= -DNDEBUG -DBOOST_UBLAS_NDEBUG

all: main.x

main.x: basis.o hamiltonian.o sparse_hamiltonian.o main.o
	$(LD) $(LDFLAGS) $^ -o $@

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.x *.o
