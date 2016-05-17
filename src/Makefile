CXX=g++
CXXFLAGS=
LD=g++
LDFLAGS=

all: main.x

main.x: basis.o hamiltonian.o sparse_hamiltonian.o main.o
	$(LD) $(LDFLAGS) $^ -o $@

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.x *.o
