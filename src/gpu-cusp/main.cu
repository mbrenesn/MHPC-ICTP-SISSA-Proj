#include <iostream>

#include <cusp/print.h>

#include "basis.h"
#include "sparse_hamiltonian.h"


int main(int argc, char **argv)
{
  unsigned int l = 4;
  unsigned int n = 2;
  double V = 0.2;
  double t = -1.0;

  Basis basis(l,n);

  // The size of the Hilbert space
  unsigned int bsize = basis.basis_size(); 

  // Create an integer basis in a host memory container
  unsigned long long int *int_basis = new unsigned long long int[bsize];
  basis.construct_int_basis(int_basis);

  std::cout << "Here's the basis in int notation:" << std::endl;
  for(unsigned int i=0;i<basis.basis_size();++i) std::cout << int_basis[i] << std::endl;

  // Create an instance of the sparse hamiltonian class to fill the matrix
  // The matrix is using memory space from the host and the construct_hamiltonian_matrix
  // method fills it with the proper values
  SparseHamiltonian sparse_hamiltonian(bsize);

  sparse_hamiltonian.construct_hamiltonian_matrix(V, t, l, n, int_basis);

  delete [] int_basis;
  return 0;
}
