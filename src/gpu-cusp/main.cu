#include <iostream>

#include <cusp/print.h>

#include "basis.h"
#include "sparse_hamiltonian.h"


int main(int argc, char **argv)
{
  unsigned int l = 18;
  unsigned int n = 9;
  float V = 0.2;
  float t = -1.0;

  Basis basis(l,n);

  // The size of the Hilbert space
  unsigned int bsize = basis.basis_size(); 

  // Create an integer basis in a host memory container
  unsigned long long int *int_basis = new unsigned long long int[bsize];
  basis.construct_int_basis(int_basis);

  //std::cout << "Here's the basis in int notation:" << std::endl;
  //for(unsigned int i=0;i<basis.basis_size();++i) std::cout << int_basis[i] << std::endl;

  // Create an instance of the sparse hamiltonian class to fill the matrix
  // The matrix is using memory space from the host and the construct_hamiltonian_matrix
  // method fills it with the proper values
  SparseHamiltonian sparse_hamiltonian(bsize);

  sparse_hamiltonian.construct_hamiltonian_matrix(V, t, l, n, int_basis);

  // Now the matrix resides in host and device memory. 
  cusp::array2d<VType, DSpace, cusp::row_major> mat(4,4,VType (1.0, 1.0));
  cusp::array2d<VType, DSpace, cusp::row_major> exp_mat(4,4);

  cublasHandle_t handle;
  if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
    throw cusp::runtime_exception("cublasCreate failed");
  }

  sparse_hamiltonian.expm_pade(exp_mat, mat, 4, 6, handle);

  cublasDestroy(handle);
  delete [] int_basis;
  return 0;
}
