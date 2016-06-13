#include <iostream>

#include <sys/time.h>
#include <sys/resource.h>

#include "basis.h"
#include "hamiltonian.h"
#include "sparse_hamiltonian.h"

double seconds()
{
    //Returns the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

int main(int argc, char **argv)
{
  unsigned int l = 4;
  unsigned int n = 2;
  double V = 0.2;
  double t = -1.0;

  Basis basis(l,n);

  unsigned long long int *int_basis = new unsigned long long int[basis.basis_size()];
  basis.construct_int_basis(int_basis);

  std::cout << "Here's the basis in int notation:" << std::endl;
  for(unsigned int i=0;i<basis.basis_size();++i) std::cout << int_basis[i] << std::endl;

  boost::dynamic_bitset<> *bit_basis = 
      new boost::dynamic_bitset<>[basis.basis_size()];
  basis.construct_bit_basis(bit_basis, int_basis);
  
  std::cout << "Here's the basis in binary notation:" << std::endl;
  for(unsigned int i=0;i<basis.basis_size();++i) std::cout << bit_basis[i] << std::endl;

  // Construction of the hamiltonian matrix, by calling the constructor
  // the object is the hamiltonian matrix itself
  Hamiltonian hamiltonian(basis.basis_size());

  // Call the method to construct_hamiltonian_matrix to populate the entries
  // Takes the integer basis, number of sites and particles as arguments
  hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,l,n);

  // This is a 1D array representation of the matrix, [][] is overloaded so you can
  // access the elements with the [] operator, () is a 'BETTER' choice.
  for(unsigned int i=0;i<basis.basis_size();++i){
    for(unsigned int j=0;j<basis.basis_size();++j){
      std::cout << " " << hamiltonian(i,j);
    }
    std::cout << std::endl;
  }

  // Sparse hamiltonian testing zone
  
  // Invoke the constructor with the basis size. The constructor will initialize PETSc
  // and create an instance of the PETSc MatMPIAIJ matrix type. 
 
  double t1 = seconds();
  
  SparseHamiltonian sparse_hamiltonian(basis.basis_size(), argc, argv);

  // The matrix is populated using the construct_hamiltonian_matrix method. The matrix
  // distribution among processors will inherit the distribution of the RHS vector of
  // the equation. PETSc uses distribution of rows among processors

  // Declare vectors and parameters
  Vec w;
  Vec v;
  double tv = 1.0;
  double tol = 1.0e-06;
  int maxit = 30;

  // Create the vectors and let PETSc decide the distribution between processes
  VecCreate(PETSC_COMM_WORLD, &v);
  VecSetSizes(v, PETSC_DECIDE, basis.basis_size());
  VecSetType(v, VECMPI);
 
  // An example of the RHS vector to test solutions. This will change into a prepared 
  // initial quantum state
  PetscComplex z;
  PetscInt start, end;
  VecGetOwnershipRange(v, &start, &end);

  for(int ii = start; ii < end; ++ii){
    z = std::complex<double>(ii+1,ii+1); 
    VecSetValues(v, 1, &ii, &z, INSERT_VALUES);
  }

  VecAssemblyBegin(v);  
  VecAssemblyEnd(v);

  VecDuplicate(v, &w);

  // We let PETSc decide the distribution among processes, let's pass this to the
  // construct_hamiltonian_matrix method so we create a Hamiltonian matrix with
  // the same distribution. This will make the operations compatible among processors
  int nlocal;
  VecGetLocalSize(v, &nlocal);

  sparse_hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,l,n,nlocal,start,end);

  sparse_hamiltonian.print_hamiltonian();
  
  VecView(v, PETSC_VIEWER_STDOUT_WORLD);

  sparse_hamiltonian.expv_krylov_solve(tv, tol, maxit, w, v);
  
  VecView(w, PETSC_VIEWER_STDOUT_WORLD);

  PetscReal norm;
  VecNorm(w, NORM_2, &norm);

  double t2 = seconds();
  
  if(sparse_hamiltonian.get_mpirank() == 0){  
    std::cout << "Size of the Hilbert space:" << std::endl;
    std::cout << basis.basis_size() << std::endl;
    std::cout << "L2 norm of the w vec: " << norm << std::endl; 
    std::cout << "Time: " << t2 - t1 << " seconds" << std::endl; 
  }

  VecDestroy(&v);  
  VecDestroy(&w);
  delete [] int_basis;
  delete [] bit_basis;
  return 0;
}
