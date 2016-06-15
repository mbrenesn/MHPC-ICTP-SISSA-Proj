#include <iostream>

#include "basis.h"
#include "hamiltonian.h"
#include "sparse_hamiltonian.h"

#include <petsctime.h>

int main(int argc, char **argv)
{
  unsigned int l = 4;
  unsigned int n = 2;
  double V = 0.2;
  double t = -1.0;

  Basis basis(l,n);

  unsigned long long int *int_basis = new unsigned long long int[basis.basis_size()];
  basis.construct_int_basis(int_basis);

#if 0

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

#endif

  // Sparse hamiltonian testing zone
  
  // Invoke the constructor with the basis size. The constructor will initialize PETSc
  // and create an instance of the PETSc MatMPIAIJ matrix type. 
  PetscLogDouble time1, time2;

  PetscTime(&time1);
 
  SparseHamiltonian sparse_hamiltonian(basis.basis_size(), argc, argv);

  // The matrix is populated using the construct_hamiltonian_matrix method. The matrix
  // distribution among processors will inherit the distribution of the RHS vector of
  // the equation. PETSc uses distribution of rows among processors

  // Declare vectors and parameters
  Vec w;
  Vec v;
  double tol = 1.0e-06;
  int maxit = 30;

  // Create the vectors and let PETSc decide the distribution between processes
  VecCreate(PETSC_COMM_WORLD, &v);
  VecSetSizes(v, PETSC_DECIDE, basis.basis_size());
  VecSetType(v, VECMPI);
 
  // An example of the RHS vector to test solutions. This will change into a prepared 
  // initial quantum state
  //PetscComplex z;
  PetscInt start, end;
  VecGetOwnershipRange(v, &start, &end);

  // An example of how to populate vectors
  //for(int ii = start; ii < end; ++ii){
  //  z = std::complex<double>(ii+1,0); 
  //  VecSetValues(v, 1, &ii, &z, INSERT_VALUES);
  //}

  // Random population of the initial vector
  PetscRandom rctx;
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  PetscRandomSetType(rctx, PETSCRAND);

  PetscRandomSetInterval(rctx, 0.0, 1.0);
  PetscRandomSeed(rctx);

  VecSetRandom(v, rctx);

  PetscRandomDestroy(&rctx);

  VecAssemblyBegin(v);  
  VecAssemblyEnd(v);

  // Normalize the initial vector
  PetscReal norm_initial;
  VecNorm(v, NORM_2, &norm_initial);

  VecNormalize(v, &norm_initial);

  // Construct a time evolution vector in the same manner as the initial vector
  // was constructed. Values are NOT copied over.
  VecDuplicate(v, &w);

  // We let PETSc decide the distribution among processes, let's pass this to the
  // construct_hamiltonian_matrix method so we create a Hamiltonian matrix with
  // the same distribution. This will make the operations compatible among processors
  int nlocal;
  VecGetLocalSize(v, &nlocal);

  PetscLogDouble constt1, constt2;

  PetscTime(&constt1);
  sparse_hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,l,n,nlocal,start,end);
  PetscTime(&constt2);

  // The int_basis is not required anymore, so let's reclaim some precious memory
  delete [] int_basis;

  sparse_hamiltonian.print_hamiltonian();
 
  std::cout << "Initial state at t = 0" << std::endl;
  VecView(v, PETSC_VIEWER_STDOUT_WORLD);

  PetscLogDouble kryt1, kryt2;

  /*** Time evolution ***/
  PetscTime(&kryt1);
  
  unsigned int iterations = 10;
  double times[iterations] = {0.0,0.1,0.2,0.3,0.4,0.5,1.0,2.0,5.0,10.0};
  PetscReal normcheck;
  for(unsigned int tt = 1; tt < iterations; ++tt){
   
    sparse_hamiltonian.expv_krylov_solve(times[tt] - times[tt - 1], tol, maxit, w, v);
    VecNorm(w, NORM_2, &normcheck);
    if(normcheck > 1.001 || normcheck < 0.999){
      std::cerr << "Normcheck failed!" << std::endl;
      exit(1);
    }

    VecCopy(w, v);
    std::cout << "Time" << "\t" << times[tt] << std::endl;
    VecView(v, PETSC_VIEWER_STDOUT_WORLD);
  }
   
  PetscTime(&kryt2);
  /*** End time evolution ***/
  
  // Now the state at the last step is stored in v
  VecView(w, PETSC_VIEWER_STDOUT_WORLD);

  PetscReal norm;
  VecNorm(w, NORM_2, &norm);
  
  PetscTime(&time2);

  if(sparse_hamiltonian.get_mpirank() == 0){  
    std::cout << "Size of the Hilbert space:" << std::endl;
    std::cout << basis.basis_size() << std::endl;
    std::cout << "L2 norm of the final state vec: " << norm << std::endl; 
    std::cout << "Time Construction: " << constt2 - constt1 << " seconds" << std::endl; 
    std::cout << "Time Krylov Evolution: " << kryt2 - kryt1 << " seconds" << std::endl; 
    std::cout << "Time total: " << time2 - time1 << " seconds" << std::endl; 
  }

  VecDestroy(&v);  
  VecDestroy(&w);
  //delete [] bit_basis;
  return 0;
}
