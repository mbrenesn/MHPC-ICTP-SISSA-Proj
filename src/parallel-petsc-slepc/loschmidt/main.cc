#include <iostream>

#include "basis.h"
#include "hamiltonian.h"
#include "sparse_hamiltonian.h"

#include <petsctime.h>

int main(int argc, char **argv)
{
  unsigned int l = 18;
  unsigned int n = 9;
  double V = 0.2;
  double t = -1.0;
  double h = 1.0;
  double beta = 5.0 / 8.0;

  Basis basis(l,n);
  // Sparse hamiltonian testing zone
  
  // Invoke the constructor with the basis size. The constructor will initialize PETSc
  // and create an instance of the PETSc MatMPIAIJ matrix type. This will also initialize
  // the MPI environment
  LLInt basis_size = basis.basis_size();
  
  SparseHamiltonian sparse_hamiltonian(basis_size, l, n, argc, argv);
  
  PetscLogDouble time1, time2;

  PetscTime(&time1);
 
  PetscMPIInt mpirank = sparse_hamiltonian.get_mpirank();
  PetscMPIInt mpisize = sparse_hamiltonian.get_mpisize();
  
  PetscMPIInt node_rank = sparse_hamiltonian.get_node_rank();
  PetscMPIInt node_size = sparse_hamiltonian.get_node_size();

  if(mpirank == 0){  
    std::cout << "Hardcore bosons" << std::endl;    
    std::cout << "System has " << l << " sites and " << n << " particles" << std::endl;
    std::cout << "Simulation with " << mpisize << " total MPI processes" << std::endl;
    std::cout << "Amount of processes user per each node is " << node_size << std::endl;
    std::cout << "Parameters: V = " << V << " t = " << t << std::endl;
    std::cout << "Size of the Hilbert space:" << std::endl;
    std::cout << basis_size << std::endl;
  }

  // Create the parallel distribution of objects.
  // This distribution is consistent with the distribution PETSc uses, you can for instance
  // use *GetLocalSize or *GetOwnerShipRange and get the same parallel distribution
  PetscInt nlocal, start, end;

  sparse_hamiltonian.distribution(nlocal, start, end);

  // Let's construct the int basis that is required to construct the Hamiltonian matrix
  // Each processor creates and holds a section of the integer basis, except rank 0
  // of each node which creates the whole basis
  PetscInt basis_local = nlocal;
  if(node_rank == 0){
    basis_local = basis.basis_size();
  }
 
  LLInt *int_basis = new LLInt[basis_local];
  
  PetscInt basis_start = start;
  if(node_rank == 0){
    basis_start = 0;
  }
  
  basis.construct_int_basis(int_basis, basis_local, basis_start); 

  // Get the information required to populate the initial vector before destroying the basis 

  // Neel initial index
  LLInt neel_index;
  sparse_hamiltonian.get_neel_index(neel_index, int_basis, basis_local);

  // We construct the Hamiltonian matrix using the above distribution
  PetscLogDouble constt1, constt2;

  PetscTime(&constt1);
  sparse_hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,h,beta,nlocal,start,end);
  PetscTime(&constt2);

  // The int_basis is not required anymore, so let's reclaim some precious memory
  delete [] int_basis;

  // Declare vectors and parameters
  Vec w;
  Vec v;
  double tol = 1.0e-07;
  int maxits = 100000;
 
  /***/

  //sparse_hamiltonian.print_hamiltonian();

  //Populate the initial vector
  VecCreate(PETSC_COMM_WORLD, &v);
  VecSetSizes(v, nlocal, basis_size);
  VecSetType(v, VECMPI);

  VecDuplicate(v, &w);
  
  // Neel initial vector
  sparse_hamiltonian.neel_initial_vec(v, neel_index);

  /***/

  PetscLogDouble kryt1, kryt2;

  /*** Time evolution ***/
  PetscTime(&kryt1);
 
  const int iterations = 36;
  double times[iterations + 1] 
      = {0.0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0};
  
  double *loschmidt = new double[iterations + 1]; 

  sparse_hamiltonian.time_evolution(iterations, times, tol, maxits, loschmidt, w, v);  

  PetscTime(&kryt2);
  /*** End time evolution ***/

  PetscReal norm;
  VecNorm(w, NORM_2, &norm);
 
  PetscTime(&time2);

  if(sparse_hamiltonian.get_mpirank() == 0){  
    std::cout << "L2 norm of the final state vec: " << norm << std::endl; 
    std::cout << "Time Construction: " << constt2 - constt1 << " seconds" << std::endl; 
    std::cout << "Time Krylov Evolution: " << kryt2 - kryt1 << " seconds" << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    std::cout << "Time total: " << time2 - time1 << " seconds" << std::endl; 
  }

  VecDestroy(&v);  
  VecDestroy(&w);
  delete [] loschmidt;
  
  return 0;
}
