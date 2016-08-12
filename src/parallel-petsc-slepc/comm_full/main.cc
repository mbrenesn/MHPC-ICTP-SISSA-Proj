#include <iostream>

#include "basis.h"
#include "hamiltonian.h"
#include "sparse_hamiltonian.h"

#include <petsctime.h>

int main(int argc, char **argv)
{
  //PetscLogDouble mem = 0.0;

  unsigned int l = 32;
  unsigned int n = 16;
  double V = 0.2;
  double t = -1.0;

  Basis basis(l,n);
  // Sparse hamiltonian testing zone
  
  // Invoke the constructor with the basis size. The constructor will initialize PETSc
  // and create an instance of the PETSc MatMPIAIJ matrix type. This will also initialize
  // the MPI environment
  PetscLogDouble time1, time2;

  PetscTime(&time1);

  LLInt basis_size = basis.basis_size();

  SparseHamiltonian sparse_hamiltonian(basis_size, l, n, argc, argv);
 
  //PetscMPIInt mpirank = sparse_hamiltonian.get_mpirank();
  //PetscMPIInt mpisize = sparse_hamiltonian.get_mpisize();
  
  PetscMPIInt node_rank = sparse_hamiltonian.get_node_rank();
  //PetscMPIInt node_size = sparse_hamiltonian.get_node_size();

  if(sparse_hamiltonian.get_mpirank() == 0){  
    std::cout << "Hardcore bosons" << std::endl;    
    std::cout << "System has " << l << " sites and " << n << " particles" << std::endl;
    std::cout << "Parameters: V = " << V << " t = " << t << std::endl;
    std::cout << "Size of the Hilbert space:" << std::endl;
    std::cout << basis_size << std::endl;
  }

  //std::cout << "After creating SparseHamiltonian instance" << std::endl;
  //PetscMemoryGetCurrentUsage(&mem);
  //std::cout << "Process memory " << mem / (1024 * 1024) << " MB" << std::endl;

  // Declare vectors and parameters
  Vec w;
  Vec v;
  double tol = 1.0e-04;
  int maxits = 100000;

  // Create the vectors and let PETSc decide the distribution between processes
  VecCreate(PETSC_COMM_WORLD, &v);
  VecSetSizes(v, PETSC_DECIDE, basis_size);
  VecSetType(v, VECMPI);
 
  // An example of the RHS vector to test solutions.
  PetscInt nlocal;
  VecGetLocalSize(v, &nlocal);
  PetscInt start, end;
  VecGetOwnershipRange(v, &start, &end);

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

  //std::cout << "After construction of int basis" << std::endl;
  //PetscMemoryGetCurrentUsage(&mem);
  //std::cout << "Process memory " << mem / (1024 * 1024) << " MB" << std::endl;

  //std::cout << "Proc " << mpirank << std::endl;
  //std::cout << "Here's the basis in int notation:" << std::endl;
  //for(unsigned int i=0;i<basis_local;++i) std::cout << int_basis[i] << std::endl;

  // An example of how to populate vectors
  //PetscComplex z;
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
  PetscLogDouble constt1, constt2;

  PetscTime(&constt1);
  sparse_hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,nlocal,start,end);
  PetscTime(&constt2);

  // The int_basis is not required anymore, so let's reclaim some precious memory
  delete [] int_basis;

  //std::cout << "After deletion of int basis" << std::endl;
  //PetscMemoryGetCurrentUsage(&mem);
  //std::cout << "Process memory " << mem / (1024 * 1024) << " MB" << std::endl;
  
  //sparse_hamiltonian.print_hamiltonian();
 
  // Initial vector
  //if(sparse_hamiltonian.get_mpirank() == 0){  
  //  std::cout << "Initial state at t = 0" << std::endl;
  //}
  //VecView(v, PETSC_VIEWER_STDOUT_WORLD);

  PetscLogDouble kryt1, kryt2;

  /*** Time evolution ***/
  PetscTime(&kryt1);
 
  const int iterations = 20;
  double times[iterations + 1] 
      = {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,10,20,30,40,50,60,70,80,90,100};
  
  double *loschmidt = new double[iterations + 1]; 

  sparse_hamiltonian.time_evolution(iterations, times, tol, maxits, loschmidt, w, v);  

  PetscTime(&kryt2);
  /*** End time evolution ***/

  //std::cout << "After time evolution" << std::endl;
  //PetscMemoryGetCurrentUsage(&mem);
  //std::cout << "Process memory " << mem / (1024 * 1024) << " MB" << std::endl;
  
// Final vector
  //if(sparse_hamiltonian.get_mpirank() == 0)  
  //  std::cout << "Final state:" << std::endl;
  //VecView(w, PETSC_VIEWER_STDOUT_WORLD);

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
  
  //std::cout << "At program completion" << std::endl;
  //PetscMemoryGetCurrentUsage(&mem);
  //std::cout << "Process memory " << mem / (1024 * 1024) << " MB" << std::endl;

  return 0;
}
