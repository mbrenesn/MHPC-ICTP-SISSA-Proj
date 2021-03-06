#include "sparse_hamiltonian.h"

SparseHamiltonian::SparseHamiltonian(PetscInt basis_size, unsigned int l, 
    unsigned int n, int argc, char **argv)
{   
  SlepcInitialize(&argc, &argv, 0, 0); 
  basis_size_ = basis_size;
  l_ = l;
  n_ = n;

  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_);

  if(mpirank_ == 0 && mpisize_ > basis_size_){
    std::cerr << "Number of processors can't be larger than the basis size" << std::endl;
    exit(1);
  }
}

SparseHamiltonian::~SparseHamiltonian()
{
  MatDestroy(&ham_mat_);
  SlepcFinalize();
}

PetscMPIInt SparseHamiltonian::get_mpisize()
{
  return mpisize_;
}

PetscMPIInt SparseHamiltonian::get_mpirank()
{
  return mpirank_;
}

inline
LLInt SparseHamiltonian::binary_to_int(boost::dynamic_bitset<> bs)
{
  LLInt integer = 0;

  for(unsigned int i = 0; i < l_; ++i){
    if(bs[i] == 1){
      integer += 1ULL << i;
    }
  }

  return integer;
}

/*******************************************************************************/
// Binary search: Divide and conquer. For the construction of the Hamiltonian
// matrix instead of looking through all the elements of the int basis a
// binary search will perform better for large systems
/*******************************************************************************/
inline
LLInt SparseHamiltonian::binsearch(const LLInt *array, LLInt len, LLInt value)
{
  if(len == 0) return -1;
  LLInt mid = len / 2;

  if(array[mid] == value) 
    return mid;
  else if(array[mid] < value){
    LLInt result = binsearch(array + mid + 1, len - (mid + 1), value);
    if(result == -1) 
      return -1;
    else
      return result + mid + 1;
  }
  else
    return binsearch(array, mid, value);
}

/*******************************************************************************/
// Initial vector filled with random numbers using the Petsc rand environment.
// The initial state will change depending on the number of MPI processes used
// in this case
/*******************************************************************************/
void SparseHamiltonian::random_initial_vec(Vec &initial)
{
  PetscRandom rctx;
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  PetscRandomSetType(rctx, PETSCRAND);

  PetscRandomSetInterval(rctx, 0.0, 1.0);
  PetscRandomSeed(rctx);

  VecSetRandom(initial, rctx);

  PetscRandomDestroy(&rctx);

  VecAssemblyBegin(initial);  
  VecAssemblyEnd(initial); 

  // Normalize the initial vector
  PetscReal norm_initial;
  VecNorm(initial, NORM_2, &norm_initial);

  VecNormalize(initial, &norm_initial);
}

/*******************************************************************************/
// Initial Neel state vector.
/*******************************************************************************/
void SparseHamiltonian::neel_initial_vec(Vec &initial, LLInt *int_basis, PetscInt n, 
    PetscInt start)
{
  if(l_ / 2 != n_){
    std::cerr << "Not implemented!" << std::endl;
    std::cerr << "The Neel state has only been implemented for half-filled systems" << std::endl;
    exit(1);
  }

  boost::dynamic_bitset<> neel(l_, 1);
  for(unsigned int site = 0; site < l_; site += 2){
    neel.set(site);
  }

  VecZeroEntries(initial);

  LLInt neel_int = binary_to_int(neel);
  LLInt index = binsearch(int_basis, n, neel_int);

  if(index != -1){
    index += start;  
    VecSetValue(initial, index, 1.0, INSERT_VALUES);
  }
  VecAssemblyBegin(initial);
  VecAssemblyEnd(initial);
}


/*******************************************************************************/
// In case you didn't find your matching index in your local array, have to find
// it elsewhere given that there has to be a matching index.
/*******************************************************************************/
inline
LLInt SparseHamiltonian::find_outside_(const LLInt value, LLInt *first_values, 
    LLInt *last_values, LLInt *start_inds)
{
  LLInt smallest = 0;
  LLInt start_c = 0;
  LLInt match_ind = -1;
  // Check the range in which the value is present
  for(PetscMPIInt i = 0; i < mpisize_; ++i){
    LLInt check_first = first_values[i];
    LLInt check_last = last_values[i];
    if(value >= check_first && value <= check_last){
      smallest = first_values[i];
      start_c = start_inds[i];
      break;
    }
    else
      continue;
  }

  if(smallest == value){
    match_ind = start_c;
  }
  else{
    LLInt counter = 1;
    LLInt w = 0;
    for(LLInt i = start_c + 1; i < basis_size_; ++i){
      LLInt t = (smallest | (smallest - 1)) + 1;
      w = t | ((((t & -t) / (smallest & -smallest)) >> 1) - 1);
  
      if(w == value){
        match_ind = counter;
        break;
      }
      smallest = w;
      counter++;
    }
  
    match_ind = counter + start_c;
  }

  return match_ind;
}

/*******************************************************************************/
// Puts the nonlocal values of the integer basis in different arrays, to be used
// during the construction of the matrix
/*******************************************************************************/
void SparseHamiltonian::gather_nonlocal_values_(LLInt *int_basis, LLInt *first_values,
    LLInt *last_values, LLInt *start_inds, PetscInt nlocal, PetscInt start)
{
  MPI_Allgather(&int_basis[0], 1, MPI_LONG_LONG, first_values, 1, MPI_LONG_LONG, 
    PETSC_COMM_WORLD);
  MPI_Allgather(&int_basis[nlocal - 1], 1, MPI_LONG_LONG, last_values, 1, MPI_LONG_LONG, 
    PETSC_COMM_WORLD);
  MPI_Allgather(&start, 1, MPI_LONG_LONG, start_inds, 1, MPI_LONG_LONG, 
    PETSC_COMM_WORLD);
}

/*******************************************************************************/
// Determines the sparsity pattern to allocate memory only for the non-zero 
// entries of the matrix
/*******************************************************************************/
void SparseHamiltonian::determine_allocation_details_(LLInt *int_basis, 
    const PetscInt m, PetscInt start, PetscInt end, PetscInt *diag, PetscInt *off,
        LLInt *first_values, LLInt *last_values, LLInt *start_inds)
{
  for(PetscInt i = 0; i < m; ++i) diag[i] = 1;

  for(PetscInt state = start; state < end; ++state){
    
    boost::dynamic_bitset<> bs(l_, int_basis[state - start]);

    // Loop over all sites of the bit representation
    bool counter = false;
    for(unsigned int site = 0; site < l_; ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1) % l_;

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          counter = true;
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          LLInt new_int1 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind1 = binsearch(int_basis, m, new_int1); 
          if(match_ind1 == -1){
            match_ind1 = this->find_outside_(new_int1, first_values, last_values, start_inds);
          }
          else{
            match_ind1 += start;
          }
          if(match_ind1 == -1){
            std::cerr << "Error in the binary search within the Ham mat alloc details" << std::endl;
            exit(1);
          }
          if(match_ind1 < end && match_ind1 >= start) diag[state - start]++;
          else off[state - start]++; 
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1) % l_;

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          LLInt new_int0 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind0 = binsearch(int_basis, m, new_int0); 
          if(match_ind0 == -1){
            match_ind0 = this->find_outside_(new_int0, first_values, last_values, start_inds);
          }
          else{
            match_ind0 += start;
          }
          if(match_ind0 == -1){
            std::cerr << "Error in the binary search within the Ham mat alloc details" << std::endl;
            exit(1);
          } 
          
          if(match_ind0 < end && match_ind0 >= start) diag[state - start]++;
          else off[state - start]++;
        }
        // Otherwise do nothing
        else{
          counter = true;
          continue;
        }
      }    
    }
    
    if(counter == false) diag[state - start]--;
  }

  // In case visualization of the allocation buffers is required
  /* 
  std::cout << "Diag" << "\t" << "Proc" << mpirank_ << std::endl;
  for(int j = 0; j < m; ++j){
    std::cout << diag[j] << std::endl;
  }
  std::cout << "Off" << "\t" << "Proc" << mpirank_ << std::endl;
  for(int j = 0; j < m; ++j){
    std::cout << off[j] << std::endl;
  }
  */
}

/*******************************************************************************/
// Computes the Hamiltonian matrix given by means of the integer basis
/*******************************************************************************/
void SparseHamiltonian::construct_hamiltonian_matrix(LLInt *int_basis, 
    double V, double t, PetscInt nlocal, PetscInt start, PetscInt end)
{
  // Two arrays of size mpisize_ containing the first value and the start
  // index of the nonlocal portions of the integer basis
  LLInt *first_values = new LLInt[mpisize_];
  LLInt *last_values = new LLInt[mpisize_];
  LLInt *start_inds = new LLInt[mpisize_];

  gather_nonlocal_values_(int_basis, first_values, last_values, start_inds, nlocal, start);
  
  // Allocating memory only for the non-zero entries of the matrix
  PetscInt *d_nnz, *o_nnz;
  PetscCalloc1(nlocal, &d_nnz);
  PetscCalloc1(nlocal, &o_nnz);

  this->determine_allocation_details_(int_basis, nlocal, start, end, d_nnz, o_nnz,
    first_values, last_values, start_inds);
  
  // Create the Hamiltonian matrix
  MatCreate(PETSC_COMM_WORLD, &ham_mat_);
  MatSetSizes(ham_mat_, nlocal, nlocal, basis_size_, basis_size_);
  MatSetType(ham_mat_, MATMPIAIJ);
  
  MatMPIAIJSetPreallocation(ham_mat_, 0, d_nnz, 0, o_nnz);

  PetscFree(d_nnz);
  PetscFree(o_nnz);

  // Hamiltonian matrix construction
  PetscComplex Vi = V * PETSC_i;
  PetscComplex ti = t * PETSC_i;
  // Off-diagonal elements: the 't' terms
  // Grab 1 of the states and turn it into bit representation
  for(PetscInt state = start; state < end; ++state){
    
    boost::dynamic_bitset<> bs(l_, int_basis[state - start]);

    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < l_; ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1) % l_;

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          // Accumulate 'V' terms
          MatSetValues(ham_mat_, 1, &state, 1, &state, &Vi, ADD_VALUES);
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          LLInt new_int1 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind1 = binsearch(int_basis, nlocal, new_int1); 
          if(match_ind1 == -1){
            match_ind1 = this->find_outside_(new_int1, first_values, last_values, start_inds);
          }
          else{
            match_ind1 += start;
          }
          if(match_ind1 == -1){
            std::cerr << "Error in the binary search within the Ham mat construction" << std::endl;
            exit(1);
          } 
          
          MatSetValues(ham_mat_, 1, &match_ind1, 1, &state, &ti, ADD_VALUES);
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1) % l_;

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          LLInt new_int0 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind0 = binsearch(int_basis, nlocal, new_int0); 
          if(match_ind0 == -1){
            match_ind0 = this->find_outside_(new_int0, first_values, last_values, start_inds);
          }
          else{
            match_ind0 += start;
          }
          if(match_ind0 == -1){
            std::cerr << "Error in the binary search within the Ham mat construction" << std::endl;
            exit(1);
          } 
          
          MatSetValues(ham_mat_, 1, &match_ind0, 1, &state, &ti, ADD_VALUES);
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }
    }
  }

  MatAssemblyBegin(ham_mat_, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ham_mat_, MAT_FINAL_ASSEMBLY);

  MatSetOption(ham_mat_, MAT_SYMMETRIC, PETSC_TRUE);

  delete [] first_values;
  delete [] last_values;
  delete [] start_inds;
}

/*******************************************************************************/
// Uses PETSc's own output to view the matrix
/*******************************************************************************/
void SparseHamiltonian::print_hamiltonian()
{
  MatView(ham_mat_, PETSC_VIEWER_STDOUT_WORLD);
}

/*******************************************************************************/
// SLEPc's MFN component to solve the system. It's not important not to mix
// calls to this function with the time evolution, given that this will create
// another instance of MFN. This can be used for testing an debugging, for
// the actual time evolution use the given method.
/*******************************************************************************/
void SparseHamiltonian::expv_krylov_solve(const double tv, const double tol, const int maxits,
    Vec &w, Vec &v)
{
  MFNCreate(PETSC_COMM_WORLD, &mfn_);
  MFNSetOperator(mfn_, ham_mat_);
  
  MFNGetFN(mfn_, &f_);
  FNSetType(f_, FNEXP);
  FNSetScale(f_, tv, 1.0);
  MFNSetTolerances(mfn_, tol, maxits);
  
  MFNSetType(mfn_, MFNKRYLOV);

  MFNSetUp(mfn_);
  MFNSolve(mfn_, v, w);

  MFNDestroy(&mfn_);
}

/*******************************************************************************/
// Time evolution using SLEPc's MFN component. Need to pass an array containing
// the time values in which the first entry corresponds to the the time of the 
// initial vector. 
// The time used in consecutive iterations is t_{i+1} - t_{i}.
// This calculates the Loschmidt echo as well and puts it an an array.
/*******************************************************************************/
void SparseHamiltonian::time_evolution(const unsigned int iterations, const double *times, 
    const double tol, const int maxits, double *loschmidt, Vec &w, const Vec &v)
{
  MFNCreate(PETSC_COMM_WORLD, &mfn_);
  MFNSetOperator(mfn_, ham_mat_);
  
  MFNGetFN(mfn_, &f_);
  FNSetType(f_, FNEXP);
  MFNSetTolerances(mfn_, tol, maxits);
  
  MFNSetType(mfn_, MFNEXPOKIT);
  MFNSetUp(mfn_);

#ifdef NORMCHECK
  PetscReal normcheck;
#endif
  
  PetscComplex l_echo;
  MFNConvergedReason reason;
  // A copy of the initial vector, given that we need the initial state for later
  // computations
  VecDuplicate(v, &vec_help_);
  VecCopy(v, vec_help_);

  VecDot(v, v, &l_echo);
  loschmidt[0] = (PetscRealPart(l_echo) * PetscRealPart(l_echo))
      + (PetscImaginaryPart(l_echo) * PetscImaginaryPart(l_echo));

  if(mpirank_ == 0)
    std::cout << "Time" << "\t" << "Loschmidt echo" << std::endl;
  for(unsigned int tt = 1; tt < (iterations + 1); ++tt){
  
    FNSetScale(f_, times[tt] - times[tt - 1], 1.0);

    MFNSolve(mfn_, vec_help_, w);

    MFNGetConvergedReason(mfn_, &reason);
    if(reason < 0){
      std::cerr << "WARNING! Solver did not converged with given parameters" << std::endl;
      std::cerr << "Change the tolerance or the maximum number of iterations" << std::endl;
      exit(1);
    }

    VecDot(v, w, &l_echo);
    loschmidt[tt] = (PetscRealPart(l_echo) * PetscRealPart(l_echo))
        + (PetscImaginaryPart(l_echo) * PetscImaginaryPart(l_echo));

    if(mpirank_ == 0)
      std::cout << times[tt] << "\t" << loschmidt[tt] << std::endl;
#ifdef NORMCHECK
    VecNorm(w, NORM_2, &normcheck);
    if(normcheck > 1.001 || normcheck < 0.999){
      std::cerr << "Normcheck failed!" << std::endl;
      exit(1);
    }
#endif

    VecCopy(w, vec_help_);
    //if(mpirank_ == 0)  
    //  std::cout << "Delta_Time" << "\t" << times[tt] -times[tt - 1] << std::endl;
    //VecView(w, PETSC_VIEWER_STDOUT_WORLD);
  }

  VecDestroy(&vec_help_);
  MFNDestroy(&mfn_);
}
