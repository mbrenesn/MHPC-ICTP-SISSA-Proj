#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <slepcmfn.h>

class SparseHamiltonian
{
  private:
    PetscInt basis_size_;
    unsigned int l_, n_;
    Mat ham_mat_;
    Vec vec_help_;
    PetscMPIInt mpirank_, mpisize_;
    MFN mfn_;
    FN f_;
    void determine_allocation_details_(PetscInt *int_basis, 
        const PetscInt m, int start, int end, PetscInt *diag, PetscInt *off);
    inline PetscInt find_outside_(const PetscInt value);
  public:
    SparseHamiltonian(PetscInt basis_size, unsigned int l, unsigned int n, 
        int argc, char **argv);
    ~SparseHamiltonian();
    PetscMPIInt get_mpisize();
    PetscMPIInt get_mpirank();
    inline PetscInt binary_to_int(boost::dynamic_bitset<> bs);
    inline PetscInt binsearch(const PetscInt *array, PetscInt len, PetscInt value);
    void construct_hamiltonian_matrix(PetscInt *int_basis, 
        double V, double t, int nlocal, int start, int end);
    void print_hamiltonian();
    void expv_krylov_solve(const double tv, const double tol, const int maxits, Vec &w, Vec &v);
    void time_evolution(const unsigned int iterations, const double *times, 
        const double tol, const int maxits, double *loschmidt, Vec &w, const Vec &v);
};
#endif
