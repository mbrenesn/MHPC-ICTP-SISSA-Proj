#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <slepcmfn.h>

typedef unsigned long long int ULLInt;
typedef long long int LLInt;

class SparseHamiltonian
{
  private:
    LLInt basis_size_;
    unsigned int l_, n_;
    Mat ham_mat_;
    Vec vec_help_;
    MPI_Comm node_comm_;
    PetscMPIInt mpirank_, mpisize_, node_rank_, node_size_;
    MFN mfn_;
    FN f_;
    void determine_allocation_details_(LLInt *int_basis, LLInt *recv_sizes, 
        const PetscInt m, PetscInt start, PetscInt end, PetscInt *diag, PetscInt *off);
  public:
    SparseHamiltonian(LLInt basis_size, unsigned int l, unsigned int n, 
        int argc, char **argv);
    ~SparseHamiltonian();
    PetscMPIInt get_mpisize();
    PetscMPIInt get_mpirank();
    PetscMPIInt get_node_size();
    PetscMPIInt get_node_rank();
    inline LLInt binary_to_int(boost::dynamic_bitset<> bs);
    inline LLInt binsearch(const LLInt *array, LLInt len, LLInt value);
    void construct_hamiltonian_matrix(LLInt *int_basis, 
        double V, double t, PetscInt nlocal, PetscInt start, PetscInt end);
    void print_hamiltonian();
    void expv_krylov_solve(const double tv, const double tol, const int maxits, Vec &w, Vec &v);
    void time_evolution(const unsigned int iterations, const double *times, 
        const double tol, const int maxits, double *loschmidt, Vec &w, const Vec &v);
};
#endif
