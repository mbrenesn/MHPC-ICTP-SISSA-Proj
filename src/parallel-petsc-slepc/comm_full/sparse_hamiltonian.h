#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <ctime>

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
    void distribution(PetscInt &nlocal, PetscInt &start, PetscInt &end);
    inline LLInt binary_to_int(boost::dynamic_bitset<> bs);
    inline LLInt binsearch(const LLInt *array, LLInt len, LLInt value);
    void random_initial_vec(Vec &initial);
    void get_neel_index(LLInt &index, LLInt *int_basis, PetscInt n);
    void get_random_initial_pick(LLInt &pick_ind, LLInt *int_basis, bool wtime, bool verbose);
    void neel_initial_vec(Vec &initial, LLInt neel_index);
    void random_initial_basis_vec(Vec &initial, LLInt random_pick);
    void construct_hamiltonian_matrix(LLInt *int_basis, 
        double V, double t, PetscInt nlocal, PetscInt start, PetscInt end);
    void print_hamiltonian();
    void expv_krylov_solve(const double tv, const double tol, const int maxits, Vec &w, Vec &v);
    void time_evolution(const unsigned int iterations, const double *times, 
        const double tol, const int maxits, double *loschmidt, Vec &w, const Vec &v);
};
#endif
