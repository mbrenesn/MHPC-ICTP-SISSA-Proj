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
    unsigned long long int basis_size_;
    Mat ham_mat_;
    Vec vec_help_;
    PetscMPIInt mpirank_, mpisize_;
    MFN mfn_;
    FN f_;
    void determine_allocation_details(unsigned long long int* int_basis, 
        unsigned int l, const PetscInt m, int start, int end, PetscInt *diag, 
            PetscInt *off);
  public:
    SparseHamiltonian(unsigned long long int basis_size, int argc, char **argv);
    ~SparseHamiltonian();
    int get_mpisize();
    int get_mpirank();
    inline unsigned long long int binary_to_int(boost::dynamic_bitset<> bs, unsigned int l);
    inline int binsearch(const unsigned long long int *array, 
        unsigned long long int len, unsigned long long int value);
    void construct_hamiltonian_matrix(unsigned long long int* int_basis, 
        double V, double t, unsigned int l, 
            unsigned int n, int nlocal, int start, int end);
    void print_hamiltonian();
    void expv_krylov_solve(const double tv, const double tol, const int maxits, Vec &w, Vec &v);
    void time_evolution(const unsigned int iterations, const double *times, 
        const double tol, const int maxits, double *loschmidt, Vec &w, const Vec &v);
};
#endif
