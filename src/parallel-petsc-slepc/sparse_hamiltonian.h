#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <iostream>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <slepcmfn.h>

class SparseHamiltonian
{
  private:
    unsigned long long int basis_size_;
    Mat ham_mat_;
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
    void expv_krylov_solve(const double tv, const double tol, const int maxits,
            Vec &w, Vec &v);
};
#endif
