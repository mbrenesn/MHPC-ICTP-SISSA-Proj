#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <iostream>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

class SparseHamiltonian
{
  private:
    unsigned long long int basis_size_;
  public:
    SparseHamiltonian(unsigned long long int basis_size);
    ~SparseHamiltonian();
    boost::numeric::ublas::compressed_matrix<double> ham_mat;
    inline unsigned long long int binary_to_int(boost::dynamic_bitset<> bs, unsigned int l);
    void construct_hamiltonian_matrix(unsigned long long int* int_basis, 
            double V, double t, unsigned int l, unsigned int n);
    void print_hamiltonian();
};
#endif
