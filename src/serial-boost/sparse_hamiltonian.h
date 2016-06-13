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

class SparseHamiltonian
{
  private:
    unsigned long long int basis_size_;
    boost::numeric::ublas::compressed_matrix< std::complex<double> > ham_mat_;
  public:
    SparseHamiltonian(unsigned long long int basis_size,
            boost::numeric::ublas::compressed_matrix< std::complex<double> > &ham_mat);
    ~SparseHamiltonian();
    boost::numeric::ublas::vector<double> w;
    inline unsigned long long int binary_to_int(boost::dynamic_bitset<> bs, unsigned int l);
    long long int binsearch(const unsigned long long int *array, 
            unsigned long long int len, unsigned long long int value);
    void construct_hamiltonian_matrix(unsigned long long int* int_basis, 
            double V, double t, unsigned int l, unsigned int n);
    void print_hamiltonian();
    double sign(double x);
    boost::numeric::ublas::matrix< std::complex<double> > 
        expm_pade(boost::numeric::ublas::matrix< std::complex<double> > mat,unsigned int p);
    void expv_krylov_solve(double tv, boost::numeric::ublas::vector< std::complex<double> > &w,
            double &err, double &hump, 
                boost::numeric::ublas::vector< std::complex<double> > &v, 
                    double tol, unsigned int m);
};
#endif
