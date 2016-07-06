#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <cusp/coo_matrix.h>
#include <cusp/complex.h>
#include <cusp/print.h>

#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

typedef unsigned int IType;
typedef cusp::complex<double> VType;
typedef cusp::device_memory DSpace;
typedef cusp::host_memory HSpace;

class SparseHamiltonian
{
  private:
    unsigned int basis_size_;
    cusp::coo_matrix<IType, VType, HSpace> ham_mat_host_;
  public:
    SparseHamiltonian(unsigned int basis_size);
    ~SparseHamiltonian();
    inline unsigned int binary_to_int(boost::dynamic_bitset<> bs, unsigned int l);
    long int binsearch(const unsigned long long int *array, 
        long int len, long int value);
    void determine_allocation_details(const unsigned long long int *int_basis, unsigned int l,
        unsigned int &counter);
    void construct_hamiltonian_matrix(double V, double t, unsigned int l,  
        unsigned int n, const unsigned long long int *int_basis);
    //void print_hamiltonian();
    //double sign(double x);
    //boost::numeric::ublas::matrix< std::complex<double> > 
    //    expm_pade(boost::numeric::ublas::matrix< std::complex<double> > mat,unsigned int p);
    //void expv_krylov_solve(double tv, boost::numeric::ublas::vector< std::complex<double> > &w,
    //        double &err, double &hump, 
    //            boost::numeric::ublas::vector< std::complex<double> > &v, 
    //                double tol, unsigned int m);
};
#endif
