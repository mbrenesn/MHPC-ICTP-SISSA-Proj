#ifndef __SPARSE_HAMILTONIAN_H
#define __SPARSE_HAMILTONIAN_H

#include <cusp/coo_matrix.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/blas/blas.h>
#include <cusp/system/cuda/detail/cublas/blas.h>
#include <cusp/linear_operator.h>
#include <cusp/complex.h>
#include <cusp/print.h>

#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

typedef int IType;
typedef cusp::complex<float> VType;
typedef cusp::device_memory DSpace;
typedef cusp::host_memory HSpace;

class SparseHamiltonian
{
  private:
    unsigned int basis_size_;
  public:
    cusp::coo_matrix<IType, VType, HSpace> ham_mat_host;
    cusp::coo_matrix<IType, VType, DSpace> ham_mat_device;
    SparseHamiltonian(unsigned int basis_size);
    ~SparseHamiltonian();
    inline unsigned int binary_to_int(boost::dynamic_bitset<> bs, unsigned int l);
    long int binsearch(const unsigned long long int *array, 
        long int len, long int value);
    void determine_allocation_details(const unsigned long long int *int_basis, unsigned int l,
        unsigned int &counter);
    void construct_hamiltonian_matrix(float V, float t, unsigned int l,  
        unsigned int n, const unsigned long long int *int_basis);
    //double sign(double x);
    void expm_pade(cusp::array2d<VType, DSpace> exp_mat,
        cusp::array2d<VType, DSpace> mat, int n, unsigned int p, cublasHandle_t handle);
    //void expv_krylov_solve(double tv, boost::numeric::ublas::vector< std::complex<double> > &w,
    //        double &err, double &hump, 
    //            boost::numeric::ublas::vector< std::complex<double> > &v, 
    //                double tol, unsigned int m);
};
#endif
