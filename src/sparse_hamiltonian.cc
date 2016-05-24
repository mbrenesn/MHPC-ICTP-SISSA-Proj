#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "sparse_hamiltonian.h"

// TODO
// Be careful with the copy!
SparseHamiltonian::SparseHamiltonian(unsigned long long int basis_size)
{
  basis_size_ = basis_size;
  boost::numeric::ublas::compressed_matrix<double> loc_ham_mat(basis_size_, basis_size_, 0.0);

  this->ham_mat = loc_ham_mat;
}

SparseHamiltonian::~SparseHamiltonian()
{}

inline
unsigned long long int SparseHamiltonian::binary_to_int(boost::dynamic_bitset<> bs, 
        unsigned int l)
{
  unsigned long long integer = 0;

  for(unsigned int i = 0; i < l; ++i){
    if(bs[i] == 1){
      integer += 1 << i;
    }
  }

  return integer;
}

/*******************************************************************************/
// Computes the Hamiltonian matrix given by means of the integer basis
/*******************************************************************************/
void SparseHamiltonian::construct_hamiltonian_matrix(unsigned long long int* int_basis, 
        double V, double t, unsigned int l, unsigned int n)
{
  // Off-diagonal elements: the 't' terms
  // Grab 1 of the states and turn it into bit representation
  for(unsigned int state = 0; state < basis_size_; ++state){
    
    boost::dynamic_bitset<> bs(l, int_basis[state]);

    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < l; ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1) % l;

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          // Accumulate 'V' terms
          ham_mat(state, state) += V;
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          unsigned long long new_int1 = binary_to_int(bitset, l);
          // Loop over all states and look for a match
          for(unsigned int i = 0; i < basis_size_; ++i){
            if(new_int1 == int_basis[i]){
              ham_mat(i, state) += t;
              break;
            }
          }
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1) % l;

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          unsigned long long new_int0 = binary_to_int(bitset, l);
          // Loop over all states and look for a match
          for(unsigned int j = 0; j < basis_size_; ++j){
            if(new_int0 == int_basis[j]){
              ham_mat(j, state) += t;
              break;
            }
          }
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }
    }
  }
}

/*******************************************************************************/
// Uses boost's library for IO
/*******************************************************************************/
void SparseHamiltonian::print_hamiltonian()
{
  std::cout << ham_mat << std::endl;
}

/*******************************************************************************/
// Returns the sign of a number
/*******************************************************************************/
double SparseHamiltonian::sign(double x)
{
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;

  return 0;
}

/*******************************************************************************/
// Computes matrix exponential using the Padé aproximation
/*******************************************************************************/
boost::numeric::ublas::matrix<double>
SparseHamiltonian::expm_pade(boost::numeric::ublas::matrix<double> mat, unsigned int p)
{
  boost::numeric::ublas::matrix<double> exp_mat(mat.size1(), mat.size2());
 
  if(mat.size1() != mat.size2()){
    std::cerr << "Pade approx error: mat.size1 != mat.size2" << std::endl; 
    exit(1);
  }
  
  unsigned int n = mat.size1();
  boost::numeric::ublas::vector<double> c(p + 1);
  c(0) = 1.0;
  for(unsigned int k = 1; k <= p; ++k)
      c(k) = c(k - 1) * 
          (static_cast<double>(p + 1 - k) / static_cast<double>(k * (2*p + 1 - k)));

  // Computation of the inf norm of the hamiltonian matrix
  double norm = 0.0;
  for(unsigned int ia = 0; ia < n; ++ia){
    double tmp = 0.0;
    for (unsigned int ja = 0; ja < n; ++ja){
      tmp += std::abs(mat(ia, ja));
    }
    norm = std::max<double>(norm, tmp);
  }

  int s = 0;
  if(norm > 0.5){
    s = std::max(0.0, floor(std::log10(s)/std::log10(2)) + 2);
    for(unsigned int ii = 0; ii < n; ++ii)
      for(unsigned int jj = 0; jj < n; ++jj)
          mat(ii,jj) = std::pow(2, -s) * mat(ii,jj);
  }

  boost::numeric::ublas::identity_matrix<double> identity(n, n);
  boost::numeric::ublas::matrix<double> mat2(n, n);
  mat2 = boost::numeric::ublas::prod(mat, mat);
  
  boost::numeric::ublas::matrix<double> q_mat(n, n);
  boost::numeric::ublas::matrix<double> p_mat(n, n);

  q_mat.assign(c(p) * identity);
  p_mat.assign(c(p - 1) * identity);

  int odd = 1;
  for(unsigned int i = p - 1; i > 0; --i){
    (odd == 1) ?
      (q_mat = boost::numeric::ublas::prod(q_mat, mat2) + c(i - 1) * identity):
      (p_mat = boost::numeric::ublas::prod(p_mat, mat2) + c(i - 1) * identity);
    odd = 1 - odd;
  }

  (odd == 1) ?
    (q_mat = boost::numeric::ublas::prod(q_mat, mat)):
    (p_mat = boost::numeric::ublas::prod(p_mat, mat));
  q_mat -= p_mat;

  // Implement matrix inversion by means of uBlas LU-decomposition backsubstitution
  boost::numeric::ublas::permutation_matrix<double> perm_mat(n);
  int res = boost::numeric::ublas::lu_factorize(q_mat, perm_mat);
  if(res != 0){
    std::cerr << "Error in the matrix inversion in the Pade approximation" << std::endl;
    exit(1);
  }

  // Use the mat2 container for identity since is not needed anymore
  mat2.assign(identity);
  boost::numeric::ublas::lu_substitute(q_mat, perm_mat, mat2);

  (odd == 1) ?
    (exp_mat.assign( -(identity + 2.0 * boost::numeric::ublas::prod(mat2, p_mat)) )):
    (exp_mat.assign(   identity + 2.0 * boost::numeric::ublas::prod(mat2, p_mat))  );

  // Final result is squared
  for(unsigned int j = 0; j < s; ++j)
    exp_mat = boost::numeric::ublas::prod(exp_mat, exp_mat);

  return exp_mat;
}

/*******************************************************************************/
// Computes matrix exponential using the Krylov method
// Relies on the computation of the matrix exponential using Padé aproximation
// for a smaller matrix
/*******************************************************************************/
void SparseHamiltonian::expv_krylov_solve(double tv, boost::numeric::ublas::vector<double> &w,
        double &err, double &hump, boost::numeric::ublas::vector<double> &v, 
            double tol, unsigned int m)
{
  double anorm = norm_inf(ham_mat);

  // Declarations
  int mxrej = 10; double btol = 1.0e-7; double err_loc;
  double gamma = 0.9; double delta = 1.2;
  unsigned int mb = m; double t_out = std::fabs(tv);
  int nstep = 0; int mx; double t_new = 0.0;
  double t_now = 0.0; double s_error = 0.0; double avnorm = 0.0;

  // Precomputed values
  const double pi = boost::math::constants::pi<double>();
  int k1 = 2; double xm = 1.0 / m; double normv = norm_2(v); double beta = normv;
  double fac = std::pow(((m+1)/std::exp(1)),(m+1)) * std::sqrt(2*pi*(m+1));
  t_new = (1.0 / anorm) * std::pow((fac * tol) / (4 * beta * anorm), xm);
  double s = std::pow(10, floor(std::log10(t_new)) - 1); 
  t_new = ceil(t_new / s) * s;
  double sgn = sign(tv);

  // Boost declarations
  boost::numeric::ublas::vector<double> p(basis_size_), av_vec(basis_size_);
  boost::numeric::ublas::compressed_matrix<double> v_m(basis_size_, m + 1, 0.0);
  boost::numeric::ublas::compressed_matrix<double> h_m(m + 2, m + 2, 0.0);
  boost::numeric::ublas::vector<double> v_m_tmp1(basis_size_, 0.0), 
                                        v_m_tmp2(basis_size_, 0.0); 
  
  w = v;
  hump = normv;
  while(t_now < t_out){
    nstep++;
    double t_step = std::min(t_out - t_now, t_new);

    // V_m and H_m need to be initialized to 0 each step
    for(unsigned int iv = 0; iv < basis_size_; ++iv)
      for(unsigned int jv = 0; jv < (m + 1); ++jv)
        v_m(iv, jv) = 0.0;

    for(unsigned int iv = 0; iv < (m + 2); ++iv)
      for(unsigned int jv = 0; jv < (m + 2); ++jv)
        h_m(iv, jv) = 0.0;

    // First step of the iterative method
    for(unsigned int iv = 0; iv < basis_size_; ++iv) 
        v_m(iv,0) = (1.0 / beta) * w(iv);

    for(unsigned int j = 0; j < m; ++j){
      p = boost::numeric::ublas::prod(ham_mat, 
              boost::numeric::ublas::column(v_m, j));

      for(unsigned int i = 0; i <= j; ++i){
        v_m_tmp1 = boost::numeric::ublas::column(v_m, i);
        h_m(i, j) = boost::numeric::ublas::inner_prod(v_m_tmp1, p);
        p = p - h_m(i, j) * boost::numeric::ublas::column(v_m, i);
      }
      
      s = norm_2(p);
      if(s < btol){
        k1 = 0;
        mb = j + 1;
        t_step = t_out - t_now;
        break;
      }
      
      h_m(j + 1, j) = s;
      for(unsigned int jv = 0; jv < basis_size_; ++jv)
        v_m(jv, j + 1) = (1.0 / s) * p(jv);
    }

    if(k1 != 0){
      h_m(m + 1, m) = 1.0;
      v_m_tmp2 = boost::numeric::ublas::column(v_m, m);
      av_vec = boost::numeric::ublas::prod(ham_mat, v_m_tmp2);
      avnorm = norm_2(av_vec);
    }

    mx = mb + k1;
    boost::numeric::ublas::matrix<double> f_copy(mx, mx, 0.0);
    unsigned int ireject = 0;
    while(ireject <= mxrej){
      boost::numeric::ublas::matrix<double> h_m_tmp(mx, mx);
      boost::numeric::ublas::matrix<double> f(mx, mx, 0.0);

      boost::numeric::ublas::range r1(0, mx);
      h_m_tmp = boost::numeric::ublas::project(h_m, r1, r1);

      f = this->expm_pade(h_m_tmp * t_step * sgn, 6);
      f_copy = f;  

      if(k1 == 0){  
        err_loc = btol;
        break;
      }
      else{
        double phi1 = std::fabs(beta * f(m, 0));
        double phi2 = std::fabs(beta * f(m+1, 0) * avnorm);  

        if(phi1 > (10 * phi2)){
          err_loc = phi2;
          xm = 1.0 / static_cast<double>(m);
        }
        else if(phi1 > phi2){
          err_loc = (phi1 * phi2) / (phi1 - phi2);
          xm = 1.0 / static_cast<double>(m);
        }
        else{
          err_loc = phi1;
          xm = 1.0 / static_cast<double>(m - 1);
        }
      }
      
      if(err_loc <= (delta * t_step * tol) ){
        break;
      }
      else{
        t_step = gamma * t_step * std::pow((t_step * tol / err_loc), xm);
        s = std::pow(10, floor(std::log10(t_step)) - 1);
        t_step = ceil(t_step / s) * s;

        if(ireject == mxrej){
          std::cerr << "Expm error: Requested tolerance is too high" << std::endl;
          exit(1);
        }

        ireject++;
      }
    } // End of inner while loop

    mx = mb + std::max(0, k1 - 1);
    boost::numeric::ublas::range r0(0, 0), r2(0, basis_size_), r3(0, mx);
    boost::numeric::ublas::compressed_matrix<double> v_m_tmp3(basis_size_, mx);
    v_m_tmp3 = boost::numeric::ublas::project(v_m , r2, r3);    
    
    w = boost::numeric::ublas::prod(v_m_tmp3, 
            beta * boost::numeric::ublas::column(f_copy, 0));

    beta = norm_2(w);
    hump = std::max(hump, beta);

    t_now = t_now + t_step;
    t_new = gamma * t_step * std::pow((t_step * tol / err_loc), xm);
    s = std::pow(10, floor(std::log10(t_new)) - 1);

    s_error = s_error + err_loc;
  } // End of outer while loop
  err = s_error;
  hump = hump / normv;
}
