#include "sparse_hamiltonian.h"

SparseHamiltonian::SparseHamiltonian(unsigned int basis_size)
{
  basis_size_ = basis_size;
}

SparseHamiltonian::~SparseHamiltonian()
{}

inline
unsigned int SparseHamiltonian::binary_to_int(boost::dynamic_bitset<> bs, unsigned int l)
{
  unsigned int integer = 0;

  for(unsigned int i = 0; i < l; ++i){
    if(bs[i] == 1){
      integer += 1 << i;
    }
  }

  return integer;
}

/*******************************************************************************/
// Binary search: Divide and conquer. For the construction of the Hamiltonian
// matrix instead of looking through all the elements of the int basis a
// binary search will perform better for large systems
/*******************************************************************************/
long int SparseHamiltonian::binsearch(const unsigned long long int *array, 
    long int len, long int value)
{
  if(len == 0) return -1;
  long int mid = len / 2;

  if(array[mid] == value) 
    return mid;
  else if(array[mid] < value){
    long int result = binsearch(array + mid + 1, len - (mid + 1), value);
    if(result == -1) 
      return -1;
    else
      return result + mid + 1;
  }
  else
    return binsearch(array, mid, value);
}

/*******************************************************************************/
// Determines number of non-zero entries of the matrix 
/*******************************************************************************/
void SparseHamiltonian::determine_allocation_details(const unsigned long long int *int_basis, 
    unsigned int l, unsigned int &counter)
{

  counter = basis_size_;
  for(int state = 0; state < basis_size_; ++state){
    
    boost::dynamic_bitset<> bs(l, int_basis[state]);

    // Loop over all sites of the bit representation
    bool empty = false;
    for(unsigned int site = 0; site < l; ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1) % l;

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          empty = true;
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          unsigned long long new_int1 = binary_to_int(bitset, l);
          // Loop over all states and look for a match
          int match_ind1 = binsearch(int_basis, basis_size_, new_int1); 
          if(match_ind1 == -1){
            std::cerr << "Error in the binary search within the Ham mat alloc details" << std::endl;
            exit(1);
          } 
      
          counter++; 
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
          int match_ind0 = binsearch(int_basis, basis_size_, new_int0); 
          if(match_ind0 == -1){
            std::cerr << "Error in the binary search within the Ham mat alloc details" << std::endl;
            exit(1);
          } 
          
          counter++;
        }
        // Otherwise do nothing
        else{
          empty = true;
          continue;
        }
      }    
    }
  
    if(!empty) counter--;
  }
}

/*******************************************************************************/
// Computes the Hamiltonian matrix given by means of the integer basis
/*******************************************************************************/
void SparseHamiltonian::construct_hamiltonian_matrix(float V, float t, unsigned int l, 
    unsigned int n, const unsigned long long *int_basis) 
{
  unsigned int num_entries;
  this->determine_allocation_details(int_basis, l, num_entries);

  ham_mat_host.resize(basis_size_, basis_size_, num_entries);
  // Off-diagonal elements: the 't' terms
  // Grab 1 of the states and turn it into bit representation
  unsigned int count = 0;
  for(unsigned int state = 0; state < basis_size_; ++state){
    
    boost::dynamic_bitset<> bs(l, int_basis[state]);

    // Loop over all sites of the bit representation
    float v_term = 0.0;
    bool passed = false;
    unsigned int saved_count;
    for(unsigned int site = 0; site < l; ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1) % l;

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          // Accumulate 'V' terms
          v_term += V;
          if(!passed){
            saved_count = count;
            count++;
            passed = true;
          }
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          unsigned int new_int1 = binary_to_int(bitset, l);
          // Loop over all states and look for a match
          long int match_ind1 = binsearch(int_basis, basis_size_, new_int1); 
          if(match_ind1 == -1){
            std::cerr << "Error in the binary search within the Ham mat construction" << std::endl;
            exit(1);
          } 
          
          ham_mat_host.row_indices[count] = match_ind1;
          ham_mat_host.column_indices[count] = state;
          ham_mat_host.values[count] = VType (0.0, t);
          count++;
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1) % l;

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          unsigned int new_int0 = binary_to_int(bitset, l);
          // Loop over all states and look for a match
          long int match_ind0 = binsearch(int_basis, basis_size_, new_int0); 
          if(match_ind0 == -1){
            std::cerr << "Error in the binary search within the Ham mat construction" << std::endl;
            exit(1);
          } 
          
          ham_mat_host.row_indices[count] = match_ind0;
          ham_mat_host.column_indices[count] = state;
          ham_mat_host.values[count] = VType (0.0, t);
          count++;
        }
        // Otherwise do nothing
        else{
          continue;
        }
      } 
    }
    
    if(v_term != 0.0){
      ham_mat_host.row_indices[saved_count] = state;
      ham_mat_host.column_indices[saved_count] = state;
      ham_mat_host.values[saved_count] = VType (0.0, v_term);
    }
  }

  // This needs to be sorted for the next operations, this may be costly
  ham_mat_host.sort_by_row_and_column();

  // The Hamiltonian is now ready on host side. We move it to device for computations
  ham_mat_device.resize(basis_size_, basis_size_, num_entries);
  ham_mat_device = ham_mat_host;
}

#if 0
/*******************************************************************************/
// Returns the sign of a number
/*******************************************************************************/
double SparseHamiltonian::sign(double x)
{
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;

  return 0;
}
#endif

/*******************************************************************************/
// Computes dense matrix exponential using the Padé aproximation
/*******************************************************************************/
void SparseHamiltonian::expm_pade(cusp::array2d<VType, DSpace> exp_mat,
    cusp::array2d<VType, DSpace> mat, int n, unsigned int p, cublasHandle_t handle)
{
  cusp::array1d<float, DSpace> c(p + 1);
  c[0] = 1.0;
  for(unsigned int k = 1; k <= p; ++k)
    c[k] = c[k - 1] * 
      (static_cast<float>(p + 1 - k) / static_cast<float>(k * (2*p + 1 - k)));

  // TODO check if this can be computed using the norm_inf from blas
  // Computation of the inf norm of the dense matrix
  float norm = 0.0;
  for(unsigned int ia = 0; ia < n; ++ia){
    float tmp = 0.0;
    for (unsigned int ja = 0; ja < n; ++ja){
      VType val0 = mat.values[ia * n + ja];
      tmp += abs(val0);
    }
    norm = std::max<float>(norm, tmp);
  }

  int s = 0;
  if(norm > 0.5){
    s = std::max(0.0, floor(std::log10(norm)/std::log10(2)) + 2);
    for(unsigned int ii = 0; ii < n; ++ii)
      for(unsigned int jj = 0; jj < n; ++jj){
        VType val1 = mat.values[ii * n + jj];
        mat(ii, jj) = static_cast<float>(std::pow(2, -s)) * val1;
      }
  }

  cusp::array2d<VType, DSpace> mat2(mat.num_rows, mat.num_cols);
  cusp::blas::gemm(cusp::cuda::par.with(handle), mat, mat, mat2);

  cusp::array2d<VType, DSpace> q_mat(n, n, VType (0.0,0.0)); 
  cusp::array2d<VType, DSpace> p_mat(n, n, VType (0.0,0.0)); 

  for(unsigned int id = 0; id < n; ++id)
    for(unsigned int jd = 0; jd < n; ++jd){
      if(id == jd){
        q_mat(id, jd) = c[p];
        p_mat(id, jd) = c[p - 1];
      }
    }

  //TODO add from cusp algorithms?
  int odd = 1;
  for(unsigned int i = p - 1; i > 0; --i){
    if(odd == 1){
      cusp::blas::gemm(cusp::cuda::par.with(handle), q_mat, mat2, q_mat);
      for(unsigned int id = 0; id < n; ++id)
        for(unsigned int jd = 0; jd < n; ++jd){
          if(id == jd){
            VType val1 = q_mat.values[id * n + jd];
            float add1 = c[i - 1];
            q_mat(id, jd) = val1 + add1;
          }
        }
    }
    else{
      cusp::blas::gemm(cusp::cuda::par.with(handle), p_mat, mat2, p_mat);
      for(unsigned int id = 0; id < n; ++id)
        for(unsigned int jd = 0; jd < n; ++jd){
          if(id == jd){
            VType val2 = p_mat.values[id * n + jd];
            float add2 = c[i - 1];
            p_mat(id, jd) = val2 + add2;
          }
        }
    }
    odd = 1 - odd;
  }

  (odd == 1) ?
    (cusp::blas::gemm(cusp::cuda::par.with(handle), q_mat, mat, q_mat)):
    (cusp::blas::gemm(cusp::cuda::par.with(handle), p_mat, mat, p_mat));

  cusp::subtract(q_mat, p_mat, q_mat); 

#if 0
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
  for(int j = 0; j < s; ++j)
    exp_mat = boost::numeric::ublas::prod(exp_mat, exp_mat);
#endif
}
#if 0
/*******************************************************************************/
// Computes matrix exponential using the Krylov method
// Relies on the computation of the matrix exponential using Padé aproximation
// for a smaller matrix
/*******************************************************************************/
void SparseHamiltonian::expv_krylov_solve(double tv, 
        boost::numeric::ublas::vector< std::complex<double> > &w,
        double &err, double &hump, boost::numeric::ublas::vector< std::complex<double> > &v, 
            double tol, unsigned int m)
{
  double anorm = norm_inf(ham_mat_);
  // Declarations
  int mxrej = 10; double btol = 1.0e-7; double err_loc;
  double gamma = 0.9; double delta = 1.2;
  unsigned int mb = m; double t_out = std::fabs(tv);
  int nstep = 0; int mx; double t_new = 0.0;
  double t_now = 0.0; double s_error = 0.0; double avnorm = 0.0;
  double rndoff = anorm * 2.2204e-16;

  // Precomputed values
  const double pi = boost::math::constants::pi<double>();
  int k1 = 2; double xm = 1.0 / m; double normv = norm_2(v); double beta = normv;
  double fac = std::pow(((m+1)/std::exp(1)),(m+1)) * std::sqrt(2*pi*(m+1));
  t_new = (1.0 / anorm) * std::pow((fac * tol) / (4 * beta * anorm), xm);
  double s = std::pow(10, floor(std::log10(t_new)) - 1); 
  t_new = ceil(t_new / s) * s;
  double sgn = sign(tv);

  // Boost declarations
  boost::numeric::ublas::vector< std::complex<double> > p(basis_size_), av_vec(basis_size_);
  boost::numeric::ublas::compressed_matrix< std::complex<double> > v_m(basis_size_, m + 1, 0.0);
  boost::numeric::ublas::compressed_matrix< std::complex<double> > h_m(m + 2, m + 2, 0.0);
  boost::numeric::ublas::vector< std::complex<double> > v_m_tmp1(basis_size_, 0.0), 
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
      p = boost::numeric::ublas::prod(ham_mat_, 
              boost::numeric::ublas::column(v_m, j));

      for(unsigned int i = 0; i <= j; ++i){
        v_m_tmp1 = boost::numeric::ublas::column(v_m, i);
        h_m(i, j) = boost::numeric::ublas::inner_prod(
                boost::numeric::ublas::conj(v_m_tmp1), p);
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
      av_vec = boost::numeric::ublas::prod(ham_mat_, v_m_tmp2);
      avnorm = norm_2(av_vec);
    }

    mx = mb + k1;
    boost::numeric::ublas::matrix< std::complex<double> > f_copy(mx, mx, 0.0);
    int ireject = 0;
    while(ireject <= mxrej){
      boost::numeric::ublas::matrix< std::complex<double> > h_m_tmp(mx, mx);
      boost::numeric::ublas::matrix< std::complex<double> > f(mx, mx, 0.0);

      boost::numeric::ublas::range r1(0, mx);
      h_m_tmp = boost::numeric::ublas::project(h_m, r1, r1);

      f = this->expm_pade(h_m_tmp * t_step * sgn, 6);
      f_copy = f;  
      
      if(k1 == 0){  
        err_loc = btol;
        break;
      }
      else{
        double phi1 = std::abs(beta * f(m, 0));
        double phi2 = std::abs(beta * f(m+1, 0) * avnorm);  
        
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
    boost::numeric::ublas::compressed_matrix< std::complex<double> > v_m_tmp3(basis_size_, mx);
    boost::numeric::ublas::vector< std::complex<double> > f_column(mx);
    v_m_tmp3 = boost::numeric::ublas::project(v_m , r2, r3);    
    f_column = boost::numeric::ublas::column(f_copy, 0);
   
    w = boost::numeric::ublas::prod(v_m_tmp3, 
            beta * boost::numeric::ublas::project(f_column, r3));

    beta = norm_2(w);
    hump = std::max(hump, beta);

    t_now = t_now + t_step;
    t_new = gamma * t_step * std::pow((t_step * tol / err_loc), xm);
    s = std::pow(10, floor(std::log10(t_new)) - 1);
    t_new = ceil(t_new / s) * s;

    err_loc = std::max(err_loc, rndoff);
    s_error = s_error + err_loc;
  } // End of outer while loop
  err = s_error;
  hump = hump / normv;
}
#endif
