#include <iostream>

#include <sys/time.h>
#include <sys/resource.h>

#include "basis.h"
#include "hamiltonian.h"
#include "sparse_hamiltonian.h"

double seconds()
{
    //Returns the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

int main()
{
  unsigned int l = 4;
  unsigned int n = 2;
  double V = 0.2;
  double t = -1.0;

  Basis basis(l,n);

  unsigned long long int *int_basis = new unsigned long long int[basis.basis_size()];
  basis.construct_int_basis(int_basis);

  std::cout << "Size of the Hilbert space:" << std::endl;
  std::cout << basis.basis_size() << std::endl;

  std::cout << "Here's the basis in int notation:" << std::endl;
  for(unsigned int i=0;i<basis.basis_size();++i) std::cout << int_basis[i] << std::endl;

  boost::dynamic_bitset<> *bit_basis = 
      new boost::dynamic_bitset<>[basis.basis_size()];
  basis.construct_bit_basis(bit_basis, int_basis);
  
  std::cout << "Here's the basis in binary notation:" << std::endl;
  for(unsigned int i=0;i<basis.basis_size();++i) std::cout << bit_basis[i] << std::endl;

  // Construction of the hamiltonian matrix, by calling the constructor
  // the object is the hamiltonian matrix itself
  Hamiltonian hamiltonian(basis.basis_size());

  // Call the method to construct_hamiltonian_matrix to populate the entries
  // Takes the integer basis, number of sites and particles as arguments
  hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,l,n);

  // This is a 1D array representation of the matrix, [][] is overloaded so you can
  // access the elements with the [] operator, () is a 'BETTER' choice.
  for(unsigned int i=0;i<basis.basis_size();++i){
    for(unsigned int j=0;j<basis.basis_size();++j){
      std::cout << " " << hamiltonian(i,j);
    }
    std::cout << std::endl;
  }

  // Sparse hamiltonian testing zone
  // The constructor requires an instance of the matrix container, which is a ublas
  // compressed matrix at the moment
  boost::numeric::ublas::compressed_matrix< std::complex<double> > 
      ham_mat(basis.basis_size(), basis.basis_size(), 0.0);

  SparseHamiltonian sparse_hamiltonian(basis.basis_size(), ham_mat);

  sparse_hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,l,n);

  sparse_hamiltonian.print_hamiltonian();

  boost::numeric::ublas::vector< std::complex<double> > w(basis.basis_size(), 0.0);
  boost::numeric::ublas::vector< std::complex<double> > v(basis.basis_size());

  for(unsigned int i = 0; i < basis.basis_size(); ++i) 
      v(i) = std::complex<double>(i + 1, i + 1);

  double tv = 1.0;
  double tol = 1.0e-06;
  unsigned int m = 30;
  double err, hump;

  double t1 = seconds();

  sparse_hamiltonian.expv_krylov_solve(tv,w,err,hump,v,tol,m);
  
  double t2 = seconds();

  std::cout << w << std::endl;
  std::cout << err << std::endl;
  std::cout << hump << std::endl;

  std::cout << "Time: " << t2 - t1 << " seconds" << std::endl; 
  
  delete [] int_basis;
  delete [] bit_basis;
  return 0;
}
