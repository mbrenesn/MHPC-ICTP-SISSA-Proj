#include <iostream>

#include "basis.h"
#include "hamiltonian.h"

int main()
{
  unsigned int l = 4;
  unsigned int n = 2;
  double V = 1.0;
  double t = -1.0;
  
  Basis basis(l,n);

  unsigned long long int *int_basis = new unsigned long long int[basis.basis_size()];
  basis.construct_int_basis(int_basis);

  std::cout << "Size of the Hilbert space:" << std::endl;
  std::cout << basis.basis_size() << std::endl;

  std::cout << "Here's the basis in int notation:" << std::endl;
  for(int i=0;i<basis.basis_size();++i) std::cout << int_basis[i] << std::endl;

  boost::dynamic_bitset<> *bit_basis = 
      new boost::dynamic_bitset<>[basis.basis_size()];
  basis.construct_bit_basis(bit_basis, int_basis);
  
  std::cout << "Here's the basis in binary notation:" << std::endl;
  for(int i=0;i<basis.basis_size();++i) std::cout << bit_basis[i] << std::endl;
 
  // Construction of the hamiltonian matrix, by calling the constructor
  // the object is the hamiltonian matrix itself
  Hamiltonian hamiltonian(basis.basis_size());

  // Call the method to construct_hamiltonian_matrix to populate the entries
  // Takes the integer basis, number of sites and particles as arguments
  hamiltonian.construct_hamiltonian_matrix(int_basis,V,t,l,n);

  // This is a 1D array representation of the matrix, [][] is overloaded so you can
  // access the elements with the [] operator, () is a 'BETTER' choice.
  for(int i=0;i<basis.basis_size();++i){
    for(int j=0;j<basis.basis_size();++j){
      std::cout << " " << hamiltonian(i,j);
    }
    std::cout << std::endl;
  }

  delete [] int_basis;
  delete [] bit_basis;
  return 0;
}
