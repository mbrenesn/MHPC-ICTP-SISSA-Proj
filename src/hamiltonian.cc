#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include "hamiltonian.h"

Hamiltonian::Hamiltonian(unsigned long long int basis_size)
{
  basis_size_ = basis_size;
  hamiltonian_matrix = new double[basis_size * basis_size];

  //Initialization to 0.0
  for(unsigned int i = 0; i < basis_size_; ++i)
    for(unsigned int j = 0; j < basis_size_; ++j)
      hamiltonian_matrix[i * basis_size_ + j] = 0.0; 
}

Hamiltonian::~Hamiltonian()
{
  delete [] hamiltonian_matrix;
}

/*******************************************************************************/
// Operator [] overloading
/*******************************************************************************/
double* Hamiltonian::operator[](int i)
{
  if(i > this->basis_size_){
    std::cout << "Index out of bounds" << std::endl;
  }

  return hamiltonian_matrix + i * basis_size_;
}

/*******************************************************************************/
// I fear that the method .to_ulong() of dynamic bitset may overflow, let's 
// do it the slow way to begin with and look for a better option later
/*******************************************************************************/
unsigned long long int Hamiltonian::binary_to_int(boost::dynamic_bitset<> bs, unsigned int l)
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
void Hamiltonian::construct_hamiltonian_matrix(unsigned long long int* int_basis, 
        double V, double t, unsigned int l, unsigned int n)
{

  // For this Hamiltonian the elements of the diagonal are set to V
  for(unsigned int ii = 0; ii < basis_size_; ++ii)
    for(unsigned int jj = 0; jj < basis_size_; ++jj)
      if(ii == jj) hamiltonian_matrix[ii * basis_size_ + jj] = V;
    
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
              hamiltonian_matrix[i * basis_size_ + state] = t;
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
              hamiltonian_matrix[j * basis_size_ + state] = t;
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
