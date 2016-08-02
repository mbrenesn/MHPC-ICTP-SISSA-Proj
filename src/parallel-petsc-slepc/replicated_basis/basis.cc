#include <cmath>
#include "basis.h"

Basis::Basis(unsigned int l, unsigned int n)
{
  l_ = l;
  n_ = n;
}

Basis::~Basis()
{}

/*******************************************************************************/
// Helper function.
/*******************************************************************************/
LLInt Basis::factorial_(unsigned int n)
{
  return (n == 1 || n == 0) ? 1 : factorial_(n - 1) * n;
}

/*******************************************************************************/
// Computes the size of the Hilbert space by means of using all possible com
// binations, this would mean using the factorial function as:
// return factorial_(l_) / ( factorial_(n_) * factorial_(l_ - n_) );
// The problem with this is that datatype overflowing can occur even for uns
// igned long long integers. Instead we can use the following expression to
// compute the size of the system.
/*******************************************************************************/
LLInt Basis::basis_size()
{
  double size = 1.0;
  for(unsigned int i = 1; i <= (l_ - n_); ++i){
    size *= (static_cast<double> (i + n_) / static_cast<double> (i));  
  }

  return floor(size + 0.5);
}

/*******************************************************************************/
// Returns the smallest possible integer that can be expressed with a given
// binary combination.
/*******************************************************************************/
LLInt Basis::first_int()
{
  LLInt first = 0;
  for(unsigned int i = 0; i < n_; ++i){
    first += 1 << i;
  }

  return first;
}

/*******************************************************************************/
// Computes next bit permutation lexicographically, so for a given value of
// smallest ineteger, computes the next lowest integer that has a different 
// bit combination. Since we know the number of combinations possible, this 
// returns an array which contains all possible combinations represented as
// integer values.
/*******************************************************************************/
void Basis::construct_int_basis(LLInt *int_basis)
{
  LLInt w;                                         // Next permutation of bits
  LLInt first = first_int();
  LLInt size = basis_size();

  int_basis[0] = first;

  for(LLInt i = 1; i < size; ++i){
    LLInt t = (first | (first - 1)) + 1;
    w = t | ((((t & -t) / (first & -first)) >> 1) - 1);
    
    int_basis[i] = w;

    first = w;
  }
}

/*******************************************************************************/
// Given an integer represented basis we know compute the basis in binary no
// tation, boost library provides best results on the long run.
// See Pieterse, et al. (2010) for a reference
/*******************************************************************************/
void Basis::construct_bit_basis(boost::dynamic_bitset<> *bit_basis, 
    LLInt *int_basis)
{
  LLInt size = basis_size();
  for(LLInt i = 0; i < size; ++i){
    boost::dynamic_bitset<> bs(l_, int_basis[i]);
    bit_basis[i] = bs;
  }
}
