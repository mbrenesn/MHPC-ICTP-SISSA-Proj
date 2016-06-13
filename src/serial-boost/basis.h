#ifndef __BASIS_H
#define __BASIS_H

#include <boost/dynamic_bitset.hpp>

class Basis
{
  private:
    unsigned int l_, n_;
    unsigned long long int factorial_(unsigned int n);
    unsigned long long int smallest_int();

  public:
    Basis(unsigned int l, unsigned int n);
    ~Basis();
    void construct_int_basis(unsigned long long int *int_basis);
    void construct_bit_basis(boost::dynamic_bitset<> *bit_basis, 
            unsigned long long int* int_basis);
    unsigned long long int basis_size();
};
#endif
