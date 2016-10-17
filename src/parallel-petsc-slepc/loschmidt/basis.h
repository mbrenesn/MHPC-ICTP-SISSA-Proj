#ifndef __BASIS_H
#define __BASIS_H

#include <petscsys.h>
#include <boost/dynamic_bitset.hpp>

typedef unsigned long long int ULLInt;
typedef long long int LLInt;

class Basis
{
  private:
    unsigned int l_, n_;
    LLInt factorial_(unsigned int n);
    LLInt first_int(PetscInt nlocal, PetscInt start);

  public:
    Basis(unsigned int l, unsigned int n);
    ~Basis();
    void construct_int_basis(LLInt *int_basis, PetscInt nlocal,
        PetscInt start);
    void construct_bit_basis(boost::dynamic_bitset<> *bit_basis, 
            LLInt *int_basis, PetscInt nlocal);
    LLInt basis_size();
};
#endif
