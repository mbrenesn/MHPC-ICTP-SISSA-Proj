#ifndef __BASIS_H
#define __BASIS_H

#include <petscsys.h>
#include <boost/dynamic_bitset.hpp>

class Basis
{
  private:
    unsigned int l_, n_;
    PetscInt factorial_(unsigned int n);
    PetscInt first_int(PetscInt nlocal, PetscInt start, PetscInt end);

  public:
    Basis(unsigned int l, unsigned int n);
    ~Basis();
    void construct_int_basis(PetscInt *int_basis, PetscInt nlocal,
        PetscInt start, PetscInt end);
    void construct_bit_basis(boost::dynamic_bitset<> *bit_basis, 
            PetscInt *int_basis, PetscInt nlocal);
    PetscInt basis_size();
};
#endif
