#ifndef __HAMILTONIAN_H
#define __HAMILTONIAN_H

#include <iostream>

class Hamiltonian
{
  private:
    unsigned long long int basis_size_;
  public:
    Hamiltonian(unsigned long long int basis_size);
    ~Hamiltonian();
    double* hamiltonian_matrix;
    double* operator[](int i);
    unsigned long long int binary_to_int(boost::dynamic_bitset<> bs, unsigned int l);
    void construct_hamiltonian_matrix(unsigned long long int* int_basis, 
            double V, double t, unsigned int l, unsigned int n);
};
#endif
