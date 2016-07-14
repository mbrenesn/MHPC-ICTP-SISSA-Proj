# Upper directory makefile
SHELL=/bin/bash

serial:
	$(MAKE) -C src/serial-boost/
	mv src/serial-boost/serial.x .

replicated_basis:
	$(MAKE) -C src/parallel-petsc-slepc/replicated_basis/
	mv src/parallel-petsc-slepc/replicated_basis/replicated_basis.x .

full_dist:
	$(MAKE) -C src/parallel-petsc-slepc/full_dist/
	mv src/parallel-petsc-slepc/full_dist/full_dist.x .

clean:
	$(MAKE) -C src/serial-boost/ clean
	$(MAKE) -C src/parallel-petsc-slepc/full_dist/ clean
	$(MAKE) -C src/parallel-petsc-slepc/replicated_basis/ clean
	rm -r *.x 
