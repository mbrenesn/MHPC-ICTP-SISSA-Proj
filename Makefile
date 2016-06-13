# Upper directory makefile
SHELL=/bin/bash

serial:
	$(MAKE) -C src/serial-boost/
	mv src/serial-boost/serial.x .

parallel:
	$(MAKE) -C src/parallel-petsc-slepc/
	mv src/parallel-petsc-slepc/parallel.x .

clean:
	$(MAKE) -C src/serial-boost/ clean
	$(MAKE) -C src/parallel-petsc-slepc/ wipe
	rm -r *.x 
