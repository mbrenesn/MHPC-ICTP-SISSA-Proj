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

master:
	$(MAKE) -C src/parallel-petsc-slepc/master_scatter/
	mv src/parallel-petsc-slepc/master_scatter/master.x .

comm:
	$(MAKE) -C src/parallel-petsc-slepc/single_comm/
	mv src/parallel-petsc-slepc/single_comm/comm.x .

comm_node:
	$(MAKE) -C src/parallel-petsc-slepc/comm_full/
	mv src/parallel-petsc-slepc/comm_full/comm_node.x .

dist_comm:
	$(MAKE) -C src/parallel-petsc-slepc/dist_comm/
	mv src/parallel-petsc-slepc/dist_comm/dist_comm.x .

node_dist:
	$(MAKE) -C src/parallel-petsc-slepc/node_dist_comm/
	mv src/parallel-petsc-slepc/node_dist_comm/node_dist.x .

loschmidt:
	$(MAKE) -C src/parallel-petsc-slepc/loschmidt/
	mv src/parallel-petsc-slepc/loschmidt/loschmidt.x .

clean:
	$(MAKE) -C src/serial-boost/ clean
	$(MAKE) -C src/parallel-petsc-slepc/loschmidt/ clean
	$(MAKE) -C src/parallel-petsc-slepc/node_dist_comm/ clean
	$(MAKE) -C src/parallel-petsc-slepc/dist_comm/ clean
	$(MAKE) -C src/parallel-petsc-slepc/comm_full/ clean
	$(MAKE) -C src/parallel-petsc-slepc/single_comm/ clean
	$(MAKE) -C src/parallel-petsc-slepc/master_scatter/ clean
	$(MAKE) -C src/parallel-petsc-slepc/full_dist/ clean
	$(MAKE) -C src/parallel-petsc-slepc/replicated_basis/ clean
	rm -r *.x 
