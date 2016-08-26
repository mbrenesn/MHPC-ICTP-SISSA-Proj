#!/bin/bash
#PBS -q wide
#PBS -l nodes=32:ppn=20
#PBS -l walltime=04:00:00
#PBS -T flush_cache

cd $PBS_O_WORKDIR

module load openmpi/1.8.3/gnu/4.9.2
module load mkl/11.1

mpirun --map-by ppr:10:socket ./32_comm_node.x -memory_view > 640p_32half_tmax100_10pprpersocket_comm_node.dat
