#!/bin/bash
#PBS -q regular
#PBS -l nodes=2:ppn=20:mem160
#PBS -l walltime=4:00:00
#PBS -T flush_cache

cd $PBS_O_WORKDIR

module load openmpi/1.8.3/gnu/4.9.2

for i in 2 4 8 16 32 40
do
    echo NUM PROCS $i >> output_krylov;
    mpirun -np $i ./parallel.x >> output_krylov
done
