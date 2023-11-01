from SPDC_solver import SPDC_solver
from solver import crystal_prop
from time import process_time

n_sample = 1000
# For n_sample = 1000
# Took 368.86957 seconds
# Took in avarge 0.36887 seconds for sample

spdc = SPDC_solver(N=1)


start_time = process_time()

for n in range(n_sample):
    crystal_prop(
    pump_profile = spdc.pump_profile, 
    pump = spdc.pump,
    signal_field = spdc.signal_field, 
    idler_field = spdc.idler_field,
    vacuum_states = spdc.vacuum_states,
    chi2 = spdc.chi2, 
    N = spdc.N,
    shape = spdc.shape,
    print_err = spdc.print_err,
    return_err = spdc.return_err,
    data = spdc.fields
    ) 

end_time = process_time()

duration_time = end_time - start_time
print(f'Took {duration_time:.5f} seconds')
avarge_time = duration_time / n_sample
print(f'Took in avarge {avarge_time:.5f} seconds for sample')