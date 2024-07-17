# in this code u can generate EIS plot for different SOCs; you can also choose the ambient temperature
# original code by Ali Rahdarian
import itertools
import os

import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import savemat
from tqdm import tqdm
import time

from multiprocessing import Pool

from pprint import pprint


def current_function(t):
    return I_max * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


I_max = 10 * 1e-3
C = 50  # temperature in degrees Celsius
T_amb = 273.15 + C  # converting to Kelvin

number_of_periods = 15
samples_per_period = 5


def init():
    global sim
    # Set up the model and parameters
    model = pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN")
    parameter_values = pybamm.ParameterValues("Prada2013")

    parameter_values["Ambient temperature [K]"] = T_amb
    parameter_values["Initial temperature [K]"] = T_amb

    parameter_values["Current function [A]"] = current_function
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=pybamm.CasadiSolver(atol=1e-9, rtol=1e-9, mode="fast")
    )


def return_args(soc, frequency):
    return soc, frequency


def simulate_frequency(soc, frequency):
    start = time.time()

    # Solve
    period = 1 / frequency
    dt = period / samples_per_period
    t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
    sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency}, initial_soc=soc)
    # Extract final three periods of the solution
    # t = sol["Time [s]"].entries[-10 * samples_per_period - 1:]
    current = sol["Current [A]"].entries[-10 * samples_per_period - 1:]
    voltage = sol["Terminal voltage [V]"].entries[-10 * samples_per_period - 1:]
    # FFT
    current_fft = fft(current)
    voltage_fft = fft(voltage - np.mean(voltage))
    # Get index of first harmonic
    idx = np.argmax(np.abs(current_fft))
    impedance = -voltage_fft[idx] / current_fft[idx]

    return impedance


if __name__ == "__main__":
    # Set the SOC
    number_of_SOC = 19
    SOC_RANGE = np.linspace(0.05, 0.95, number_of_SOC)

    N_frequencies = 30
    frequencies = np.logspace(-2, 3, N_frequencies)

    # List of SOC values
    soc_values = [0.5]  # np.arange(0.05, 1.00, 0.05)
    N_socs = len(soc_values)  # length of frequency vector

    impedances_soc = {}  # Initialize a dictionary to store impedance data for each SOC
    impedance_data = np.zeros((N_socs, N_frequencies), dtype=complex)  # data to be stored for the project
    jj = 0

    # Loop over SOC values
    for i in tqdm(range(1)):
        with Pool(os.cpu_count(), initializer=init) as p:
            pool_output = list(p.starmap(simulate_frequency, itertools.product(SOC_RANGE, frequencies)))

    for k in range(number_of_SOC):
        impedances_soc[float(SOC_RANGE[k])] = pool_output[k * N_frequencies: (k + 1) * N_frequencies]

    pprint(impedances_soc)

    # process_execution_time = [i[1] for i in pool_output]
    # impedences_time = [i[0] for i in pool_output]
    # init_times = [i[2] for i in pool_output]

    # Store the impedance data for this SOC
    # impedances_soc[soc] = 1000 * np.array(impedences_time)
    # impedance_data[jj] = 1000 * np.array(impedences_time)
    # jj = jj + 1

    # pprint(process_execution_time)
    # pprint(init_times)
    # # Specify the file name
    # filename = 'impedance_data_50degrees.mat'
    # # Save the complex data to a .mat file
    # savemat(filename, {'impedance_data_50degrees': impedance_data})
    # Plot all impedances with their corresponding labels
    plt.figure()
    cmap = plt.get_cmap('tab20')
    for i, (soc, data) in enumerate(impedances_soc.items()):
        plt.plot(data.real, -data.imag, marker='o', label=f'SOC = {soc}', color=cmap(i % 20))
    plt.xlabel(r"$Z_\mathrm{Re}$ [mOhm]")
    plt.ylabel(r"$-Z_\mathrm{Im}$ [mOhm]")
    plt.legend()
    plt.show()

    # Optional: Plot the last simulation result variables
    #    sim.plot(output_variables=["Current [A]", "Terminal voltage [V]"])
