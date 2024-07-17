# in this code u can generate EIS plot for different SOCs; you can also choose the ambient temperature
# original code by Ali Rahdarian
# optimized by axnj2 (from 2h to 17s approximately, with higher precision)
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

number_of_periods = 15
samples_per_period = 5


def returnargs(args):
    return args


def concatenate_data(impData):
    # return a 3D array with the impedance data
    # the first dimension  the frequency
    # the second dimension the SOC
    # the third dimension the temperatures

    # the input is a dictionary with the following structure
    # impedance[temperature][SOC] = np.array(impedance values) of the different frequencies

    return np.array([[impData[temp][soc] for soc in impData[temp]] for temp in impData])


def init(temp):
    global sim
    # Set up the model and parameters
    model = pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN")
    parameter_values = pybamm.ParameterValues("Prada2013")

    T_amb = 273.15 + temp  # converting to Kelvin

    parameter_values["Ambient temperature [K]"] = T_amb
    parameter_values["Initial temperature [K]"] = T_amb

    parameter_values["Current function [A]"] = current_function
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=pybamm.CasadiSolver(atol=1e-9, rtol=1e-9, mode="fast")
    )


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


def simulate_frequency_star(args):
    return simulate_frequency(*args)


if __name__ == "__main__":
    # Set the SOC
    number_of_SOC = 50
    SOC_RANGE = np.linspace(0.05, 0.95, number_of_SOC)

    N_frequencies = 50
    frequencies = np.logspace(-2, 3, N_frequencies)

    temps = np.linspace(0, 50, 10)  # Ambient temperature in degrees Celsius

    impedance = {}  # Initialize a dictionary to store impedance data for each SOC
    jj = 0

    for temp in tqdm(temps, desc="Temperatures", leave=False):

        impedance[temp] = {}
        with Pool(os.cpu_count(), initializer=init, initargs=(temp,)) as p:
            pool_output = list(tqdm(p.imap(simulate_frequency_star, itertools.product(SOC_RANGE, frequencies)),
                                    total=number_of_SOC * N_frequencies, desc="SOC and frequency pairs", leave=False))

        for k in range(number_of_SOC):
            impedance[temp][float(SOC_RANGE[k])] = np.array(pool_output[k * N_frequencies: (k + 1) * N_frequencies])
    # pprint(process_execution_time)
    # pprint(init_times)
    # # Specify the file name
    filename = 'SimulationImpedanceData_DFN_model_Prada2013_LFP_param.mat'
    # # Save the complex data to a .mat file
    impedance_matrix = concatenate_data(impedance)
    savemat(filename, {'Z': impedance_matrix, 'SOC': SOC_RANGE, 'Frequencies': frequencies, 'Temperatures': temps})
    # Plot all impedances with their corresponding labels
    plt.figure()
    cmap = plt.get_cmap('tab20')
    for i, soc in enumerate(impedance[temps[0]].keys()):
        plt.plot(impedance[temps[0]][soc].real, -impedance[temps[0]][soc].imag, marker='o',
                 label=f'SOC = {round(soc * 100)}',
                 color=cmap(i % 20))
    plt.xlabel(r"$Z_\mathrm{Re}$ [mOhm]")
    plt.ylabel(r"$-Z_\mathrm{Im}$ [mOhm]")
    plt.legend()
    # plt.show()

    # Optional: Plot the last simulation result variables
    #    sim.plot(output_variables=["Current [A]", "Terminal voltage [V]"])
