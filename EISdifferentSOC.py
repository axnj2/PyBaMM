# in this code u can generate EIS plot for different SOCs; you can also choose the ambient temperature
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import savemat
from tqdm import tqdm

from multiprocessing import Pool

# Set up the model and parameters
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN")
parameter_values = pybamm.ParameterValues("Prada2013")
C = 50  # temperature in degrees celcius
T_amb = 273.15 + C  # converting to Kelvin
parameter_values["Ambient temperature [K]"] = T_amb
parameter_values["Initial temperature [K]"] = T_amb
frequencies = np.logspace(-2, 3, 30)
N_frequencies = len(frequencies)

# EIS parameters
I = 10 * 1e-3
number_of_periods = 100
samples_per_period = 30


def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


parameter_values["Current function [A]"] = current_function
sim = pybamm.Simulation(
    model, parameter_values=parameter_values, solver=pybamm.ScipySolver()
)

# List of SOC values

soc_values = [0.5]  # np.arange(0.05, 1.00, 0.05)
N_socs = len(soc_values)  # length of frequency vector

impedances_soc = {}  # Initialize a dictionary to store impedance data for each SOC
impedance_data = np.zeros((N_socs, N_frequencies), dtype=complex)  # data to be stored for the prject
jj = 0

# Loop over SOC values
for soc in tqdm(soc_values):
    impedances_time = []
    for frequency in tqdm(frequencies):
        # Solve
        period = 1 / frequency
        dt = period / samples_per_period
        t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
        sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency}, initial_soc=soc)
        # Extract final three periods of the solution
        time = sol["Time [s]"].entries[-3 * samples_per_period - 1:]
        current = sol["Current [A]"].entries[-3 * samples_per_period - 1:]
        voltage = sol["Terminal voltage [V]"].entries[-3 * samples_per_period - 1:]
        # FFT
        current_fft = fft(current)
        voltage_fft = fft(voltage - np.mean(voltage))
        # Get index of first harmonic
        idx = np.argmax(np.abs(current_fft))
        impedance = -voltage_fft[idx] / current_fft[idx]
        impedances_time.append(impedance)

    # Store the impedance data for this SOC
    impedances_soc[soc] = 1000 * np.array(impedances_time)
    impedance_data[jj] = 1000 * np.array(impedances_time)
    jj = jj + 1

# Specify the file name
filename = 'impedance_data_50degrees.mat'
# Save the complex data to a .mat file
savemat(filename, {'impedance_data_50degrees': impedance_data})
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
sim.plot(output_variables=["Current [A]", "Terminal voltage [V]"])
