#!/usr/bin/env python3
# vim:set fdm=indent:
"""
Exercise 1
"""
from decimal import Decimal
import sys
import numpy as np
import matplotlib.pyplot as plt

###################
#  CLASS OBJECTS  #
###################


class Jumper:  # pylint: disable=too-few-public-methods
    """Jumper is the body freefalling"""

    __slots__ = ("cross_sec_area", "drag_coeff", "mass", "starting_height")

    def __init__(self, drag_coeff: float, starting_height: float, mass: float):
        """Properties of the body

        :cross_sec_area: of body
        :drag_coeff: of body
        :mass: of body
        :starting_height: of body

        """
        self.cross_sec_area = 0.4
        self.drag_coeff = drag_coeff
        self.mass = mass
        self.starting_height = starting_height


class StringWave:
    """StringWave holds material properties of the string and also its Gaussian wavepacket"""

    __slots__ = ("std", "length_density", "tension", "phase_velocity", "a_0", "length")

    def __init__(self, tension: float, length_density: float, length: float):
        """Properties of the wave

        :std: standard deviation
        :tension: of string
        :length density: of string
        :a_0: amplitude of Gaussian
        :length: of string

        """
        self.std = 1
        self.tension = tension
        self.length_density = length_density
        self.phase_velocity = self.get_phase_velocity()
        self.a_0 = self.get_a_0()
        self.length = length

    def get_phase_velocity(self) -> float:
        """Calculate phase velocity of Gaussian

        :returns: v as float

        """
        return np.sqrt(self.tension / self.length_density)

    def get_a_0(self) -> float:
        """Calculate A_0 of Gaussian to normalise it

        :returns: a_0 as float

        """
        return 1 / (self.std * np.sqrt(2 * np.pi))


#################################
#  OPTION INITIATION FUNCTIONS  #
#################################


def run_option_a(
    jumper: Jumper, global_vars: dict, is_plot: bool, is_verbose: bool
) -> tuple:
    """Execute part a

    :param jumper: from Jumper class
    :param global_vars: global variables
    :param is_plot: toggles plotting
    :param is_verbose: toggles verbosity
    :returns: tuple of arrays

    """
    t_vals, y_vals, vy_vals = calculate_analytical_predictions(
        jumper, global_vars, is_verbose
    )
    t_vals, y_vals, vy_vals = cut_negative_distance(
        t_vals, y_vals, vy_vals, determine_negative_distance(y_vals, global_vars)
    )
    if is_verbose:
        print(f"Duration of fall is {round_with_decimal(3,t_vals[-1])} s.")
    if is_plot:
        _ = input("Press anything to show plot...")
        plot_part_a_b_c(t_vals, y_vals, vy_vals)
    return t_vals, y_vals, vy_vals


def run_option_b_c(
    jumper: Jumper,
    global_vars: dict,
    is_constant_drag: bool,
    is_plot: bool,
    is_verbose: bool,
) -> tuple:
    """Execute part b or c

    :param jumper: from Jumper class
    :param global_vars: global variables
    :param is_constant_drag: toggles drag varying
    :param is_plot: toggles plotting
    :param is_verbose: toggles verbosity
    :returns: tuple of arrays t, y, v

    """
    t_vals, y_vals, vy_vals = calculate_numerical_predictions(
        jumper, global_vars, is_constant_drag, is_verbose
    )
    t_vals, y_vals, vy_vals = cut_negative_distance(
        t_vals, y_vals, vy_vals, determine_negative_distance(y_vals, global_vars)
    )
    if is_verbose:
        print(f"Duration of fall is {round_with_decimal(3,t_vals[-1])} s.")
    if is_plot:
        _ = input("Press anything to show plot...")
        plot_part_a_b_c(t_vals, y_vals, vy_vals)
        return _, _, _
    return t_vals, y_vals, vy_vals


def run_option_b_extra(jumper: Jumper, global_vars: dict):
    """Run additional things for b for report

    :param jumper: from Jumper class
    :param global_vars: global variables

    """
    vary_timestep(jumper, global_vars)
    vary_mass(jumper, global_vars)


def run_option_d(jumper: Jumper, global_vars: dict):
    """Execute option d

    :param jumper: from Jumper class
    :param global_vars: global variables

    """
    t_vals, y_vals, vy_vals = run_option_b_c(jumper, global_vars, False, False, True)
    mach_ratio, vy_vals_abs = calculate_mach_ratios(y_vals, vy_vals, global_vars)
    print("The maximum Mach number is", round_with_decimal(3, np.max(mach_ratio)))
    print("The maximum speed is", round_with_decimal(3, np.max(vy_vals_abs)))
    print("This occurs at", round_with_decimal(3, t_vals[np.argmax(vy_vals_abs)]), "s")
    _ = input("Press anything to show plot...")
    plot_mach(mach_ratio, t_vals)
    print("Investigating jump height")
    _ = input("Press anything to show plot...")
    vary_jump_height(jumper, global_vars)
    print("Investigating drag coefficient")
    _ = input("Press anything to show plot...")
    vary_drag_coeff(jumper, global_vars)


def run_option_e(global_vars: dict, wavepacket: StringWave):
    """Execute option e

    :param global_vars: global variables
    :param wavepacket: from StringWave class

    """
    dataset, x_vals, t_vals = fill_dataset(global_vars, wavepacket)
    plot_waves(dataset, t_vals, x_vals, wavepacket)


def run_option_f(global_vars: dict, wavepacket: StringWave):
    """Vary gamma and examine effects on the wave solution
    Keeping timestep tau and lattice spacing the same, which means
    phase velocity has to change, so tension T changes, to keep
    length density the same

    :global_vars: global vars
    :wavepacket: from StringWave class

    """
    dataset_actual, _, _ = fill_dataset(global_vars, wavepacket)
    gammas = np.concatenate(
        (
            np.linspace(0.1, 1.0, 40),
            np.linspace(1.1, 1.12, 40),
            np.linspace(1.3, 1.32, 40),
            np.linspace(1.5, 1.52, 40),
        ),
        axis=None,
    )
    phase_velocities = np.sqrt(
        ((global_vars["lattice spacing"]) ** 2)
        * gammas
        / (global_vars["timestep2"] ** 2)
    )
    std_solution = np.zeros_like(gammas)
    for idx, _ in enumerate(gammas):
        wavepacket.phase_velocity = phase_velocities[idx]
        dataset_approx, _, _ = fill_dataset(global_vars, wavepacket)
        difference = dataset_approx - dataset_actual
        std_solution[idx] = np.average(np.std(difference, axis=1))
    plot_gammas_std(gammas, std_solution)


##############################
#  PROBLEM 1 MAIN FUNCTIONS  #
##############################


def vary_timestep(jumper: Jumper, global_vars: dict):
    """Vary timestep of Euler method

    :param jumper: from Jumper class
    :param global_vars: global variables

    """
    print(
        "Now varying timestep to",
        "calculate standard deviation of the difference between",
        "numerical method and analytical method",
    )
    timestep = np.concatenate(
        (
            np.linspace(0.1, 1.0, 100),
            np.linspace(1.01, 3.0, 50),
            np.linspace(3.01, 5.0, 30),
        ),
        axis=None,
    )
    std_data_y = np.ones_like(timestep)
    std_data_vy = np.ones_like(timestep)
    for idx, val in enumerate(timestep):
        global_vars["timestep"] = val
        t_vals_num, y_vals_num, vy_vals_num = run_option_b_c(
            jumper, global_vars, True, False, False
        )
        t_vals_ana, y_vals_ana, vy_vals_ana = run_option_a(
            jumper, global_vars, False, False
        )
        stop_idx = np.min([np.size(t_vals_num), np.size(t_vals_ana)])
        std_data_y[idx] = np.std(y_vals_ana[:stop_idx] - y_vals_num[:stop_idx])
        std_data_vy[idx] = np.std(vy_vals_ana[:stop_idx] - vy_vals_num[:stop_idx])
    _ = input("Press anything to show plot...")
    plot_part_b_timestep(timestep, std_data_y, std_data_vy)


def vary_mass(jumper: Jumper, global_vars: dict):
    """Vary mass to vary k/m

    :param jumper: from Jumper class
    :param global_vars: global variables

    """
    print(
        "Now varying mass to",
        "examine effect on max speed",
        "and duration of fall",
    )
    global_vars["timestep"] = 0.01
    masses = np.linspace(50, 80, 100)
    duration, max_speed = np.zeros_like(masses), np.zeros_like(masses)
    for idx, val in enumerate(masses):
        jumper.mass = val
        t_vals, _, vy_vals = run_option_b_c(jumper, global_vars, True, False, False)
        duration[idx] = t_vals[-1]
        max_speed[idx] = np.max(np.absolute(vy_vals))
    _ = input("Press anything to show plot...")
    plot_part_b_mass(masses, duration, max_speed)


def cut_negative_distance(
    t_vals: np.ndarray, y_vals: np.ndarray, vy_vals: np.ndarray, zero_idx: int
) -> tuple:
    """Return val[:zero_idx]

    :param t_vals: t array
    :param y_vals: y array
    :param vy_vals: v array
    :param zero_idx: given index
    :returns: tuple of values at given index

    """
    return t_vals[:zero_idx], y_vals[:zero_idx], vy_vals[:zero_idx]


def determine_negative_distance(y_vals: np.ndarray, global_vars: dict) -> int:
    """Determine the index where the object hits the ground

    :param y_vals: y array
    :param global_vars: global vars
    :returns: first index where value is 0

    """
    return np.where(y_vals < global_vars["tolerance"])[0][0]


def calculate_mach_ratios(
    y_vals: np.ndarray, vy_vals: np.ndarray, global_vars: dict
) -> tuple:
    """Generate main arrays for part d

    :param vy_vals_abs: absolute v values
    :returns: tuple of mach ratio and absolute v

    """
    vsound_vals = calculate_sound_values(y_vals, global_vars)
    vy_vals_abs = np.abs(vy_vals)
    mach_ratio = np.divide(vy_vals_abs, vsound_vals)
    return mach_ratio, vy_vals_abs


def vary_jump_height(jumper: Jumper, global_vars: dict):
    """Vary jump height vs Mach number

    :param jumper: from Jumper class
    :param global_vars: global variables

    """
    y_0 = np.linspace(1000, 40000, 300)
    max_mach = np.full_like(y_0, 0)
    max_speed = np.full_like(y_0, 0)
    duration = np.full_like(y_0, 0)
    for idx, val in enumerate(y_0):
        jumper.starting_height = val
        t_vals, y_vals, vy_vals = run_option_b_c(
            jumper, global_vars, False, False, False
        )
        mach_ratio, vy_vals_abs = calculate_mach_ratios(y_vals, vy_vals, global_vars)
        max_speed[idx] = np.max(vy_vals_abs)
        max_mach[idx] = np.max(mach_ratio)
        duration[idx] = t_vals[-1]
    plot_jump_height(y_0, max_mach, max_speed, duration)


def vary_drag_coeff(jumper: Jumper, global_vars: dict):
    """Vary drag coeff vs Mach number

    :param jumper: from Jumper class
    :param global_vars: global variables

    """
    drag_coeff = np.linspace(0.05, 2.0, 300)
    max_mach = np.full_like(drag_coeff, 0)
    max_speed = np.full_like(drag_coeff, 0)
    duration = np.full_like(drag_coeff, 0)
    for idx, val in enumerate(drag_coeff):
        jumper.drag_coeff = val
        t_vals, y_vals, vy_vals = run_option_b_c(
            jumper, global_vars, False, False, False
        )
        mach_ratio, vy_vals_abs = calculate_mach_ratios(y_vals, vy_vals, global_vars)
        max_speed[idx] = np.max(vy_vals_abs)
        max_mach[idx] = np.max(mach_ratio)
        duration[idx] = t_vals[-1]
    plot_drag_coeff(drag_coeff, max_mach, max_speed, duration)


def calculate_sound_values(y_vals: np.ndarray, global_vars: dict) -> np.ndarray:
    """Generate sound array using temperature array

    :param vy_vals: vy array
    :param y_vals: y varray
    :returns: array for vsound values

    """
    temp_conditions = [
        y_vals <= sys.float_info.epsilon + 11000,
        (11000 - sys.float_info.epsilon < y_vals)
        & (y_vals <= sys.float_info.epsilon + 25100),
    ]
    temp_choices = [288.0 - 0.0065 * y_vals, 216.5]
    temp_vals = np.select(temp_conditions, temp_choices, 141.3 + 0.003 * y_vals)
    return np.sqrt(
        global_vars["gamma"]
        * global_vars["molar_gas_constant"]
        * temp_vals
        / global_vars["molar_gas_mass"]
    )


def calculate_drag_factor(y_val: float, jumper: Jumper, global_vars: dict) -> float:
    """Find the drag factor which varies by altitude

    :param y_val: height
    :param jumper: from Jumper class
    :param global_vars: global variables
    :returns: drag factor (float)

    """
    rho = global_vars["rho_0"] * np.exp((-y_val) / global_vars["scale_height"])
    return jumper.drag_coeff * rho * jumper.cross_sec_area / 2


def calculate_analytical_predictions(
    jumper: Jumper, global_vars: dict, is_verbose: bool
) -> tuple:
    """Evaluate the values of y & v_y for given t

    :param t_vals: time array
    :param y_vals: y array
    :param vy_vals: vy array
    :returns: the results after evaluation

    """
    t_vals = np.arange(
        global_vars["t_min"], global_vars["t_max"], global_vars["timestep"]
    )
    y_vals, vy_vals = (
        np.zeros(global_vars["points"]),
        np.zeros(global_vars["points"]),
    )
    drag_factor = jumper.drag_coeff * global_vars["rho_0"] * jumper.cross_sec_area / 2
    if is_verbose:
        print_header(jumper, drag_factor, 0, True, "analytical")
    y_vals = jumper.starting_height - (
        jumper.mass
        * np.log(
            np.cosh(np.sqrt(drag_factor * global_vars["grav"] / jumper.mass) * t_vals)
        )
        / drag_factor
    )
    vy_vals = -np.sqrt(jumper.mass * global_vars["grav"] / drag_factor) * np.tanh(
        np.sqrt(drag_factor * global_vars["grav"] / jumper.mass) * t_vals
    )
    return t_vals, y_vals, vy_vals


def calculate_numerical_predictions(
    jumper: Jumper, global_vars: dict, is_constant_drag: bool, is_verbose: bool
) -> tuple:
    """Evaluate the values of y & v_y for given t

    :param t_vals: time array
    :param y_vals: y array
    :param vy_vals: vy array
    :returns: the results after evaluation

    """
    t_vals = np.arange(0, global_vars["points"]) * global_vars["timestep"]
    y_vals, vy_vals = (
        np.zeros(global_vars["points"]),
        np.zeros(global_vars["points"]),
    )
    drag_factor = jumper.drag_coeff * global_vars["rho_0"] * jumper.cross_sec_area / 2
    if is_verbose:
        print_header(
            jumper, drag_factor, global_vars["timestep"], is_constant_drag, "numerical"
        )
    y_vals[0] = jumper.starting_height
    for idx, _ in enumerate(t_vals[:-1]):
        if not is_constant_drag:
            drag_factor = calculate_drag_factor(
                y_vals[idx],
                jumper,
                global_vars,
            )
        vy_vals[idx + 1] = vy_vals[idx] - global_vars["timestep"] * (
            global_vars["grav"]
            + ((drag_factor / jumper.mass) * np.abs(vy_vals[idx]) * vy_vals[idx])
        )
        y_vals[idx + 1] = y_vals[idx] + global_vars["timestep"] * vy_vals[idx]
    return t_vals, y_vals, vy_vals


##############################
#  PROBLEM 2 MAIN FUNCTIONS  #
##############################


def fill_dataset(global_vars: dict, wavepacket: StringWave) -> tuple:
    """Fill the 2d array with solutions

    :param global_vars: global variables
    :param wavepacket: from StringWave class
    :returns: tuple of arrays

    """
    gamma = (
        global_vars["timestep2"]
        * wavepacket.phase_velocity
        / global_vars["lattice spacing"]
    ) ** 2
    x_vals = np.arange(
        -global_vars["lattice spacing"],
        wavepacket.length + 2 * global_vars["lattice spacing"],
        global_vars["lattice spacing"],
    )
    t_vals = np.arange(
        0,
        global_vars["T"] + global_vars["timestep2"],
        global_vars["timestep2"],
    )
    dataset = np.zeros((np.size(t_vals), np.size(x_vals)), dtype="float64")
    dataset = calculate_initial_timestep(
        global_vars, wavepacket, x_vals, t_vals, dataset
    )
    for time in t_vals[2:]:
        time_idx = map_float_to_index(time, global_vars["timestep2"], t_vals[0])
        dataset[time_idx] = calculate_next_timestep(
            (time_idx, gamma), global_vars, wavepacket, dataset, x_vals
        )

    return dataset, x_vals, t_vals


def calculate_initial_timestep(
    global_vars: dict,
    wavepacket: StringWave,
    x_vals: np.ndarray,
    t_vals: np.ndarray,
    dataset: np.ndarray,
) -> np.ndarray:
    """Generate initial timesteps t_0 and t_1

    :param position: the position x
    :param time: the time t
    :param global_vars: global variables
    :param wavepacket: from StringWave class
    :returns: the dataset array now with initial conditions

    """
    dataset[0] = np.where(
        (
            (x_vals < global_vars["tolerance"])
            | (np.abs(x_vals - wavepacket.length) < global_vars["tolerance"])
        ),
        dataset[0],
        evaluate_initial_displacement(x_vals, t_vals[0], wavepacket),
    )
    dataset[1] = np.where(
        (
            (x_vals < global_vars["tolerance"])
            | (np.abs(x_vals - wavepacket.length) < global_vars["tolerance"])
        ),
        dataset[1],
        (1 / 2)
        * (
            evaluate_initial_displacement(
                x_vals + global_vars["lattice spacing"], t_vals[0], wavepacket
            )
            + evaluate_initial_displacement(
                x_vals - global_vars["lattice spacing"], t_vals[0], wavepacket
            )
        )
        + global_vars["timestep2"]
        * evaluate_initial_time_derivative(x_vals, t_vals[0], wavepacket),
    )
    return dataset


def calculate_next_timestep(
    information: tuple,
    global_vars: dict,
    wavepacket: StringWave,
    dataset: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Generate solution for given position and time

    :param information: tuple of (position, time, gamma)
    :param global_vars: global variables
    :param wavepacket: from StringWave class
    :param dataset: the main 2d array
    :returns: the solution at that point in space/time

    """
    time_idx, gamma = information
    dataset_shift_bw = np.roll(dataset[time_idx - 1], -1)
    dataset_shift_fw = np.roll(dataset[time_idx - 1], 1)
    return np.where(
        (
            (positions <= global_vars["tolerance"])
            | (positions >= global_vars["tolerance"] + wavepacket.length)
        ),
        0.0,
        (
            gamma * (dataset_shift_fw + dataset_shift_bw)
            + (2 * (1 - gamma) * dataset[time_idx])
            - dataset[time_idx - 2]
        ),
    )


def evaluate_initial_displacement(
    positions: np.ndarray, time: float, wavepacket: StringWave
) -> np.ndarray:
    """Evaluate and return the Gaussian at (x,t)

    :param position: the position x
    :param time: the time t
    :param wavepacket: from StringWave class
    :returns: the Gaussian displacement in y

    """
    return wavepacket.a_0 * np.exp(
        -(((positions - wavepacket.length / 2) - wavepacket.phase_velocity * time) ** 2)
        / (wavepacket.std) ** 2
    )


def evaluate_initial_time_derivative(
    positions: np.ndarray, time: float, wavepacket: StringWave
) -> np.ndarray:
    """Evaluate and return the derivative of the Gaussian at (x,t)

    :param position: the position x
    :param time: the time t
    :param wavepacket: from StringWave class
    :returns: the derivative Gaussian displacement in y

    """
    return (
        2
        * wavepacket.a_0
        * wavepacket.phase_velocity
        * ((positions - wavepacket.length / 2) - wavepacket.phase_velocity * time)
        * np.exp(
            -(
                ((positions - wavepacket.length / 2) - wavepacket.phase_velocity * time)
                ** 2
            )
            / (wavepacket.std) ** 2
        )
        / (wavepacket.std) ** 2
    )


######################
#  HELPER FUNCTIONS  #
######################


def print_header(
    jumper: Jumper,
    drag_factor: float,
    timestep: float,
    is_constant_drag: bool,
    solver_type: str,
):
    """Print a verbose message

    :param jumper: from Jumper class
    :param drag_factor: drag factor
    :param timestep: timestep
    :param is_constant_drag: toggles constant drag message
    :param solver_type: either numerical/analytical

    """
    print(
        f"Calculating {solver_type} predictions for freefall with drag. Using:",
        f"\n\tmass m = {jumper.mass} kg,",
        f"\n\ty_0 = {jumper.starting_height} m,",
    )
    if solver_type == "numerical":
        print(f"\ttimestep = {timestep}")
    if is_constant_drag:
        print(
            f"\tdrag factor k = {drag_factor} kg/m,",
        )
    else:
        print(
            "\tand varying air density",
        )


def round_with_decimal(decimal_places: int, value: float) -> float:
    """Round a float to the nearest dp provided without precision error
    using quantize() from Decimal class

    :param dp: number of decimal places
    :param value: the float to round
    :returns: the answer as a float

    """
    reference = "1." + "0" * decimal_places
    return float(Decimal(str(value)).quantize(Decimal(reference)))


def map_float_to_index(value: float, step: float, start: float) -> int:
    """Given the value convert it to an index

    :param value: the original value
    :param step: the stepsize
    :param returns: the index

    """
    return int(round_with_decimal(0, (value - start) / step))


#######################
#  PLOTTING FUNCTION  #
#######################


def plot_waves(
    dataset: np.ndarray, t_vals: np.ndarray, x_vals: np.ndarray, wavepacket: StringWave
):
    """Plot multiple scatter plots of the wave for each timestep and save to OS

    :param dataset: 2d array of wave solutions
    :param t_vals: t array
    :param x_vals: x array

    """
    min_y, max_y = np.min(dataset) * 1.2, np.max(dataset) * 1.2
    for idx in range(np.size(t_vals)):
        curr_wave = dataset[idx]
        plt.clf()
        plt.ylim([min_y, max_y])
        plt.xlim([0, wavepacket.length])
        plt.xlabel("x displacement (m)")
        plt.ylabel("y displacement (m)")
        plt.scatter(x_vals, curr_wave)
        plt.savefig(f"Figure {idx}.png", format="png", dpi=150, bbox_inches="tight")
    print("Plots of the wave on a string saved for each timestep as Figure X.png.")


def plot_part_a_b_c(t_vals: np.ndarray, y_vals: np.ndarray, vy_vals: np.ndarray):
    """Plot using matplotlib the arrays

    :param t_vals: time array
    :param y_vals: y array
    :param vy_vals: vy array

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Height and speed data for Newtonian freefall with drag")
    ax1.set(xlabel="Time (s)", ylabel="Height (m)", title="Altitude")
    ax2.set(xlabel="Time (s)", ylabel=r"Speed ($ms^{-1}$)", title="Vertical speed")
    ax1.scatter(t_vals, y_vals, s=2, c="r")
    ax2.scatter(t_vals, vy_vals, s=2, c="g")
    plt.show()


def plot_mach(
    mach_ratio: np.ndarray,
    t_vals: np.ndarray,
):
    """Plot mach ratio

    :param mach_ratio: array of mach ratio by time
    :param t_vals: t array

    """
    plt.title("Mach ratio for Newtonian freefall with drag")
    plt.xlabel("Time (s)")
    plt.ylabel("Mach ratio")
    plt.scatter(t_vals, mach_ratio, s=2, c="r")
    plt.show()


def plot_jump_height(
    y_0: np.ndarray, max_mach: np.ndarray, max_speed: np.ndarray, duration: np.ndarray
):
    """Plot jump height vs mach number

    :param y_0: starting height array
    :param max_mach: max mach number array
    :param max_speed: max speed array
    :param duration: duration of fall array

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle("Investigating varying dropping heights")
    ax1.set(
        xlabel="Height (m)",
        ylabel="Mach ratio",
        title="Maximum Mach ratio",
    )
    ax2.set(
        xlabel="Height (m)",
        ylabel=r"Speed ($ms^{-1}$)",
        title="Maximum speed",
    )
    ax3.set(
        xlabel="Height (m)",
        ylabel="Time (s)",
        title="Duration of freefall",
    )
    ax1.scatter(y_0, max_mach, s=2, c="r")
    ax2.scatter(y_0, max_speed, s=2, c="g")
    ax3.scatter(y_0, duration, s=2, c="b")
    plt.show()


def plot_drag_coeff(
    drag_coeffs: np.ndarray,
    max_mach: np.ndarray,
    max_speed: np.ndarray,
    duration: np.ndarray,
):
    """Plot drag coeff vs mach number

    :param drag_coeffs: drag coefficients array
    :param max_mach: max mach number array
    :param max_speed: max speed array
    :param duration: duration of fall array

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle("Investigating varying drag coefficients")
    ax1.set(
        xlabel="Drag coefficient",
        ylabel="Mach ratio",
        title="Maximum Mach ratio",
    )
    ax2.set(
        xlabel="Drag coefficient",
        ylabel=r"Speed ($ms^{-1}$)",
        title="Maximum speed",
    )
    ax3.set(
        xlabel="Drag coefficient",
        ylabel="Time (s)",
        title="Duration of freefall",
    )
    ax1.scatter(drag_coeffs, max_mach, s=2, c="r")
    ax2.scatter(drag_coeffs, max_speed, s=2, c="g")
    ax3.scatter(drag_coeffs, duration, s=2, c="b")
    plt.show()


def plot_part_b_timestep(
    timestep: np.ndarray, std_data_y: np.ndarray, std_data_vy: np.ndarray
):
    """Plot timestep vs std of difference

    :param timestep: timestep array
    :param std_data_y: std in y array
    :param std_data_vy: std in vy array

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Standard deviation of difference between actual and predicted trajectory"
    )
    ax1.set(
        xlabel="Timestep (s)",
        ylabel="Standard deviation in height difference (m)",
        title="Altitude",
    )
    ax2.set(
        xlabel="Timestep (s)",
        ylabel=r"Standard deviation in speed difference ($ms^{-1}$)",
        title="Vertical speed",
    )
    ax1.scatter(timestep, std_data_y, s=2, c="r")
    ax2.scatter(timestep, std_data_vy, s=2, c="g")
    plt.show()


def plot_part_b_mass(masses: np.ndarray, duration: np.ndarray, max_speed: np.ndarray):
    """Plot timestep vs std of difference

    :param masses: masses array
    :param duration: duration array
    :param max speed: max speed array

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Effect of mass on duration and max speed of freefall")
    ax1.set(xlabel="Mass (kg)", ylabel="Duration (s)", title="Duration")
    ax2.set(xlabel="MAss (kg)", ylabel=r"Max speed ($ms^{-1}$)", title="Vertical speed")
    ax1.scatter(masses, duration, s=2, c="r")
    ax2.scatter(masses, max_speed, s=2, c="g")
    plt.show()


def plot_gammas_std(gammas: np.ndarray, std_solution: np.ndarray):
    """Plot gammas versus standard deviation

    :param gammas: gamma values
    :param std_solution: std values

    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle("Effect of gamma on the standard deviation of the solution")
    ax1.set(
        xlabel=r"$\gamma$",
        ylabel="Standard deviation in height (m)",
        title=r"$0.1 \leq \gamma \leq 1.0$",
    )
    ax2.set(
        xlabel=r"$\gamma$",
        ylabel="Standard deviation in height (m)",
        title=r"$1.1 \leq \gamma \leq 1.12$",
    )
    ax3.set(
        xlabel=r"$\gamma$",
        ylabel="Standard deviation in height (m)",
        title=r"$1.3 \leq \gamma \leq 1.32$",
    )
    ax4.set(
        xlabel=r"$\gamma$",
        ylabel="Standard deviation in height (m)",
        title=r"$1.5 \leq \gamma \leq 1.52$",
    )
    ax1.scatter(gammas[:40], std_solution[:40], s=2, c="r")
    ax2.scatter(gammas[40:80], std_solution[40:80], s=2, c="r")
    ax3.scatter(gammas[80:120], std_solution[80:120], s=2, c="r")
    ax4.scatter(gammas[120:], std_solution[120:], s=2, c="r")
    plt.show()


def main():
    """Driver code"""
    user_input = "0"
    global_vars = {
        "t_min": 0,
        "t_max": 30,
        "rho_0": 1.2,
        "grav": 9.81,
        "timestep": 0.1,
        "scale_height": 7640,
        "points": 4000,
        "gamma": 1.4,
        "molar_gas_constant": 8.314462,
        "molar_gas_mass": 0.0289645,
        "lattice spacing": 0.2,
        "timestep2": 0.05,
        "T": 6.5,
        "tolerance": 1e-14,
    }
    default_jumper = Jumper(1.0, 1000, 70)
    baumgartner = Jumper(1.0, 39045, 70)
    default_wave = StringWave(8, 0.5, 12)
    plt.rcParams.update({"font.size": 22})
    while user_input != "q":
        user_input = input(
            'Enter a choice, "a", "b", "c", "d", "e", "f", "g", "h" for help, or "q" to quit: '
        )
        print("You entered the choice: ", user_input)
        print(f"You have chosen part ({user_input})")
        if user_input == "a":
            run_option_a(default_jumper, global_vars, True, True)
        elif user_input == "b":
            run_option_b_c(default_jumper, global_vars, True, True, True)
            run_option_b_extra(default_jumper, global_vars)
        elif user_input == "c":
            run_option_b_c(baumgartner, global_vars, False, True, True)
        elif user_input == "d":
            run_option_d(baumgartner, global_vars)
        elif user_input == "e":
            run_option_e(global_vars, default_wave)
        elif user_input == "f":
            run_option_f(global_vars, default_wave)
        # elif user_input == "h":
        #     run_option_h()
        elif user_input != "q":
            print("This is not a valid choice.")
    print("You have chosen to finish - goodbye.")


if __name__ == "__main__":
    main()
