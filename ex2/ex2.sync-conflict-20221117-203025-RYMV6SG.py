#!/usr/bin/env python3
# vim:set fdm=indent:
"""
Exercise 1
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


class Jumper:  # pylint: disable=too-few-public-methods
    """Docstring for  Jumper."""

    __slots__ = ("cross_sec_area", "drag_coeff", "mass", "starting_height")

    def __init__(self, drag_coeff, starting_height):
        """TODO: to be defined."""
        self.cross_sec_area = 0.4  # falling flat
        self.drag_coeff = drag_coeff
        self.mass = 70
        self.starting_height = starting_height


class StringWave:
    """Docstring for Wavepacket."""

    __slots__ = ("std", "length_density", "tension", "phase_velocity", "a_0", "length")

    def __init__(self, tension, length_density, length):
        """TODO: to be defined."""
        self.std = 1
        self.tension = tension
        self.length_density = length_density
        self.phase_velocity = self.get_phase_velocity()
        self.a_0 = self.get_a_0()
        self.length = length

    def get_phase_velocity(self):
        """doc"""
        return np.sqrt(self.tension / self.length_density)

    def get_a_0(self):
        """doc"""
        return 1 / (self.std * np.sqrt(2 * np.pi))


def run_option_a(jumper: Jumper, global_vars: dict):
    """Execute part a"""
    t_vals, y_vals, vy_vals = calculate_analytical_predictions(jumper, global_vars)
    _ = input("Press anything to show plot...")
    plot_part_a_b_c(t_vals, y_vals, vy_vals)


def run_option_b_c(
    jumper: Jumper,
    global_vars: dict,
    is_constant_drag: bool,
    plot: bool,
    is_verbose: bool,
) -> tuple:
    """Execute part b"""
    t_vals, y_vals, vy_vals = calculate_numerical_predictions(
        jumper, global_vars, is_constant_drag, is_verbose
    )
    t_vals, y_vals, vy_vals = cut_negative_distance(
        t_vals, y_vals, vy_vals, determine_negative_distance(y_vals)
    )
    if plot:
        _ = input("Press anything to show plot...")
        plot_part_a_b_c(t_vals, y_vals, vy_vals)
        return _, _, _
    return t_vals, y_vals, vy_vals


def run_option_d(jumper: Jumper, global_vars: dict):
    """Execute option d"""
    t_vals, y_vals, vy_vals = run_option_b_c(jumper, global_vars, False, False, True)
    mach_ratio, vy_vals_abs, vsound_vals = generate_d_arrays(
        y_vals, vy_vals, global_vars
    )
    _ = input("Press anything to show plot...")
    plot_part_d_1(mach_ratio, t_vals, vy_vals_abs, vsound_vals)
    print("The maximum Mach number is", np.max(mach_ratio))
    print("The maximum speed is", np.max(vy_vals_abs))
    print("Investigating jump height")
    _ = input("Press anything to show plot...")
    vary_jump_height(global_vars)
    print("Investigating drag coefficient")
    _ = input("Press anything to show plot...")
    vary_drag_coeff(global_vars)


def run_option_e(global_vars: dict):
    """TODO: Docstring for run_option_e.

    :global_vars: TODO
    :returns: TODO

    """
    wavepacket = StringWave(2, 0.5, 12)
    fill_dataset(global_vars, wavepacket)


def cut_negative_distance(
    t_vals: np.ndarray, y_vals: np.ndarray, vy_vals: np.ndarray, zero_idx: int
) -> tuple:
    """Return val[:zero_idx]

    :vals: TODO
    :returns: TODO

    """
    return t_vals[:zero_idx], y_vals[:zero_idx], vy_vals[:zero_idx]


def determine_negative_distance(y_vals: np.ndarray) -> int:
    """Determine the index where the object hits the ground

    :y_vals: y array
    :returns: index

    """
    return np.where(y_vals < 0)[0][0]


def generate_d_arrays(
    y_vals: np.ndarray, vy_vals: np.ndarray, global_vars: dict
) -> tuple:
    """TODO: Docstring for generate_mach_ratio.

    :vy_vals_abs: TODO
    :returns: TODO

    """
    vsound_vals = generate_sound_values(vy_vals, y_vals, global_vars)
    vy_vals_abs = np.abs(vy_vals)
    mach_ratio = np.divide(vy_vals_abs, vsound_vals)
    return mach_ratio, vy_vals_abs, vsound_vals


def vary_jump_height(global_vars: dict):
    """TODO: Docstring for vary_jump_height.
    :returns: TODO

    """
    y_0 = np.linspace(1000, 40000, 300)
    max_mach = np.full_like(y_0, 0)
    max_speed = np.full_like(y_0, 0)
    duration = np.full_like(y_0, 0)
    for idx, val in enumerate(y_0):
        t_vals, y_vals, vy_vals = run_option_b_c(
            Jumper(1.0, val), global_vars, False, False, False
        )
        mach_ratio, vy_vals_abs, _ = generate_d_arrays(y_vals, vy_vals, global_vars)
        max_speed[idx] = np.max(vy_vals_abs)
        max_mach[idx] = np.max(mach_ratio)
        duration[idx] = t_vals[-1]
    plot_jump_height(y_0, max_mach, max_speed, duration)


def vary_drag_coeff(global_vars: dict):
    """TODO: Docstring for vary_jump_height.
    :returns: TODO

    """
    drag_coeff = np.linspace(0.05, 2.5, 300)
    max_mach = np.full_like(drag_coeff, 0)
    max_speed = np.full_like(drag_coeff, 0)
    duration = np.full_like(drag_coeff, 0)
    for idx, val in enumerate(drag_coeff):
        t_vals, y_vals, vy_vals = run_option_b_c(
            Jumper(val, 100), global_vars, False, False, False
        )
        mach_ratio, vy_vals_abs, _ = generate_d_arrays(y_vals, vy_vals, global_vars)
        max_speed[idx] = np.max(vy_vals_abs)
        max_mach[idx] = np.max(mach_ratio)
        duration[idx] = t_vals[-1]
    plot_drag_coeff(drag_coeff, max_mach, max_speed, duration)


def generate_sound_values(
    vy_vals: np.ndarray, y_vals: np.ndarray, global_vars: dict
) -> np.ndarray:
    """TODO: Docstring for generate_sound_values.

    :vy_vals: TODO
    :y_vals: TODO
    :returns: TODO

    """
    vsound_vals = np.full_like(vy_vals, 0)
    for idx, val in enumerate(y_vals):
        vsound_vals[idx] = calculate_sound_speed(
            calculate_temperature(val), global_vars
        )
    return vsound_vals


def calculate_temperature(altitude) -> float:
    """Evaluate the temperature of air given the altitude

    :altitude: TODO
    :returns: TODO

    """
    if altitude <= 11000:
        return 288.0 - 0.0065 * altitude
    if 11000 < altitude <= 25100:
        return 216.5
    return 141.3 + 0.003 * altitude


def calculate_sound_speed(temperature: float, global_vars: dict) -> float:
    """Evaluate the speed of sound given the temperature

    :temperature: TODO
    :returns: TODO

    """
    return np.sqrt(
        global_vars["gamma"]
        * global_vars["molar_gas_constant"]
        * temperature
        / global_vars["molar_gas_mass"]
    )


def calculate_drag_factor(y_val: float, jumper: Jumper, global_vars: dict) -> float:
    """Find the drag factor which varies by altitude"""
    rho = global_vars["rho_0"] * np.exp((-y_val) / global_vars["scale_height"])
    return jumper.drag_coeff * rho * jumper.cross_sec_area / 2


def calculate_numerical_predictions(
    jumper: Jumper, global_vars: dict, is_constant_drag: bool, is_verbose: bool
) -> tuple:
    """Evaluate the values of y & v_y for given t

    :param t_vals: time array
    :param y_vals: y array
    :param vy_vals: vy array
    :returns: the results after evaluation

    """
    t_vals, y_vals, vy_vals = (
        np.zeros(global_vars["points"]),
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
        t_vals[idx + 1] = t_vals[idx] + global_vars["timestep"]
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


def fill_dataset(global_vars: dict, wavepacket: StringWave):
    """TODO: Docstring for fill_dataset.

    :global_vars: TODO
    :wavepacket: TODO
    :returns: TODO

    """
    gamma = (
        global_vars["timestep"]
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
    dataset = {(round(x, 2), round(t, 2)): 0.0 for t in t_vals for x in x_vals}
    dataset = calculate_initial_timestep(
        global_vars, wavepacket, x_vals, t_vals, dataset
    )

    print(dataset)
    for point, value in dataset.items():
        if value > sys.float_info.epsilon:
            continue
        position = point[0]
        time = point[1]
        dataset[point] = calculate_next_timestep(
            (position, time, gamma), global_vars, wavepacket, dataset
        )
    print("gamma", gamma)
    print(wavepacket.std, wavepacket.phase_velocity, wavepacket.a_0)
    wave = list(dataset.values())
    for idx in range(np.size(t_vals)):
        curr_wave = wave[idx*np.size(x_vals):(idx+1)*np.size(x_vals)]
        plt.clf()
        plt.ylim([np.min(wave), np.max(wave)])
        plt.scatter(x_vals,curr_wave)
        plt.savefig(f"./waves/Figure {idx}.png", format="png", dpi=150, bbox_inches="tight")



def calculate_initial_timestep(
    global_vars: dict,
    wavepacket: StringWave,
    x_vals: np.ndarray,
    t_vals: np.ndarray,
    dataset: dict,
) -> dict:
    """TODO: Docstring for calculate_wave_equation.

    :position: TODO
    :time: TODO
    :global_vars: TODO
    :wavepacket: TODO
    :returns: TODO

    """
    l_s = round(global_vars["lattice spacing"], 2)
    t_s = round(global_vars["timestep2"], 2)
    t_0 = round(t_vals[0], 2)

    for position in x_vals:
        if (
            position < sys.float_info.epsilon
            or np.abs(position - wavepacket.length) < sys.float_info.epsilon
        ):
            continue
        position = round(position, 2)
        dataset[(position, t_0)] = evaluate_initial_displacement(
            position, t_0, wavepacket
        )
    for position in x_vals:
        if (
            position < sys.float_info.epsilon
            or np.abs(position - wavepacket.length) < sys.float_info.epsilon
        ):
            continue
        position = round(position, 2)
        dataset[(position, t_s)] = (1 / 2) * (
            evaluate_initial_displacement(position + l_s, t_0, wavepacket)
            + evaluate_initial_displacement(position - l_s, t_0, wavepacket)
        ) + t_s * 5
    return dataset


def calculate_next_timestep(
    information: tuple,
    global_vars: dict,
    wavepacket: StringWave,
    dataset: dict,
):
    """TODO: Docstring for calculate_next_timestep.
    :returns: TODO

    """
    position, time, gamma = information
    if (
        position <= sys.float_info.epsilon
        or np.abs(wavepacket.length - position) <= sys.float_info.epsilon
    ):
        return 0.0
    if round(position + global_vars["lattice spacing"], 2) <= wavepacket.length:
        return (
            gamma
            * (
                dataset[
                    (
                        round(position + global_vars["lattice spacing"], 2),
                        round(time - global_vars["timestep2"], 2),
                    )
                ]
                + dataset[
                    (
                        round(position - global_vars["lattice spacing"], 2),
                        round(time - global_vars["timestep2"], 2),
                    )
                ]
            )
            + (2 * (1 - gamma) * dataset[(round(position, 2), round(time, 2))])
            - dataset[round(position, 2), round(time - 2*global_vars["timestep2"], 2)]
        )
    return 0.0


def evaluate_initial_displacement(position: float, time: float, wavepacket: StringWave):
    """TODO: Docstring for evaluate_displacement.

    :position: TODO
    :time: TODO
    :returns: TODO

    """
    return wavepacket.a_0 * np.exp(
        -(((position - wavepacket.length / 2) - wavepacket.phase_velocity * time) ** 2)
        / (wavepacket.std) ** 2
    )


def evaluate_initial_time_derivative(
    position: float, time: float, wavepacket: StringWave
):
    """TODO: Docstring for evaluate_displacement.

    :position: TODO
    :time: TODO
    :returns: TODO

    """
    return (
        2
        * wavepacket.a_0
        * wavepacket.phase_velocity
        * (position - wavepacket.phase_velocity * time)
        * np.exp(
            -(
                ((position - wavepacket.length / 2) - wavepacket.phase_velocity * time)
                ** 2
            )
            / (wavepacket.std) ** 2
        )
        / (wavepacket.std) ** 2
    )


def print_header(
    jumper: Jumper,
    drag_factor: float,
    timestep: float,
    is_constant_drag: bool,
    solver_type: str,
):
    """TODO: Docstring for print_header.

    :jumper: TODO
    :drag_factor: TODO
    :returns: TODO

    """
    print(
        f"Calculating {solver_type} predictions for freefall with drag. Using:,",
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


def calculate_analytical_predictions(
    jumper: Jumper,
    global_vars: dict,
) -> tuple:
    """Evaluate the values of y & v_y for given t

    :param t_vals: time array
    :param y_vals: y array
    :param vy_vals: vy array
    :returns: the results after evaluation

    """
    t_vals = np.linspace(
        global_vars["t_min"], global_vars["t_max"], global_vars["points"]
    )
    y_vals, vy_vals = (
        np.zeros(global_vars["points"]),
        np.zeros(global_vars["points"]),
    )
    drag_factor = jumper.drag_coeff * global_vars["rho_0"] * jumper.cross_sec_area / 2
    print_header(jumper, drag_factor, 0, True, "analytical")
    y_vals = jumper.starting_height - (
        jumper.mass
        * np.log(
            np.cosh(np.sqrt(drag_factor * global_vars["grav"] / jumper.mass) * t_vals)
            / drag_factor
        )
    )
    vy_vals = -np.sqrt(jumper.mass * global_vars["grav"] / drag_factor) * np.tanh(
        np.sqrt(drag_factor * global_vars["grav"] / jumper.mass) * t_vals
    )
    return t_vals, y_vals, vy_vals


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


def plot_part_d_1(
    mach_ratio: np.ndarray,
    t_vals: np.ndarray,
    vy_vals_abs: np.ndarray,
    vsound_vals: np.ndarray,
):
    """TODO: Docstring for plot_part_d.

    :mach_ratio: TODO
    :t_vals: TODO
    :vy_vals_abs: TODO
    :vsound_vals: TODO
    :returns: TODO

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Height and speed data for Newtonian freefall with drag")
    ax1.set(xlabel="Time (s)", ylabel="Mach ratio", title="Mach ratio over time")
    ax2.set(
        xlabel=r"Speed of fall ($ms^{-1}$)",
        ylabel=r"Speed of sound ($ms^{-1}$)",
        title="Ratio of speeds",
    )
    ax1.scatter(t_vals, mach_ratio, s=2, c="r")
    ax2.scatter(vsound_vals, vy_vals_abs, s=2, c="g")
    plt.show()


def plot_jump_height(
    y_0: np.ndarray, max_mach: np.ndarray, max_speed: np.ndarray, duration: np.ndarray
):
    """TODO: Docstring for plot_jump_height.

    :y_0: TODO
    :max_mach: TODO
    :max_speed: TODO
    :duration: TODO
    :returns: TODO

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle("Investigating varying dropping heights")
    ax1.set(
        xlabel="Height (m)",
        ylabel="Mach ratio",
        title="Maximum Mach ratio at different dropping heights",
    )
    ax2.set(
        xlabel="Height (m)",
        ylabel=r"Speed ($ms^{-1}$)",
        title="Terminal speed at different dropping heights",
    )
    ax3.set(
        xlabel="Height (m)",
        ylabel="Time (s)",
        title="Duration of freefall at different dropping heights",
    )
    ax1.scatter(y_0, max_mach, s=2, c="r")
    ax2.scatter(y_0, max_speed, s=2, c="g")
    ax3.scatter(y_0, duration, s=2, c="b")
    plt.show()


def plot_drag_coeff(
    y_0: np.ndarray, max_mach: np.ndarray, max_speed: np.ndarray, duration: np.ndarray
):
    """TODO: Docstring for plot_jump_height.

    :y_0: TODO
    :max_mach: TODO
    :max_speed: TODO
    :duration: TODO
    :returns: TODO

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle("Investigating varying drag coefficients")
    ax1.set(
        xlabel="Drag coefficient",
        ylabel="Mach ratio",
        title="Maximum Mach ratio at different drag coefficients",
    )
    ax2.set(
        xlabel="Drag coefficient",
        ylabel=r"Speed ($ms^{-1}$)",
        title="Terminal speed at different drag coefficients",
    )
    ax3.set(
        xlabel="Drag coefficient",
        ylabel="Time (s)",
        title="Duration of freefall at drag coefficients",
    )
    ax1.scatter(y_0, max_mach, s=2, c="r")
    ax2.scatter(y_0, max_speed, s=2, c="g")
    ax3.scatter(y_0, duration, s=2, c="b")
    plt.show()


def main():
    """Driver code"""
    user_input = "0"
    global_vars = {
        "t_min": 0,
        "t_max": 60,
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
        "T": 20,
    }
    default_jumper = Jumper(1.0, 1000)
    while user_input != "q":
        user_input = input(
            'Enter a choice, "a", "b", "c", "d", "e", "f", "g", "h" for help, or "q" to quit: '
        )
        print("You entered the choice: ", user_input)
        print(f"You have chosen part ({user_input})")
        if user_input == "a":
            run_option_a(default_jumper, global_vars)
        elif user_input == "b":
            run_option_b_c(default_jumper, global_vars, True, True, True)
        elif user_input == "c":
            run_option_b_c(Jumper(1.0, 39045), global_vars, False, True, True)
        elif user_input == "d":
            run_option_d(Jumper(1.0, 39045), global_vars)
        elif user_input == "e":
            run_option_e(global_vars)
        # elif user_input == "f":
        #     run_option_f()
        # elif user_input == "h":
        #     run_option_h()
        elif user_input != "q":
            print("This is not a valid choice.")
    print("You have chosen to finish - goodbye.")


if __name__ == "__main__":
    main()
