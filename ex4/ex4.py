#!/usr/bin/env python3
"""
Exercise 4
"""
from copy import deepcopy as dc
from decimal import Decimal

import matplotlib.animation as anim
import matplotlib.collections as clt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

#################################
#  OPTION INITIATION FUNCTIONS  #
#################################


def run_option_a(global_vars: dict):
    """Execute part a

    :param global_vars: global variables

    """
    print("-> Scenario: Earth at origin.")
    print(
        "-> Example initial conditions (x,y,v_x,v_y):",
        "\n\t -> Near/approximately circular orbits:"
        "\n\t\t -> (0,-1.1,-7966,0) max_time = 10000"
        "\n\t -> Elliptical orbits:"
        "\n\t\t -> (0,-1,-10000,0) max_time = 25000"
        "\n\t -> Crash scenario:"
        "\n\t\t -> (0,-1,-7900,-3000) max_time = 9000",
        "\n\t\t -> (0,-1,0,-9000) max_time = 13000",
        "\n\t -> Escape velocity:" "\n\t\t -> (0,-1,0,-11500) max_time = 1000000",
    )
    print_useful_values(global_vars)
    x_0 = global_vars["radius_earth"] * float(
        take_input_float("initial x displacement in multiples of R_E (default: 0)")
        or "0"
    )
    y_0 = global_vars["radius_earth"] * float(
        take_input_float("initial y displacement in multiples of R_E (default: 0)")
        or "0"
    )
    v_x_0 = float(take_input_float("initial x velocity in m/s (default: 0)") or "0")
    v_y_0 = float(take_input_float("initial y velocity in m/s (default: 0)") or "0")
    max_time = int(take_input_int("max time to render (default: 0)", False) or "0")
    x_pos, y_pos, x_vel, y_vel, time = populate_rk4_1body(
        global_vars, np.array([x_0, y_0, v_x_0, v_y_0]), max_time, False
    )
    stop = check_crash(
        x_pos[1:],
        y_pos[1:],
        np.zeros_like(x_pos[1:]),
        np.zeros_like(y_pos[1:]),
        global_vars["radius_earth"],
    )
    skip = calculate_skip(global_vars, stop, x_pos, y_pos, True)
    print("-> Displaying overview of selected orbit.")
    input("* Press anything to continue...")
    plot_overview_orbit_1body(
        global_vars,
        x_pos[:stop] / global_vars["radius_earth"],
        y_pos[:stop] / global_vars["radius_earth"],
        False,
    )
    print("-> Displaying animation of selected orbit.")
    input("* Press anything to continue...")
    filename = (
        f"1orbit_{x_0/global_vars['radius_earth']}"
        + f"_{y_0/global_vars['radius_earth']}"
        + f"_{v_x_0}_{v_y_0}_{global_vars['step_size']}"
    )
    animate_orbit_1body(
        global_vars,
        x_pos[:stop:skip] / global_vars["radius_earth"],
        y_pos[:stop:skip] / global_vars["radius_earth"],
        filename,
        False,
    )
    print("-> Displaying energy.")
    input("* Press anything to continue...")
    _ = check_energy_conservation_1body(
        global_vars,
        x_pos[:stop],
        y_pos[:stop],
        x_vel[:stop],
        y_vel[:stop],
        time[:stop],
        True,
        False,
    )
    run_option_a_aux(global_vars)
    input("* Press anything to return to menu...")


def run_option_a_aux(global_vars: dict):
    """Execute part a auxiliary stuff

    :global_vars: TODO
    :returns: TODO

    """
    step_sizes = np.linspace(10, 200, 400)
    errors = np.zeros_like(step_sizes)
    print("-> Investigating impact of step size on energy conservation accuracy.")
    default_variables = np.array([0, global_vars["radius_earth"] * 1.1, -10000, 0])
    max_time = 30000
    for idx, var in enumerate(step_sizes):
        global_vars["step_size"] = var
        x_pos, y_pos, x_vel, y_vel, time = populate_rk4_1body(
            global_vars, default_variables, max_time, False
        )
        errors[idx] = check_energy_conservation_1body(
            global_vars, x_pos, y_pos, x_vel, y_vel, time, False, False
        )
    plot_energy_conservation(step_sizes, errors)


def run_option_b(global_vars: dict):
    """Execute part b

    :param global_vars: global variables

    """
    print("-> Scenario: Earth at origin and moon at x=0 and y=earth-moon distance.")
    print(
        "\n-> Example initial conditions (x,y,v_x,v_y):",
        "\n\t -> Figure 8 flyby (and closest approach):"
        "\n\t\t -> (0,-1.09,-10591,0) max_time = 900000"
        "\n\t -> Moon crash:"
        "\n\t\t -> (0,-1.09,-10600,0) max_time = 900000"
        "\n\t -> Moon elliptical flyby:"
        "\n\t\t -> (0,-1.09,-10610,0) max_time = 1600000",
    )
    print_useful_values(global_vars)
    x_0 = global_vars["radius_earth"] * float(
        take_input_float("initial x displacement in multiples of R_E (default: 0)")
        or "0"
    )
    y_0 = global_vars["radius_earth"] * float(
        take_input_float("initial y displacement in multiples of R_E (default: 0)")
        or "0"
    )
    v_x_0 = float(take_input_float("initial x velocity in m/s (default: 0)") or "0")
    v_y_0 = float(take_input_float("initial y velocity in m/s (default: 0)") or "0")
    max_time = int(take_input_int("max time to render (default: 0)", False) or "0")
    x_pos, y_pos, x_vel, y_vel, time = populate_rk4_1body(
        global_vars, np.array([x_0, y_0, v_x_0, v_y_0], dtype=np.float_), max_time, True
    )
    stop = check_crash(
        x_pos[1:],
        y_pos[1:],
        np.zeros_like(x_pos[1:]),
        np.zeros_like(y_pos[1:]),
        global_vars["radius_earth"],
    ) or check_crash(
        x_pos[1:],
        y_pos[1:],
        np.zeros_like(x_pos[1:]),
        np.full_like(y_pos[1:], global_vars["earth_moon_distance"]),
        global_vars["radius_moon"],
    )
    skip = calculate_skip(global_vars, stop, x_pos, y_pos, False)
    print("-> Displaying overview of selected orbit.")
    input("* Press anything to continue...")
    plot_overview_orbit_1body(
        global_vars,
        x_pos[:stop] / global_vars["radius_earth"],
        y_pos[:stop] / global_vars["radius_earth"],
        True,
    )
    print("-> Displaying animation of selected orbit.")
    input("* Press anything to continue...")
    filename = (
        f"1orbit_{x_0/global_vars['radius_earth']}"
        + f"_{y_0/global_vars['radius_earth']}"
        + f"_{v_x_0}_{v_y_0}_{global_vars['step_size']}"
    )
    animate_orbit_1body(
        global_vars,
        x_pos[:stop:skip] / global_vars["radius_earth"],
        y_pos[:stop:skip] / global_vars["radius_earth"],
        filename,
        True,
    )
    distance_from_earth = calculate_distance(x_pos[:stop], y_pos[:stop])
    distance_from_moon = calculate_distance(
        x_pos[:stop], y_pos[:stop] - global_vars["earth_moon_distance"]
    )
    print("-> Displaying distances from earth and moon across flight.")
    input("* Press anything to continue...")
    plot_rocket_earth_moon(time[:stop], distance_from_earth, distance_from_moon)
    if stop is None:
        analyse_rocket_trip(distance_from_earth, distance_from_moon, time)
    print("-> Displaying energy.")
    input("* Press anything to continue...")
    _ = check_energy_conservation_1body(
        global_vars,
        x_pos[:stop],
        y_pos[:stop],
        x_vel[:stop],
        y_vel[:stop],
        time[:stop],
        True,
        True,
    )
    input("* Press anything to return to menu...")


def run_option_c(global_vars: dict):
    """Execute part c

    :param global_vars: global variables

    """
    print("-> Scenario: Two bodies moving under the other's gravitational field.")
    print(
        "-> Example initial conditions 1:(x,y,v_x,v_y) 2:(x,y,v_x,v_y)",
        "for 2 identical bodies:",
        "\n\t ! DON'T use earth-moon distance scale."
        "\n\t -> Bodies moving away:"
        "\n\t\t -> (-5,0,0,-1500) (5,0,0,1500) max_time=900000 scale=10"
        "\n\t\t -> (-5,0,0,-1800) (5,0,0,1800) max_time=900000 scale=10"
        "\n\t -> Close deflect:"
        "\n\t\t -> (-20,-20,1500,2000) (20,20,-1500,-2000) max_time=130000 scale=30"
        "\n\t -> Crash:"
        "\n\t\t -> (-5,0,0,-500) (5,0,0,500) max_time=100000 scale=10"
        "\n\t\t -> (-5,0,300,0) (5,0,0,0) max_time=100000 scale=10"
        "\n-> Example initial conditions 1:(x,y,v_x,v_y) 2:(x,y,v_x,v_y)",
        "for earth & moon:",
        "\n\t ! USE earth-moon distance scale."
        "\n\t -> Close deflect/slingshot:"
        "\n\t\t -> (0,0,0,0) (0,1,1000,-5000) max_time=120000 scale=60",
        "\n\t\t -> (0,0,0,0) (0,0.5,500,0) max_time=500000 scale=40",
        "\n\t -> Crash:",
        "\n\t\t -> (0,0,0,0) (0,1,0,-8000) max_time=120000 scale=60",
        "\n\t -> Several orbits (note: takes a long time to calculate):"
        "\n\t\t -> (0,0,0,0) (0,1,1000,0) max_time=10000000 scale=75",
    )
    print_useful_values(global_vars)
    is_alt_scale = bool(
        take_input_bool("Use earth-moon distance scale? True/False (default: False)")
    )
    mass_2 = (
        float(
            take_input_float(
                "mass of body 2 in multiples of M_E (body 1's mass) (default: 0)"
            )
            or "0"
        )
        * global_vars["mass_earth"]
    )
    radius_2 = float(
        take_input_float(
            "radius of body 2 in multiples of R_E (body 1's radius) (default: 0)"
        )
        or "0"
    )
    x_0_1 = global_vars["radius_earth"] * float(
        take_input_float(
            "initial body 1 x displacement in multiples of R_E (default: 0)"
        )
        or "0"
    )
    y_0_1 = global_vars["radius_earth"] * float(
        take_input_float(
            "initial body 1 y displacement in multiples of R_E (default: 0)"
        )
        or "0"
    )
    v_x_0_1 = float(
        take_input_float("initial body 1 x velocity in m/s (default: 0)") or "0"
    )
    v_y_0_1 = float(
        take_input_float("initial body 1 y velocity in m/s (default: 0)") or "0"
    )
    x_0_2 = (
        (is_alt_scale) * global_vars["earth_moon_distance"]
        + global_vars["radius_earth"] * (not is_alt_scale)
    ) * float(
        take_input_float(
            "initial body 2 x displacement in multiples of earth-moon-separation or R_E (default: 0)"
        )
        or "0"
    )
    y_0_2 = (
        (is_alt_scale) * global_vars["earth_moon_distance"]
        + global_vars["radius_earth"] * (not is_alt_scale)
    ) * float(
        take_input_float(
            "initial body 2 y displacement in multiples of earth-moon-separation or R_E (default: 0)"
        )
        or "0"
    )
    v_x_0_2 = float(
        take_input_float("initial body 2 x velocity in m/s (default: 0)") or "0"
    )
    v_y_0_2 = float(
        take_input_float("initial body 2 y velocity in m/s (default: 0)") or "0"
    )
    max_time = int(take_input_int("max time to render (default: 0)", False) or "0")
    start_1 = np.array([x_0_1, y_0_1, v_x_0_1, v_y_0_1])
    start_2 = np.array([x_0_2, y_0_2, v_x_0_2, v_y_0_2])
    (
        x_pos_1,
        y_pos_1,
        x_pos_2,
        y_pos_2,
        x_vel_1,
        y_vel_1,
        x_vel_2,
        y_vel_2,
        time,
    ) = populate_rk4_2body(
        global_vars,
        start_1,
        start_2,
        max_time,
        mass_2,
        radius_2 * global_vars["radius_earth"],
        False,
    )
    stop = check_crash(
        x_pos_1[1:],
        y_pos_1[1:],
        x_pos_2[1:],
        y_pos_2[1:],
        (radius_2 + 1) * global_vars["radius_earth"],
    )
    print("-> Displaying overview of selected orbit.")
    input("* Press anything to continue...")
    plot_overview_orbit_2body(
        global_vars,
        x_pos_1[:stop] / global_vars["radius_earth"],
        y_pos_1[:stop] / global_vars["radius_earth"],
        x_pos_2[:stop] / global_vars["radius_earth"],
        y_pos_2[:stop] / global_vars["radius_earth"],
        False,
    )
    skip = calculate_skip(global_vars, stop, x_pos_1, y_pos_1, False)
    filename = (
        f"2orbit_{start_1[0]/global_vars['radius_earth']}"
        + f"_{start_1[1]/global_vars['radius_earth']}_{start_1[2]}"
        + f"_{start_1[3]}_{start_2[0]/global_vars['radius_earth']}"
        + f"_{start_2[1]/global_vars['radius_earth']}"
        + f"_{start_2[2]}_{start_2[3]}_{global_vars['step_size']}"
    )
    animate_orbit_2body(
        global_vars,
        x_pos_1[:stop:skip] / global_vars["radius_earth"],
        y_pos_1[:stop:skip] / global_vars["radius_earth"],
        x_pos_2[:stop:skip] / global_vars["radius_earth"],
        y_pos_2[:stop:skip] / global_vars["radius_earth"],
        radius_2,
        filename,
        False,
    )
    print("-> Displaying energy.")
    input("* Press anything to continue...")
    _ = check_energy_conservation_2body(
        global_vars,
        x_pos_1[:stop],
        y_pos_1[:stop],
        x_pos_2[:stop],
        y_pos_2[:stop],
        x_vel_1[:stop],
        y_vel_1[:stop],
        x_vel_2[:stop],
        y_vel_2[:stop],
        mass_2,
        time[:stop],
        True,
        False,
    )
    input("* Press anything to return to menu...")


def run_option_d(global_vars: dict):
    """Execute option d

    :param global_vars: global variables

    """
    print(
        "-> Scenario: Earth and Sun moving under the other's and",
        "the Sun's gravitational field.",
    )
    print("-> Recommended max_time: >900000 s (31,557,514 s in a year)")
    print("-> Recommended scale: 23600")
    max_time = int(take_input_int("max time to render (default: 0)", False) or "0")
    body_1_initial = np.array([0, global_vars["earth_sun_distance"], -2.98e4, 0])
    body_2_initial = np.array(
        [
            0,
            global_vars["earth_sun_distance"] + global_vars["earth_moon_distance"],
            -2.98e4 - 1000,
            0,
        ]
    )
    (
        x_pos_1,
        y_pos_1,
        x_pos_2,
        y_pos_2,
        x_vel_1,
        y_vel_1,
        x_vel_2,
        y_vel_2,
        time,
    ) = populate_rk4_2body(
        global_vars,
        body_1_initial,
        body_2_initial,
        max_time,
        global_vars["radius_moon"],
        (global_vars["mass_moon"] / global_vars["mass_earth"]),
        True,
    )
    stop = check_crash(
        x_pos_1[1:],
        y_pos_1[1:],
        x_pos_2[1:],
        y_pos_2[1:],
        global_vars["radius_moon"] + global_vars["radius_earth"],
    )
    print("-> Displaying overview of selected orbit.")
    input("* Press anything to continue...")
    plot_overview_orbit_2body(
        global_vars,
        x_pos_1 / global_vars["radius_earth"],
        y_pos_1 / global_vars["radius_earth"],
        x_pos_2 / global_vars["radius_earth"],
        y_pos_2 / global_vars["radius_earth"],
        True,
    )
    skip = calculate_skip(global_vars, stop, x_pos_1, y_pos_1, False)
    filename = f"sun_2orbit_{max_time}"
    animate_orbit_2body(
        global_vars,
        x_pos_1[:stop:skip] / global_vars["radius_earth"],
        y_pos_1[:stop:skip] / global_vars["radius_earth"],
        x_pos_2[:stop:skip] / global_vars["radius_earth"],
        y_pos_2[:stop:skip] / global_vars["radius_earth"],
        global_vars["radius_moon"] / global_vars["radius_earth"],
        filename,
        True,
    )
    earth_sun_distance = calculate_distance(x_pos_1[:stop], y_pos_1[:stop])
    moon_sun_distance = calculate_distance(x_pos_2[:stop], y_pos_2[:stop])
    earth_moon_distance = calculate_distance(
        x_pos_1[:stop] - x_pos_2[:stop], y_pos_1[:stop] - y_pos_2[:stop]
    )
    plot_sun_earth_moon(
        time[:stop],
        earth_sun_distance,
        moon_sun_distance,
        earth_moon_distance,
    )
    print("-> Displaying energy.")
    input("* Press anything to continue...")
    _ = check_energy_conservation_2body(
        global_vars,
        x_pos_1[:stop],
        y_pos_1[:stop],
        x_pos_2[:stop],
        y_pos_2[:stop],
        x_vel_1[:stop],
        y_vel_1[:stop],
        x_vel_2[:stop],
        y_vel_2[:stop],
        global_vars["mass_moon"],
        time[:stop],
        True,
        True,
    )
    input("* Press anything to return to menu...")


##################################
#  PROBLEM 1 & 2 MAIN FUNCTIONS  #
##################################


def check_shape(global_vars: dict, x_pos: np.ndarray, y_pos: np.ndarray):
    """Check the shape of the orbit by calculating its major and minor axes

    :param global_vars: global variables
    :param x_pos: the x positions of orbit
    :param y_pos: the y positions of orbit

    """
    d_x = (np.abs(np.max(x_pos)) + np.abs(np.min(x_pos))) / global_vars["radius_earth"]
    d_y = (np.abs(np.max(y_pos)) + np.abs(np.min(y_pos))) / global_vars["radius_earth"]
    print("-> Major axis:", round_with_decimal(3, np.max([d_x, d_y])), "R_E")
    print("-> Minor axis:", round_with_decimal(3, np.min([d_x, d_y])), "R_E")
    if np.abs((d_x / d_y) - 1) <= 0.1:
        print("-> Orbit is approximately circular.")
    else:
        print("-> Orbit is not circular.")


def solve_rk4(
    global_vars: dict,
    k_1: np.ndarray,
    k_2: np.ndarray,
    k_3: np.ndarray,
    k_4: np.ndarray,
    variables: np.ndarray,
) -> np.ndarray:
    """Return the system variables (x,y,v_x,v_y) at next timestep

    :param global_vars: global variables
    :param k_1: k1 array
    :param k_2: k2 array
    :param k_3: k3 array
    :param k_4: k4 array
    :param variables: system variables at present timestep
    :returns: system variables at next timestep

    """
    factor = np.sum(np.column_stack((k_1, 2 * k_2, 2 * k_3, k_4)), axis=1)
    return variables + global_vars["step_size"] * (factor) / 6


def calculate_skip(
    global_vars: dict,
    stop: None | int,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    is_check: bool,
) -> int | None:
    """Find the skip factor to stay under max frame size

    :param global_vars: global variables
    :param stop: the final index
    :param x_pos: body's x position array
    :param y_pos: body's y position array
    :param is_check: boolean to calculate orbit shape
    :returns: skip factor

    """
    if stop is None:
        skip = (np.size(x_pos) // global_vars["gif_max_frames"]) * (
            np.size(x_pos) >= global_vars["gif_max_frames"]
        ) + -1 * (np.size(x_pos) < global_vars["gif_max_frames"])
        if is_check:
            check_shape(global_vars, x_pos, y_pos)
    else:
        skip = stop // global_vars["gif_max_frames"] * (
            stop >= global_vars["gif_max_frames"]
        ) + -1 * (stop < global_vars["gif_max_frames"])
        print("-> Collision detected.")
    return skip + 1 if skip != -1 else None


def check_crash(
    x_pos_earth: np.ndarray,
    y_pos_earth: np.ndarray,
    x_pos_moon: np.ndarray,
    y_pos_moon: np.ndarray,
    min_length: float,
):
    """Return the index where, if valid, a collision happens. A quick
    vectorised function to double check for collisions.

    :param x_pos_earth: x positions of body 1
    :param y_pos_earth: y positions of body 1
    :param x_pos_moon: x positions of body 2
    :param y_pos_moon: y positions of body 2
    :param min_length: the minimum distance between two before collision
    :returns: TODO

    """
    crash_idx = np.where(
        np.sqrt((x_pos_earth - x_pos_moon) ** 2 + (y_pos_earth - y_pos_moon) ** 2)
        <= min_length
    )
    if np.size(crash_idx[0]) == 0:
        return None
    return crash_idx[0][1] + 1


def get_all_ks(
    global_vars: dict,
    variables_1: np.ndarray,
    variables_2: np.ndarray,
    is_moon: bool,
    is_sun: bool,
    bodies: int,
    mass_2: float,
) -> tuple:
    """Fetch all the k arrays 1 to 4

    :param global_vars: global variables
    :param variables_1: system variables for body 1
    :param variables_2: system variables for body 2
    :param is_moon: boolean to factor in moon's G-field
    :param is_sun: boolean to factor in sun's G-field
    :param bodies: number of moving bodies with G-fields
    :param moon_mass: mass of body 2
    :returns: tuples with all k arrays

    """
    if bodies == 1:
        k_1 = np.array(get_k_factors_1body(global_vars, variables_1, is_moon))
        k_2 = np.array(
            get_k_factors_1body(
                global_vars, variables_1 + global_vars["step_size"] * k_1 / 2, is_moon
            )
        )
        k_3 = np.array(
            get_k_factors_1body(
                global_vars, variables_1 + global_vars["step_size"] * k_2 / 2, is_moon
            )
        )
        k_4 = np.array(
            get_k_factors_1body(
                global_vars, variables_1 + global_vars["step_size"] * k_3, is_moon
            )
        )
    else:
        k_1 = np.array(
            get_k_factors_2body(
                global_vars, variables_1, variables_2, is_moon, is_sun, mass_2
            )
        )
        k_2 = np.array(
            get_k_factors_2body(
                global_vars, variables_1, variables_2, is_moon, is_sun, mass_2
            )
        )
        k_3 = np.array(
            get_k_factors_2body(
                global_vars, variables_1, variables_2, is_moon, is_sun, mass_2
            )
        )
        k_4 = np.array(
            get_k_factors_2body(
                global_vars, variables_1, variables_2, is_moon, is_sun, mass_2
            )
        )
    return k_1, k_2, k_3, k_4


def analyse_rocket_trip(
    distance_from_earth: np.ndarray,
    distance_from_moon: np.ndarray,
    time: np.ndarray,
):
    """Analyse rocket earth-moon round trip

    :param distance_from_earth: rocket distance from earth
    :param distance_from_moon: rocket distance from moon
    :param time: time array

    """
    check = bool(
        take_input_bool(
            "Analyse rocket's distances from earth and moon? True/False (default: False)"
        )
    )
    if check:
        minima_earth = ss.argrelextrema(distance_from_earth, np.less)
        minima_moon = ss.argrelextrema(distance_from_moon, np.less)
        print(
            "-> Closest approach to earth:",
            round_with_decimal(3, float(distance_from_earth[minima_earth[0][0]])),
            "m.",
        )
        print(
            "-> Closest approach to moon:",
            round_with_decimal(3, float(distance_from_moon[minima_moon[0][0]])),
            "m.",
        )
        print(
            "-> Total time of one round trip:",
            round_with_decimal(3, float(time[minima_earth[0][0]]) / (3600 * 24)),
            "days.",
        )


#################################
#  ONE-BODY SPECIFIC FUNCTIONS  #
#################################


def populate_rk4_1body(
    global_vars: dict, variables: np.ndarray, max_time: int, is_moon: bool
) -> tuple:
    """Perform the RK4 method by fetching k arrays for each timestep,
    solve the next timestep, and then repeat for 1 body, checking for crash
    every 20 iterations to halt early.

    :param global_vars: global variables
    :param variables: initial conditions
    :param max_time: maximum time to render in secs
    :param is_moon: boolean to factor in moon's G-field
    :returns: system variables (x,y,v_x,v_y) for body 1 and time

    """
    time = np.arange(0, max_time, global_vars["step_size"])
    early_halt = False
    halt_idx = None
    x_pos, y_pos = np.zeros_like(time, dtype=np.float_), np.zeros_like(
        time, dtype=np.float_
    )
    x_vel, y_vel = np.zeros_like(time), np.zeros_like(time)
    for idx in range(np.size(time)):
        x_pos[idx], y_pos[idx] = variables[0], variables[1]
        x_vel[idx], y_vel[idx] = variables[2], variables[3]
        k_1, k_2, k_3, k_4 = get_all_ks(
            global_vars, variables, np.array([]), is_moon, False, 1, 0
        )
        variables_temp = solve_rk4(global_vars, k_1, k_2, k_3, k_4, dc(variables))
        if idx != 0 and idx % 20 == 0:
            crash = calculate_distance(variables_temp[0], variables[1]) <= global_vars[
                "radius_earth"
            ] or is_moon * (
                calculate_distance(
                    variables_temp[0],
                    np.abs(global_vars["earth_moon_distance"] - variables_temp[1]),
                )
                <= global_vars["radius_moon"]
            )
            if crash:
                print("-> Note: Collision detected before max time reached.")
                early_halt = True
                halt_idx = idx
                break
        variables = dc(variables_temp)
    if early_halt:
        x_pos, y_pos = x_pos[:halt_idx], y_pos[:halt_idx]
        x_vel, y_vel = x_vel[:halt_idx], y_vel[:halt_idx]
        time = time[:halt_idx]
    return x_pos, y_pos, x_vel, y_vel, time


def get_k_factors_1body(
    global_vars: dict, variables: np.ndarray, is_moon: bool
) -> tuple:
    """Return the k values (time derivs) for each value in variables for 1 body

    :param global_vars: global variables
    :param variables: system variables (x, y, v_x, v_y)
    :param is_moon: boolean to factor in moon's G-field
    :returns: (kx, ky, kvx, kvy)

    """
    x_i, y_i, v_x, v_y = variables
    denom_1 = (x_i**2 + y_i**2) ** (3 / 2)
    denom_2 = 1
    if is_moon:
        denom_2 = (x_i**2 + (y_i - global_vars["earth_moon_distance"]) ** 2) ** (
            3 / 2
        )
    return (
        v_x,
        v_y,
        -global_vars["grav_constant"]
        * (
            global_vars["mass_earth"] * x_i / denom_1
            + is_moon * (global_vars["mass_moon"] * x_i / denom_2)
        ),
        -global_vars["grav_constant"]
        * (
            global_vars["mass_earth"] * y_i / denom_1
            + is_moon
            * (
                global_vars["mass_moon"]
                * (y_i - global_vars["earth_moon_distance"])
                / denom_2
            )
        ),
    )


def check_energy_conservation_1body(
    global_vars: dict,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_vel: np.ndarray,
    y_vel: np.ndarray,
    time: np.ndarray,
    is_display: bool,
    is_moon: bool,
) -> float:
    """Calculate PE, KE and total energy

    :param global_vars: global_variables
    :param x_pos: x positions
    :param y_pos: y positions
    :param x_vel: x velocity arrays
    :param y_vel: y velocity arrays
    :param is_display: boolean to show information
    :returns: the standard deviation in energy

    """
    energy_kin = 0.5 * (x_vel**2 + y_vel**2)
    energy_pot = -(global_vars["grav_constant"] * global_vars["mass_earth"]) / (
        x_pos**2 + y_pos**2
    ) ** (0.5)
    if is_moon:
        energy_pot += -(
            global_vars["grav_constant"] * global_vars["mass_moon"]
        ) / calculate_distance(
            x_pos, np.abs(y_pos - global_vars["earth_moon_distance"])
        )
    energy = energy_kin + energy_pot
    avg_energy = float(np.mean(energy))
    std_energy = float(np.std(energy))
    if is_display:
        print(
            "-> Average energy per unit rocket mass:",
            round_with_decimal(3, avg_energy),
            "J/kg",
        )
        print(
            "-> Standard deviation in energy per unit rocket mass:",
            round_with_decimal(3, std_energy),
            "J/kg",
        )
        print("-> Showing total energy at each timestep")
        input("* Press anything to continue...")
        plot_energies(time, energy_kin, energy_pot, energy, True)
    return std_energy


def animate_orbit_1body(
    global_vars: dict,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    filename: str,
    is_moon: bool,
):
    """Animate a moving body

    :param global_vars: global variables,
    :param x_pos: x position array,
    :param y_pos: y position array
    :param filename: filename used to save the animation,
    :param is_moon: boolean to factor in the moon's G-field,

    """
    fig, axi = plt.subplots(figsize=(9, 9))

    xmax = np.max([np.abs(np.min(x_pos)), np.max(x_pos)])
    ymax = np.max([np.abs(np.min(y_pos)), np.max(y_pos)])
    ani_scale = np.max([xmax, ymax])
    earth = plt.Circle((0, 0), radius=1, color="blue", edgecolor=None)

    def animate(i):
        axi.clear()  # type: ignore
        x_i, y_i = x_pos[:i], y_pos[:i]
        axi.plot(x_i, y_i, color="lightsteelblue", lw=2, alpha=0.8)  # type: ignore
        axi.plot(x_pos[i], y_pos[i], color="lightsteelblue", marker="o", ms=10)  # type: ignore
        axi.set_xlim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
        axi.set_ylim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
        axi.set_xlabel(r"x distance ($R_E$)")  # type: ignore
        axi.set_ylabel(r"y distance ($R_E$)")  # type: ignore
        axi.add_patch(earth)  # type: ignore
        if is_moon:
            moon = plt.Circle(
                (0, global_vars["earth_moon_distance"] / global_vars["radius_earth"]),
                radius=(global_vars["radius_moon"] / global_vars["radius_earth"]),
                color="olive",
                edgecolor=None,
            )
            axi.add_patch(moon)  # type: ignore

    ani = anim.FuncAnimation(fig, animate, frames=len(x_pos) - 1, interval=1)
    plt.show()
    is_save = bool(
        take_input_bool("Do you want to save a gif? True/False (default: False)")
    )
    if is_save:
        if len(x_pos) - 1 > global_vars["gif_max_frames"] + 1:
            print(
                f"! {len(x_pos) - 1} frames in total.",
                f"Too many frames (>{global_vars['gif_max_frames']}).",
            )
        else:
            print("-> Saving gif...")
            writergif = anim.PillowWriter(fps=30)
            ani.save(f"{filename}.gif", writer=writergif)


def plot_overview_orbit_1body(
    global_vars: dict, x_pos: np.ndarray, y_pos: np.ndarray, is_moon: bool
):
    """Plot the trajectory of flight

    :param global_vars: global variables
    :param x_pos: x position arrays
    :param y_pos: y_position arrays
    :param is_moon: boolean to factor in moon's G-field

    """
    _, axi = plt.subplots(figsize=(9, 9))
    xmax = np.max([np.abs(np.min(x_pos)), np.max(x_pos)])
    ymax = np.max([np.abs(np.min(y_pos)), np.max(y_pos)])
    ani_scale = np.max([xmax, ymax])
    axi.set_xlabel(r"x distance ($R_E$)")  # type: ignore
    axi.set_ylabel(r"y distance ($R_E$)")  # type: ignore
    axi.set_xlim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
    axi.set_ylim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
    axi.plot(x_pos, y_pos, color="k")  # type: ignore

    earth = plt.Circle((0, 0), radius=1, color="blue", edgecolor=None)
    axi.add_patch(earth)  # type: ignore
    if is_moon:
        moon = plt.Circle(
            (0, global_vars["earth_moon_distance"] / global_vars["radius_earth"]),
            radius=(global_vars["radius_moon"] / global_vars["radius_earth"]),
            color="olive",
            edgecolor=None,
        )
        axi.add_patch(moon)  # type: ignore
    plt.show()


#################################
#  TWO-BODY SPECIFIC FUNCTIONS  #
#################################


def get_k_factors_2body(
    global_vars: dict,
    variables_1: np.ndarray,
    variables_2: np.ndarray,
    is_moon: bool,
    is_sun: bool,
    mass_2: float,
) -> tuple:
    """Return the k values (time derivs) for each value in variables for 2 bodies

    :param global_vars: global variables
    :param variables_1: system variables (x, y, v_x, v_y) for body 1
    :param variables_2: system variables (x, y, v_x, v_y) for body 2
    :param is_moon: boolean to switch between earth/moon's G-field
    :param is_sun: boolean to factor in sun's G-field
    :param moon_mass: mass of body 2
    :returns: (kx, ky, kvx, kvy)

    """
    x_i, y_i, v_x, v_y = variables_1
    x_prime, y_prime, _, _ = variables_2
    denom_1 = ((x_i - x_prime) ** 2 + (y_i - y_prime) ** 2) ** (3 / 2)
    denom_2 = 1
    if is_sun:
        denom_2 = (x_i**2 + y_i**2) ** (3 / 2)
    return (
        v_x,
        v_y,
        -global_vars["grav_constant"]
        * (
            (
                (global_vars["mass_earth"] * is_moon + mass_2 * (not is_moon))
                * (x_i - x_prime)
                / denom_1
            )
            + (global_vars["mass_sun"] * is_sun * x_i / denom_2)
        ),
        -global_vars["grav_constant"]
        * (
            (
                (global_vars["mass_earth"] * is_moon + mass_2 * (not is_moon))
                * (y_i - y_prime)
                / denom_1
            )
            + (global_vars["mass_sun"] * is_sun * y_i / denom_2)
        ),
    )


def populate_rk4_2body(
    global_vars: dict,
    body_1_initial: np.ndarray,
    body_2_initial: np.ndarray,
    max_time: int,
    body_2_mass: float,
    body_2_radius: float,
    is_sun: bool,
) -> tuple:
    """Perform the RK4 method by fetching k arrays for each timestep, solve the
    next timestep, and then repeat for 2 bodies. Checking for crash
    every 20 iterations to halt early.

    :param global_vars: global variables
    :param body_1_initial: body 1 initial conditions
    :param body_2_initial: body 2 initial conditions
    :param max_time: maximum time to render in secs
    :param moon_mass: mass of body 2
    :param is_sun: boolean to factor in sun's G-field
    :returns: system variables (x,y,v_x,v_y) for body 1 and 2, and time

    """
    time = np.arange(0, max_time, global_vars["step_size"])
    x_pos_1, y_pos_1 = np.zeros_like(time, dtype=np.float_), np.zeros_like(
        time, dtype=np.float_
    )
    x_vel_1, y_vel_1 = np.zeros_like(time), np.zeros_like(time)
    x_pos_2, y_pos_2 = np.zeros_like(time, dtype=np.float_), np.zeros_like(
        time, dtype=np.float_
    )
    x_vel_2, y_vel_2 = np.zeros_like(time), np.zeros_like(time)
    variables_1 = body_1_initial
    variables_2 = body_2_initial
    early_halt = False
    halt_idx = None
    for idx in range(np.size(time)):
        x_pos_1[idx], y_pos_1[idx] = variables_1[0], variables_1[1]
        x_vel_1[idx], y_vel_1[idx] = variables_1[2], variables_1[3]
        k_1, k_2, k_3, k_4 = get_all_ks(
            global_vars, variables_1, variables_2, False, is_sun, 2, body_2_mass
        )
        variables_temp_1 = solve_rk4(global_vars, k_1, k_2, k_3, k_4, dc(variables_1))
        x_pos_2[idx], y_pos_2[idx] = variables_2[0], variables_2[1]
        x_vel_2[idx], y_vel_2[idx] = variables_2[2], variables_2[3]
        k_1, k_2, k_3, k_4 = get_all_ks(
            global_vars, variables_2, variables_1, True, is_sun, 2, body_2_mass
        )
        variables_temp_2 = solve_rk4(global_vars, k_1, k_2, k_3, k_4, dc(variables_2))
        if idx != 0 and idx % 20 == 0:
            crash = calculate_distance(
                np.abs(variables_temp_1[0] - variables_temp_2[0]),
                np.abs(variables_temp_1[1] - variables_temp_2[1]),
            ) <= (body_2_radius + global_vars["radius_earth"])
            if crash:
                print("-> Warning: Collision detected before max time reached.")
                early_halt = True
                halt_idx = idx
                break
        variables_1 = dc(variables_temp_1)
        variables_2 = dc(variables_temp_2)
    if early_halt:
        x_pos_1, y_pos_1 = x_pos_1[:halt_idx], y_pos_1[:halt_idx]
        x_pos_2, y_pos_2 = x_pos_2[:halt_idx], y_pos_2[:halt_idx]
        x_vel_1, y_vel_1 = x_vel_1[:halt_idx], y_vel_1[:halt_idx]
        x_vel_2, y_vel_2 = x_vel_2[:halt_idx], y_vel_2[:halt_idx]
        time = time[:halt_idx]
    return (
        x_pos_1,
        y_pos_1,
        x_pos_2,
        y_pos_2,
        x_vel_1,
        y_vel_1,
        x_vel_2,
        y_vel_2,
        time,
    )


def check_energy_conservation_2body(
    global_vars: dict,
    x_pos_1: np.ndarray,
    y_pos_1: np.ndarray,
    x_pos_2: np.ndarray,
    y_pos_2: np.ndarray,
    x_vel_1: np.ndarray,
    y_vel_1: np.ndarray,
    x_vel_2: np.ndarray,
    y_vel_2: np.ndarray,
    mass_2: float,
    time: np.ndarray,
    is_display: bool,
    is_sun: bool,
) -> float:
    """Calculate PE, KE and total energy

    :param global_vars: global_variables
    :param x_pos: x positions
    :param y_pos: y positions
    :param x_vel: x velocity arrays
    :param y_vel: y velocity arrays
    :param is_display: boolean to show information
    :returns: the standard deviation in energy

    """
    energy_kin = (0.5 * global_vars["mass_earth"] * (x_vel_1**2 + y_vel_1**2)) + (
        0.5 * mass_2 * (x_vel_2**2 + y_vel_2**2)
    )
    distance_between_bodies = calculate_distance(
        np.abs(x_pos_1 - x_pos_2), np.abs(y_pos_1 - y_pos_2)
    )
    energy_pot = (
        -global_vars["grav_constant"]
        * global_vars["mass_earth"]
        * mass_2
        / distance_between_bodies
    )
    if is_sun:
        energy_pot += (
            -global_vars["grav_constant"]
            * global_vars["mass_sun"]
            * (
                global_vars["mass_earth"] / calculate_distance(x_pos_1, y_pos_1)
                + global_vars["mass_moon"] / calculate_distance(x_pos_2, y_pos_2)
            )
        )
    energy = energy_kin + energy_pot
    avg_energy = float(np.mean(energy))
    std_energy = float(np.std(energy))
    if is_display:
        print(
            "-> Average energy:",
            # round_with_decimal(3, avg_energy),  # TODO: this throws errors
            avg_energy,
            "J",
        )
        print(
            "-> Standard deviation in energy:",
            # round_with_decimal(3, std_energy),  # TODO: this throws errors
            std_energy,
            "J",
        )
        print("-> Showing total energy at each timestep")
        input("* Press anything to continue...")
        plot_energies(time, energy_kin, energy_pot, energy, False)
    return float(std_energy)


def plot_overview_orbit_2body(
    global_vars: dict,
    x_pos_1: np.ndarray,
    y_pos_1: np.ndarray,
    x_pos_2: np.ndarray,
    y_pos_2: np.ndarray,
    is_sun: bool,
):
    """Plot the trajectory of flight for 2 bodies

    :param global_vars: global variables
    :param x_pos: x position arrays
    :param y_pos: y_position arrays
    :param is_moon: boolean to factor in moon's G-field

    """
    _, axi = plt.subplots(figsize=(9, 9))
    xmax = np.max(
        [
            np.abs(np.min(x_pos_1)),
            np.abs(np.min(x_pos_2)),
            np.max(x_pos_1),
            np.max(x_pos_2),
        ]
    )
    ymax = np.max(
        [
            np.abs(np.min(y_pos_1)),
            np.abs(np.min(y_pos_2)),
            np.max(y_pos_1),
            np.max(y_pos_2),
        ]
    )
    if is_sun:
        sun = plt.Circle(
            (0, 0),
            radius=global_vars["radius_sun"] / global_vars["radius_earth"],
            color="red",
            edgecolor=None,
        )
        axi.add_patch(sun)  # type: ignore

    ani_scale = np.max([xmax, ymax])
    axi.set_xlabel(r"x distance ($R_E$)")  # type: ignore
    axi.set_ylabel(r"y distance ($R_E$)")  # type: ignore
    axi.set_xlim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
    axi.set_ylim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
    axi.plot(x_pos_1, y_pos_1, color="blue")  # type: ignore
    axi.plot(x_pos_2, y_pos_2, color="olive")  # type: ignore

    plt.show()


def animate_orbit_2body(
    global_vars: dict,
    x_pos_body_1: np.ndarray,
    y_pos_body_1: np.ndarray,
    x_pos_body_2: np.ndarray,
    y_pos_body_2: np.ndarray,
    radius_2: float,
    filename: str,
    is_sun: bool,
):
    """Animate 2 moving bodies

    :param global_vars: global variables,
    :param x_pos_body_1: x position array of body 1,
    :param y_pos_body_1: y position array of body 2 ,
    :param x_pos_body_2: x position array of body 1,
    :param y_pos_body_2: y position array of body 2,
    :param radius_2: radius of second body/moon,
    :param filename: filename used to save the animation,
    :param is_sun: boolean to factor in the sun's G-field,

    """
    fig, axi = plt.subplots(figsize=(9, 9))

    ani_scale = int(
        take_input_int("animation scale (square of inputted value) (default: 0)", False)
        or "0"
    )

    def animate(i, radius_2):
        axi.clear()  # type: ignore
        x_1, y_1 = x_pos_body_1[:i], y_pos_body_1[:i]
        x_2, y_2 = x_pos_body_2[:i], y_pos_body_2[:i]
        axi.plot(x_1, y_1, color="blue", lw=1, alpha=1)  # type: ignore
        axi.plot(x_2, y_2, color="olive", lw=1, alpha=1)  # type: ignore

        if is_sun:
            patches = [
                plt.Circle(
                    (0, 0),
                    radius=global_vars["radius_sun"] / global_vars["radius_earth"],
                    color="red",
                    edgecolor=None,
                )
            ]
        else:
            patches = [
                plt.Circle(
                    (x_pos_body_1[i], y_pos_body_1[i]),
                    radius=1,
                    color="blue",
                    edgecolor=None,
                ),
                plt.Circle(
                    (x_pos_body_2[i], y_pos_body_2[i]),
                    radius=radius_2,
                    color="olive",
                    edgecolor=None,
                ),
            ]

        collection = clt.PatchCollection(patches, match_original=True)
        axi.add_collection(collection)  # type: ignore
        axi.set_xlim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
        axi.set_ylim([-1.1 * ani_scale, ani_scale * 1.1])  # type: ignore
        axi.set_xlabel(r"x distance ($R_E$)")  # type: ignore
        axi.set_ylabel(r"y distance ($R_E$)")  # type: ignore

    ani = anim.FuncAnimation(
        fig, animate, frames=len(x_pos_body_1) - 1, interval=1, fargs=[radius_2]
    )
    plt.show()
    is_save = bool(
        take_input_bool("Do you want to save a gif? True/False (default: False)")
    )
    if is_save:
        if len(x_pos_body_1) - 1 > global_vars["gif_max_frames"] + 1:
            print(
                f"! {len(x_pos_body_1) - 1} frames in total.",
                f"Too many frames (>{global_vars['gif_max_frames']}).",
            )
        else:
            print("-> Saving gif...")
            writergif = anim.PillowWriter(fps=30)
            ani.save(f"{filename}.gif", writer=writergif)


######################
#  HELPER FUNCTIONS  #
######################


def calculate_distance(x_pos, y_pos):
    """Calculate 2D distance using Pythagoras theorem

    :param x_pos: x position arrays
    :param y_pos: y position arrays
    :returns: sqrt(x^2 + y^2)

    """
    return np.sqrt(x_pos**2 + y_pos**2)


def print_useful_values(global_vars: dict):
    """Print values that are useful for parameter inputs

    :param global_vars: global variables

    """
    print(
        "-> Useful values for inputs:",
        "\n\t -> Moon_radius/earth_radius =",
        round_with_decimal(2, global_vars["radius_moon"] / global_vars["radius_earth"]),
        "\n\t -> Moon_mass/earth_mass =",
        round_with_decimal(4, global_vars["mass_moon"] / global_vars["mass_earth"]),
        "\n",
    )


def take_input_int(var_name: str, is_signed: bool) -> int | str:
    """Take in a value as an input with error catching. These should be integers
    Except in the case where the input is empty, then use default value implemented
    in the parent function

    :param var_name: a string name for variable
    :param is_signed: boolean to toggle both +/- inputs

    :returns: a positive integer

    """
    input_var = input(f"+ Enter a value for {var_name}: ")
    while True:
        if input_var == "":
            return input_var
        try:
            var = int(input_var)
            if not is_signed:
                assert var > 0
            break
        except ValueError:
            input_var = input(f"! Please enter a valid value for {var_name}: ")
        except AssertionError:
            input_var = input(f"! Please enter a positive value for {var_name}: ")
    return var


def take_input_complex(var_name: str) -> complex | str:
    """Take in a value as an input with error catching These should be complex
    Except in the case where the input is empty, then use default value implemented
    in the parent function

    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"+ Enter a value for {var_name}: ")
    while True:
        if input_var == "":
            return input_var
        try:
            var = complex(input_var)
            break
        except ValueError:
            input_var = input(f"! Please enter a valid value for {var_name}: ")
    return var


def take_input_float(var_name: str) -> float | str:
    """Take in a value as an input with error catching These should be floats
    Except in the case where the input is empty, then use default value implemented
    in the parent function

    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"+ Enter a value for {var_name}: ")
    while True:
        if input_var == "":
            return input_var
        try:
            var = float(input_var)
            break
        except ValueError:
            input_var = input(f"! Please enter a valid value for {var_name}: ")
    return var


def take_input_bool(var_name: str) -> bool | str:
    """Take in a value as an input with error catching These should be booleans
    Except in the case where the input is empty, then use default value implemented
    in the parent function

    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"+ {var_name}: ")
    var = ""
    while True:
        if input_var == "":
            return input_var
        if input_var == "True":
            var = True
            break
        if input_var == "False":
            var = False
            break
        input_var = input(f"! Please enter a valid value for {var_name}: ")
        break
    return var


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
    :returns: the index

    """
    return int(round_with_decimal(0, (value - start) / step))


########################
#  PLOTTING FUNCTIONS  #
########################


def plot_energy_conservation(step_sizes: np.ndarray, errors: np.ndarray):
    """Plot step sizes vs. errors in linear and log plot

    :param step_sizes: step sizes array
    :param errors: errors array

    """
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(step_sizes, errors, s=2, color="k")
    ax1.set_xlabel("Step sizes (s)")
    ax1.set_ylabel("Standard deviation in energy per unit rocket mass (J/kg)")
    log_x = np.log(step_sizes)
    log_y = np.log(errors)
    grad, _ = np.polyfit(log_x, log_y, 1)
    ax2.loglog(step_sizes, errors, color="k")
    ax2.set_title(f"Gradient: {grad}")
    ax2.set_xlabel("log Step sizes (s)")
    ax2.set_ylabel("log Standard deviation in energy per unit rocket mass (J/kg)")
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def plot_rocket_earth_moon(
    time: np.ndarray, distance_from_earth: np.ndarray, distance_from_moon: np.ndarray
):
    """Plot the distance between rocket & moon and rocket & earth

    :param time: time array
    :param earth: distance from earth array
    :param moon: distance from moon array

    """
    plt.scatter(
        time, distance_from_earth, color="blue", label="distance from earth", s=2
    )
    plt.scatter(
        time, distance_from_moon, color="olive", label="distance from moon", s=2
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.show()


def plot_sun_earth_moon(
    time: np.ndarray,
    earth_sun_distance: np.ndarray,
    moon_sun_distance: np.ndarray,
    earth_moon_distance: np.ndarray,
):
    """Plot the distance between earth & sun, moon & sun, earth & moon

    :param time: time array
    :param earth_sun_distance: distance between earth & sun array
    :param moon_sun_distance: distance between moon & sun array
    :param earth_moon_distance: distance between earth & moon array

    """
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(time, earth_sun_distance, color="blue", label="earth", s=2)
    ax1.scatter(time, moon_sun_distance, color="olive", label="moon", s=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.legend()
    ax2.scatter(time, earth_moon_distance, color="black", s=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance (m)")
    plt.show()


def plot_energies(
    time: np.ndarray,
    energy_kin: np.ndarray,
    energy_pot: np.ndarray,
    energy: np.ndarray,
    is_rocket: bool,
):
    """Plot the PE, KE and total energy of the rocket

    :param time: time array
    :param energy_kin: kinetic energy array
    :param energy_pot: potential energy array
    :param energy: total energy array

    """
    y_label = "Energy (J)"
    if is_rocket:
        y_label = r"Energy/$m_{rocket}$ (J/kg)"
    plt.scatter(time, energy_kin, label="Kinetic energy", s=2)
    plt.scatter(time, energy_pot, label="Potential energy", s=2)
    plt.scatter(time, energy, label="Total energy", s=2)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


##########
#  MAIN  #
##########


def main():
    """Driver code"""
    user_input = "0"
    global_vars = {
        "gif_max_frames": 700,
        "grav_constant": 6.67e-11,
        "mass_earth": 5.97e24,
        "mass_sun": 1.99e30,
        "mass_moon": 7.34e22,
        "radius_sun": 6.96e8,
        "radius_earth": 6.38e6,
        "radius_moon": 1.74e6,
        "step_size": 40,
        "earth_moon_distance": 3.84e8,
        "earth_sun_distance": 1.5e11,
    }
    # Uncomment for bigger plot fonts (for report only)
    # plt.rcParams.update({"font.size": 20})
    while user_input != "q":
        print(
            "---------------------------------------------------------"
            "\n(a) Launch rocket with fixed earth",
            "\n(b) Flyby with fixed earth & moon",
            "\n(c) Interaction between two moving bodies",
            "\n(d) Interaction between two moving bodies and a fixed body",
            "\n---------------------------------------------------------",
        )
        user_input = input('+ Enter a choice, "a", "b", "c", "d", or "q" to quit: ')
        print("-> You entered the choice:", user_input)
        if user_input == "a":
            run_option_a(dc(global_vars))
        elif user_input == "b":
            run_option_b(dc(global_vars))
        elif user_input == "c":
            run_option_c(dc(global_vars))
        elif user_input == "d":
            run_option_d(dc(global_vars))
        elif user_input != "q":
            print("! This is not a valid choice.")
    print("-> You have chosen to finish - goodbye.")


if __name__ == "__main__":
    main()
