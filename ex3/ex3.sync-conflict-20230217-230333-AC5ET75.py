#!/usr/bin/env python3
# vim:set fdm=indent:
"""
Exercise 2
"""
from decimal import Decimal
from copy import deepcopy as dc
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import scipy.signal as ss

#################################
#  OPTION INITIATION FUNCTIONS  #
#################################


def run_option_a(global_vars: dict):
    """Execute part a

    :param global_vars: global variables

    """
    print(
        "Creating diffraction pattern for the following parameters:",
        f"\n\tWavelength: {global_vars['wavelength']} m",
        f"\n\tAperture width: {global_vars['aperture_width']} m",
        f"\n\tScreen distance: {global_vars['screen_distance']} m",
        f"\n\tScreen coordinates: {global_vars['screen_coords']} m",
        f"\n\tPoints for Simpson's method: {global_vars['points_simpson']}",
    )
    input("Press anything to continue...")
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    intensity_simpson, intensity_quadrature = get_intensity(
        global_vars,
        x_vals,
        get_x_prime_vals(
            global_vars["aperture_width"],
            global_vars["points_simpson"],
        ),
        True,
        True,
    )
    plot_1d_diffraction_1(intensity_simpson, intensity_quadrature, x_vals)


def run_option_b(global_vars: dict):
    """Execute part b

    :param global_vars: global variables

    """
    print(
        "Changing aperture width & calculating effects",
        "on maximum intensity & width of central maximum",
    )
    input("Press anything to continue...")
    ap_widths, max_intensity, central_width = vary_aperture(global_vars)
    plot_far_field_effects(ap_widths, max_intensity, central_width, "aperture width")

    print(
        "Changing screen distance & calculating effects",
        "on maximum intensity & width of central maximum",
    )
    input("Press anything to continue...")
    screen_distances, max_intensity, central_width = vary_screen_distance(global_vars)
    plot_far_field_effects(
        screen_distances, max_intensity, central_width, "screen distance"
    )

    global_vars["screen_coords"] = (-30e-3, 30e-3)
    global_vars["aperture_width"] = 2e-3
    global_vars["screen_distance"] = 3e-3
    global_vars["points_simpson"] = 101
    print(
        "Showing near-field effects",
        f"\n\tExtended screen size: {global_vars['screen_coords']} m",
        f"\n\tNew aperture width: {global_vars['aperture_width']} m",
        f"\n\tNew screen distance: {global_vars['screen_distance']} m",
        f"\n\tPoints for Simpson's method: {global_vars['points_simpson']}",
    )
    input("Press anything to continue...")
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    intensity_nf, _ = get_intensity(
        global_vars,
        x_vals,
        get_x_prime_vals(
            global_vars["aperture_width"],
            global_vars["points_simpson"],
        ),
        True,
        False,
    )
    plot_1d_diffraction_2(x_vals, intensity_nf)

    global_vars["aperture_width"] = 2e-5
    global_vars["screen_distance"] = 2e-2
    print(
        "Examine the effect of varying number of points for Simpson's method",
        "for far-field diffraction",
        f"\n\tExtended screen size: {global_vars['screen_coords']} m",
        f"\n\tDefault aperture width: {global_vars['aperture_width']} m",
        f"\n\tDefault screen distance: {global_vars['screen_distance']} m",
    )
    input("Press anything to continue...")
    vary_plot_points(global_vars)

    global_vars["aperture_width"] = 2e-3
    global_vars["screen_distance"] = 3e-3
    print(
        "Examine the effect of varying number of points for Simpson's method",
        "for near-field diffraction",
        f"\n\tExtended screen size: {global_vars['screen_coords']} m",
        f"\n\tNew aperture width: {global_vars['aperture_width']} m",
        f"\n\tNew screen distance: {global_vars['screen_distance']} m",
    )
    input("Press anything to continue...")
    vary_plot_points(global_vars)


def run_option_c(global_vars: dict):
    """Execute part c

    :param global_vars: global variables

    """
    print("Creating a 2D diffraction pattern with a square aperture")
    input("Press anything to continue...")
    global_vars["points_screen"] = 100
    global_vars["screen_coords"] = (-5e-5, 5e-5)
    global_vars["aperture_width"] = 2e-5
    # distances = np.array([25,30])*1e-5
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    y_scale = float(
        take_input_general(
            "aperture y dimension relative to x dimension (default: 1 for square aperture)"
        )
        or "1"
    )
    y_vals = get_x_vals(
        (
            global_vars["screen_coords"][0] * y_scale,
            global_vars["screen_coords"][1] * y_scale,
        ),
        global_vars["points_screen"],
    )
    # for val in distances:
    #     global_vars["screen_distance"] = val
    #     intensity = get_intensity_2d(
    #         global_vars,
    #         x_vals,
    #         y_vals,
    #         lambda y: -global_vars["aperture_width"] / 2,
    #         lambda y: global_vars["aperture_width"] / 2,
    #     )
    #     plot_2d_diffraction(global_vars, intensity)
    global_vars["screen_distance"] = 20e-5
    intensity = get_intensity_2d(
        global_vars,
        x_vals,
        y_vals,
        lambda y: -global_vars["aperture_width"] / 2,
        lambda y: global_vars["aperture_width"] / 2,
    )
    plot_2d_diffraction(global_vars, intensity)


def run_option_d(global_vars: dict):
    """Execute option d

    :param global_vars: global variables

    """
    print("Creating a 2D diffraction pattern with a circular aperture")
    input("Press anything to continue...")
    global_vars["points_screen"] = 100
    global_vars["screen_coords"] = (-5e-5, 5e-5)
    global_vars["aperture_width"] = 2e-5
    distances = np.array([25,30])*1e-5
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    y_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    for val in distances:
        global_vars["screen_distance"] = val
        intensity = get_intensity_2d(
            global_vars,
            x_vals,
            y_vals,
            lambda y: -np.sqrt((global_vars["aperture_width"] / 2) ** 2 - y**2),
            lambda y: np.sqrt((global_vars["aperture_width"] / 2) ** 2 - y**2),
        )
        plot_2d_diffraction(global_vars, intensity)

    # get_intensity_2d(
    #     global_vars,
    #     x_vals,
    #     y_vals,
    #     lambda y: -np.sqrt((global_vars["aperture_width"] / 2) ** 2 - y**2),
    #     lambda y: np.sqrt((global_vars["aperture_width"] / 2) ** 2 - y**2),
    # )


def run_option_e(global_vars: dict):
    """Execute option e

    :param global_vars: global variables

    """
    print("Evaluating the area of a circle using the Monte Carlo method.")
    radius = float(take_input_general("circle radius (default: 1)") or "1")
    min_size = [-radius, -radius]
    max_size = [radius, radius]
    sample = np.random.uniform(
        low=min_size, high=max_size, size=(global_vars["mc_sample_size"], 2)
    )
    input("Press anything to continue...")
    result, error = evaluate_monte(sample, radius, dimension=2)
    print(
        f"For N={global_vars['mc_sample_size']}, the area is:",
        result,
        "with error:",
        error,
    )


def run_option_f():
    """Execute option f

    :param global_vars: global variables

    """
    print(
        "Investigating the effect of the number of samples on the area",
        "of a unit circle using the Monte Carlo method.",
    )
    input("Press anything to continue...")
    radius = 1
    min_size, max_size = -radius, radius
    sample_sizes = np.arange(50, 100000, 100)
    errors, results = np.zeros_like(sample_sizes, dtype=np.float_), np.zeros_like(
        sample_sizes, dtype=np.float_
    )
    for idx, sample_size_var in enumerate(sample_sizes):
        sample = np.random.uniform(
            low=min_size, high=max_size, size=(sample_size_var, 2)
        )
        results[idx], errors[idx] = evaluate_monte(sample, radius, dimension=2)
    plot_part_f(errors, results, sample_sizes, np.pi, dimension=2)


def run_option_g():
    """Execute option g

    :param global_vars: global variables

    """

    def evaluate_volumes(
        prev_prev_volume: float, radius: float, dimension: int
    ) -> float:
        return 2 * np.pi * radius**2 / dimension * prev_prev_volume

    radius = 1
    min_size, max_size = -radius, radius
    sample_sizes = np.arange(50, 100000, 100)
    dimensions = np.arange(2, 11, 1, dtype=np.int_)
    volumes = np.zeros_like(dimensions, dtype=np.float_)
    volumes[0] = np.pi * radius**2
    volumes[1] = 4 * np.pi * radius**3 / 3
    for idx, dimension_val in enumerate(dimensions):
        print(f"Generating plot for hypersphere dimension {dimension_val}...")
        errors, results = np.zeros_like(sample_sizes, dtype=np.float_), np.zeros_like(
            sample_sizes, dtype=np.float_
        )
        volumes[idx] = volumes[idx] * (dimension_val in [2, 3]) + evaluate_volumes(
            volumes[idx - 2], radius, dimension_val
        ) * (dimension_val not in [2, 3])
        if dimension_val == 10:
            for idx2, sample_size_var in enumerate(sample_sizes):
                sample = np.random.uniform(
                    low=min_size, high=max_size, size=(sample_size_var, dimension_val)
                )
                results[idx2], errors[idx2] = evaluate_monte(
                    sample, radius, dimension_val
                )
            plot_part_f(errors, results, sample_sizes, volumes[idx], dimension_val)


##############################
#  PROBLEM 1 MAIN FUNCTIONS  #
##############################


def vary_plot_points(global_vars: dict):
    """Vary points for Simpson's method & observe effects with plots

    :param global_vars: global variables

    """
    points = [3, 7, 11, 31, 51, 101]
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    for point in points:
        x_prime_vals = get_x_prime_vals(
            global_vars["aperture_width"],
            point,
        )
        intensity_simpson, _ = get_intensity(
            global_vars,
            x_vals,
            x_prime_vals,
            True,
            False,
        )
        plt.scatter(x_vals, intensity_simpson, s=1, label=point)
    plt.legend(
        markerscale=4.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        fancybox=True,
    )
    plt.xlabel("Screen coordinate (m)")
    plt.ylabel("Intensity (arbitrary)")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def vary_aperture(global_vars: dict) -> tuple:
    """Vary aperture width & investigate effects

    :param global_vars: global variables

    """
    ap_widths = np.linspace(1e-5, 4e-5, 100)
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    max_intensity, central_width = np.zeros_like(ap_widths), np.zeros_like(ap_widths)
    for idx, ap_val in enumerate(ap_widths):
        intensity_simpson, _ = get_intensity(
            global_vars,
            x_vals,
            get_x_prime_vals(
                ap_val,
                global_vars["points_simpson"],
            ),
            True,
            False,
        )
        max_intensity[idx] = np.max(intensity_simpson)
        central_width[idx] = (
            2
            * x_vals[
                global_vars["points_screen"] // 2
                - 1
                + ss.argrelextrema(
                    intensity_simpson[global_vars["points_screen"] // 2 :], np.less
                )[0][0]
            ]
        )
    return ap_widths, max_intensity, central_width


def vary_screen_distance(global_vars: dict) -> tuple:
    """Vary screen distance & observe effects

    :global_vars: global variables

    """
    screen_distances = np.linspace(5e-3, 9e-2, 200)
    x_vals = get_x_vals(
        global_vars["screen_coords"],
        global_vars["points_screen"],
    )
    x_prime_vals = get_x_prime_vals(
        global_vars["aperture_width"],
        global_vars["points_simpson"],
    )
    max_intensity, central_width = np.zeros_like(screen_distances), np.zeros_like(
        screen_distances
    )
    for idx, dist_val in enumerate(screen_distances):
        global_vars["screen_distance"] = dist_val
        intensity_simpson, _ = get_intensity(
            global_vars,
            x_vals,
            x_prime_vals,
            True,
            False,
        )
        max_intensity[idx] = np.max(intensity_simpson)
        central_width[idx] = (
            2
            * x_vals[
                global_vars["points_screen"] // 2
                - 1
                + ss.argrelextrema(
                    intensity_simpson[global_vars["points_screen"] // 2 :], np.less
                )[0][0]
            ]
        )
    return screen_distances, max_intensity, central_width


def get_intensity(
    global_vars: dict,
    x_vals: np.ndarray,
    x_prime_vals: np.ndarray,
    is_simp: bool,
    is_quad: bool,
) -> tuple:
    """Get the intensity for 1d aperture using either method

    :param global_vars: global variables
    :param x_vals: screen coordinates
    :param x_prime_vals: aperture coordinates
    :param is_simp: boolean to execute Simpsons' method
    :param is_quad: boolean to execute Quad method
    :returns: tuple of two ndarray (simp,quad)

    """
    e_field_vals_simpson, e_field_vals_quadrature = np.zeros_like(
        x_vals, dtype=np.complex_
    ), np.zeros_like(x_vals, dtype=np.complex_)
    intensity_simpson, intensity_quadrature = np.zeros_like(x_vals), np.zeros_like(
        x_vals
    )
    if is_simp:
        for idx, x_val in enumerate(x_vals):
            e_field_vals_simpson[idx] = integrate_diffraction_simpson(
                global_vars, x_val, x_prime_vals
            )
        intensity_simpson = calculate_intensity(global_vars, e_field_vals_simpson)
    if is_quad:
        for idx, x_val in enumerate(x_vals):
            e_field_vals_quadrature[idx] = integrate_diffraction_quadrature(
                global_vars,
                x_val,
                -global_vars["aperture_width"] / 2,
                global_vars["aperture_width"] / 2,
            )
        intensity_quadrature = calculate_intensity(global_vars, e_field_vals_quadrature)
    return intensity_simpson, intensity_quadrature


def get_intensity_2d(
    global_vars: dict,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_lower_bound,
    x_upper_bound,
) -> np.ndarray:
    """Get intensity for 2 dimensional aperture using quad method

    :param global_vars: global variables
    :param x_vals: screen coordinates in x
    :param y_vals: screen coordinates in y
    :param x_lower_bound: lower bound in x
    :param x_upper_bound: upper bound in x
    :returns: 2d intensity array

    """
    e_field_vals_quadrature_2d = np.zeros(
        (np.size(x_vals), np.size(y_vals)), dtype=np.complex_
    )
    for idx, x_val in enumerate(x_vals):
        for idx2, y_val in enumerate(y_vals):
            e_field_vals_quadrature_2d[idx, idx2] = integrate_diffraction_dblquad(
                global_vars,
                x_val,
                y_val,
                x_lower_bound,
                x_upper_bound,
                (-global_vars["aperture_width"] / 2, global_vars["aperture_width"] / 2),
            )
    return calculate_intensity(global_vars, e_field_vals_quadrature_2d)


def get_x_vals(screen_coordinates: tuple, points: int) -> np.ndarray:
    """Get screen coordinates

    :param screen_coordinates: location on screen
    :param points: number of points for screen coordinate
    :returns: array of screen coordinates

    """
    return np.linspace(screen_coordinates[0], screen_coordinates[1], points)


def get_x_prime_vals(aperture_width: float, points: int) -> np.ndarray:
    """Get aperture coordinates

    :param aperture_width: width of aperture
    :param points: number of simpson points
    :returns: an array for aperture coordinates

    """
    return np.linspace(-aperture_width / 2, aperture_width / 2, points)


def integrate_diffraction_simpson(
    global_vars: dict, x_val: float, x_prime_vals: np.ndarray
) -> np.complex128:
    """
    Use Simpson's method to find the electric field at a given x coordinate
    on the screen. Takes two model arrays and gives the area.

    :param global_vars: global variables
    :param x_val: current x coordinate
    :param x_prime_vals: aperture coordinates
    :returns: a complex value of electric field at x

    """
    integrand = np.exp(
        1j
        * np.pi
        * (x_val - x_prime_vals) ** 2
        / (global_vars["wavelength"] * global_vars["screen_distance"])
    )
    return (global_vars["E0"] * si.simpson(integrand, x_prime_vals)) / (
        global_vars["wavelength"] * global_vars["screen_distance"]
    )


def integrate_diffraction_quadrature(
    global_vars: dict, x_val: float, lower_bound: float, upper_bound: float
) -> np.complex128:
    """Use the quadrature method to integrate for the electric field at x

    :param global_vars: global variables
    :param x_val: coordinate on screen
    :param lower_bound: one end of the aperture
    :param upper_bound: the other end of the aperture
    :returns: the electric field value at x

    """

    def real_integrand(x):
        return np.cos(
            np.pi
            * (x_val - x) ** 2
            / (global_vars["wavelength"] * global_vars["screen_distance"])
        )

    def imaginary_integrand(x):
        return np.sin(
            np.pi
            * (x_val - x) ** 2
            / (global_vars["wavelength"] * global_vars["screen_distance"])
        )

    real_integral, _ = si.quad(real_integrand, lower_bound, upper_bound)
    imaginary_integral, _ = si.quad(imaginary_integrand, lower_bound, upper_bound)
    return (
        global_vars["E0"] / (global_vars["wavelength"] * global_vars["screen_distance"])
    ) * (real_integral + 1j * imaginary_integral)


def integrate_diffraction_dblquad(
    global_vars: dict,
    x_val: float,
    y_val: float,
    x_bound_lower,
    x_bound_upper,
    y_bound: tuple,
) -> np.complex128:
    """Perform double integration using dblquad

    :param global_vars: global variables
    :param x_val: screen coordinates in x
    :param y_val: screen coordinates in y
    :param x_bound_lower: lower x bound (function or float)
    :param x_bound_upper: upper x bound (function or float)
    :param y_bound: y bounds as a tuple
    :returns: TODO

    """

    def real_integrand(x, y):
        return np.cos(
            np.pi
            * ((x_val - x) ** 2 + (y_val - y) ** 2)
            / (global_vars["wavelength"] * global_vars["screen_distance"])
        )

    def imaginary_integrand(x, y):
        return np.sin(
            np.pi
            * ((x_val - x) ** 2 + (y_val - y) ** 2)
            / (global_vars["wavelength"] * global_vars["screen_distance"])
        )

    real_integral = si.dblquad(
        real_integrand,
        y_bound[0],
        y_bound[1],
        x_bound_lower,
        x_bound_upper,
        epsrel=1e-12,
        epsabs=1e-12,
    )
    imaginary_integral = si.dblquad(
        imaginary_integrand, y_bound[0], y_bound[1], x_bound_lower, x_bound_upper
    )
    return (
        global_vars["E0"] / (global_vars["wavelength"] * global_vars["screen_distance"])
    ) * (real_integral[0] + 1j * imaginary_integral[0])


def calculate_intensity(global_vars: dict, e_field_vals: np.ndarray) -> np.ndarray:
    """Return the intensity given the electric field value

    :param global_vars: global variables
    :param e_field_val: complex electric field value
    :returns: intensity as a real float

    """
    return np.real(
        (
            global_vars["c"]
            * global_vars["epsilon_0"]
            * e_field_vals
            * np.conjugate(e_field_vals)
        )
    )


##############################
#  PROBLEM 2 MAIN FUNCTIONS  #
##############################


def evaluate_monte(sample: np.ndarray, radius: float, dimension: int) -> tuple:
    """Evaluate multidimensional equations using the Monte Carlo method

    :param sample: the sample array (random numbers)
    :param radius: radius of hypersphere
    :param dimension: dimension of hypersphere

    :returns: a tuple of result and error

    """

    def pythag_function(sample, radius):
        return np.sum(sample**2, axis=1) <= radius

    func = pythag_function(sample, radius).astype(int)
    integral = ((2 * radius) ** dimension) * np.mean(func)
    error = ((2 * radius) ** dimension) * np.std(func) / np.sqrt(np.size(sample) // 2)
    return integral, error


######################
#  HELPER FUNCTIONS  #
######################


def take_input_int(var_name: str) -> int | str:
    """Take in a value as an input with error catching. These should be integers
    Except in the case where the input is empty, then use default value implemented in the mother
    function

    :param var_name: a string name for variable

    :returns: a positive integer

    """
    input_var = input(f"Enter a value for {var_name}: ")
    while True:
        if input_var == "":
            return input_var
        try:
            var = int(input_var)
            # assert var > 0
            break
        except Exception:
            input_var = input(f"Please enter a valid value for {var_name}: ")
    return var


def take_input_complex(var_name: str) -> complex | str:
    """Take in a value as an input with error catching These should be complex
    Except in the case where the input is empty, then use default value implemented in the mother
    function

    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"Enter a value for {var_name}: ")
    while True:
        if input_var == "":
            return input_var
        try:
            var = complex(input_var)
            break
        except ValueError:
            input_var = input(f"Please enter a valid value for {var_name}: ")
    return var


def take_input_general(var_name: str) -> float | str:
    """Take in a value as an input with error catching These should be floating point numbers
    Except in the case where the input is empty, then use default value implemented in the mother
    function

    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"Enter a value for {var_name}: ")
    while True:
        if input_var == "":
            return input_var
        try:
            var = float(input_var)
            break
        except ValueError:
            input_var = input(f"Please enter a valid value for {var_name}: ")
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


def plot_1d_diffraction_1(
    intensity_simpson: np.ndarray, intensity_quadrature: np.ndarray, x_vals: np.ndarray
):
    """Plot intensity vs screen coordinate for 2 methods

    :param intensity_simpson: intensity array from simpsons method
    :param intensity_quadrature: intensity array from quadrature method
    :param x_vals: screen coordinate array

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("1D Fresnel diffraction")
    ax1.set(
        xlabel="Screen coordinate (m)",
        ylabel="Intensity (arbitrary)",
        title="Simpson's method",
    )
    ax2.set(
        xlabel="Screen coordinate (m)",
        ylabel="Intensity (arbitrary)",
        title="Quadrature method",
    )
    ax1.scatter(x_vals, intensity_simpson, s=2, c="k")
    ax2.scatter(x_vals, intensity_quadrature, s=2, c="k")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_1d_diffraction_2(x_vals: np.ndarray, intensity: np.ndarray):
    """Plot intensity vs screen coordinate for 1 methods

    :param x_vals: TODO
    :param intensity: TODO
    :returns: TODO

    """
    plt.xlabel("Screen coordinate (m)")
    plt.ylabel("Intensity (arbitrary)")
    plt.scatter(x_vals, intensity, s=2, c="k")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_far_field_effects(
    variable: np.ndarray,
    max_intensity: np.ndarray,
    central_width: np.ndarray,
    variable_name: str,
):
    """Plot variable vs maximum intensity and central width

    :param variable: the variable to plot on x-axis
    :param max_intensity: maximum intensity array
    :param central_width: centra width array
    :param variable_name: string of variable name

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Effects of changing {variable_name}")
    ax1.set(
        xlabel=f"{variable_name.capitalize()} (m)",
        ylabel="Intensity (arbitrary)",
        title="Peak intensity of central maxima",
    )
    ax2.set(
        xlabel=f"{variable_name.capitalize()} (m)",
        ylabel="Central peak width (m)",
        title="Width of central maxima",
    )
    ax1.scatter(variable, max_intensity, s=2, c="r")
    ax2.scatter(variable, central_width, s=2, c="b")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_2d_diffraction(global_vars: dict, intensity: np.ndarray):
    """TODO: Docstring for plot_2d_diffraction.

    :global_vars: TODO
    :intensity: TODO
    :returns: TODO

    """
    plt.imshow(intensity)
    plt.ylabel("Relative screen coordinate (vertical)")
    plt.xlabel("Relative screen coordinate (horizontal)")
    plt.title(
        f"Screen distance = {global_vars['screen_distance']} m; Aperture width = {global_vars['aperture_width']} m"
    )
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_part_f(
    errors: np.ndarray,
    results: np.ndarray,
    sample_sizes: np.ndarray,
    real_value: float,
    dimension: int,
):
    """Plot errors & results vs number of samples

    :param errors: errors array
    :param results: results array
    :param sample_sizes: sample array
    :param real_value: expected volume of hypersphere
    :param dimension: hypersphere dimension

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"The effect of the number of samples on results of the Monte Carlo method ({dimension}D)."
    )
    ax1.set(
        xlabel="Number of samples",
        ylabel="Area of circle (arbitrary)",
        title="Convergence of results",
    )
    ax2.set(
        xlabel="Number of samples",
        ylabel="Error (arbitrary)",
        title="Improvement of errors",
    )
    ax1.scatter(sample_sizes, results, s=2, c="r")
    ax1.plot(
        sample_sizes,
        np.full_like(sample_sizes, real_value, dtype=np.float_),
        c="k",
        alpha=0.5,
    )
    ax2.scatter(sample_sizes, errors, s=2, c="b")
    # ax2.plot(sample_sizes,1/np.sqrt(sample_sizes), c="k", alpha=0.5)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def main():
    """Driver code"""
    user_input = "0"
    global_vars = {
        "wavelength": 1e-6,
        "aperture_width": 2e-5,
        "screen_distance": 2e-2,
        "screen_coords": (-5e-3, 5e-3),
        "points_simpson": 101,
        "points_screen": 5000,
        "E0": 1,
        "c": 3e8,
        "epsilon_0": 8.85e-12,
        "mc_sample_size": 100000,
    }
    # plt.rcParams.update({"font.size": 12})
    while user_input != "q":
        user_input = input(
            'Enter a choice, "a", "b", "c", "d", "e", "f", "g", "h" for help, or "q" to quit: '
        )
        print("You entered the choice: ", user_input)
        print(f"You have chosen part ({user_input})")
        if user_input == "a":
            run_option_a(dc(global_vars))
        elif user_input == "b":
            run_option_b(dc(global_vars))
        elif user_input == "c":
            run_option_c(dc(global_vars))
        elif user_input == "d":
            run_option_d(dc(global_vars))
        elif user_input == "e":
            run_option_e(dc(global_vars))
        elif user_input == "f":
            run_option_f()
        elif user_input == "g":
            run_option_g()
        elif user_input != "q":
            print("This is not a valid choice.")
    print("You have chosen to finish - goodbye.")


if __name__ == "__main__":
    main()
