#!/usr/bin/env python3
"""
Exercise 1
"""
from decimal import Decimal
import numpy as np
from numpy.polynomial import polynomial as ply
import matplotlib.pyplot as plt


def find_arctan(x: float, N: int) -> float:
    """Calculate the arctan of x using numerical methods.

    :param x: the value to find arctan of
    :param N: number of iterations

    """
    if np.abs(x) <= 1:
        return sum_to_arctan(x, N)
    return use_other_arctan(x, N)


def sum_to_arctan(x: float, N: int) -> float:
    """
    Use Taylor series to expand and find arctan

    :param x: the value to find arctan of
    :param N: number of iterations

    """
    num = 0
    running_sum = 0
    for num in range(N + 1):
        running_sum += ((-1) ** num) * (x ** (2 * num + 1)) / (2 * num + 1)
    return running_sum


def use_other_arctan(x: float, N: int) -> float:
    """For values of x > |1|, we have to do some arithmetic first
    and we can also use sum_to_arctan()

    :param x: the value to find arctan of
    :param N: number of iterations

    """
    recip = -sum_to_arctan(1 / x, N)
    return (recip + np.pi / 2) if x > 0 else (recip - np.pi / 2)


def generate_values() -> np.ndarray:
    """Generate values in a range for generate_table()

    :returns: a list of values

    """
    increment = 0.1
    return np.arange(-2, 2 + increment, increment)


def generate_table(N: int, is_print: bool) -> float:
    """Find the arctan of x_values and compare it to built-in/numpy methods, and print out a table

    :param N: the number of iterations
    :returns: the average difference around x = 1

    """
    x_values = generate_values()
    approx = np.zeros(len(x_values))
    computer = np.zeros(len(x_values))
    diff = np.zeros(len(x_values))
    for idx, val in enumerate(x_values):
        approx[idx], computer[idx] = find_arctan(val, N), np.arctan(val)
        diff[idx] = np.abs(approx[idx] - computer[idx])
    if is_print:
        padding = "│".ljust(4)
        draw_boxes("top", 3, 19)
        print(
            padding
            + "Value (2dp)".ljust(16)
            + padding
            + "Approx (6dp)".ljust(16)
            + padding
            + "Built-in (6dp)".ljust(16)
            + padding
            + "Diff (6dp)".ljust(16)
            + padding
        )
        draw_boxes("mid", 3, 19)
        for idx, val in enumerate(x_values):
            print(
                padding
                + f"{round_with_decimal(2, val)}".ljust(16)
                + padding
                + f"{round_with_decimal(6, approx[idx])}".ljust(16)
                + padding
                + f"{round_with_decimal(6, computer[idx])}".ljust(16)
                + padding
                + f"{round_with_decimal(6, diff[idx])}".ljust(16)
                + padding,
            )
        draw_boxes("bot", 3, 19)
    return float(
        np.mean(diff)
    )  # most of the errors not at |x| = 1 are roughly zero anyway


def take_input_n() -> int:
    """Take in N as an input with error catching

    :returns: a positive integer

    """
    input_n = input("Enter a value for the number of iterations N (positive integer): ")
    while True:
        try:
            N = int(input_n)
            assert N > 0
            break
        except Exception:
            input_n = input("Please enter a positive integer: ")
    return N


def take_input_complex(var_name: str) -> complex:
    """Take in a value as an input with error catching These should be complex
    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"Enter a value for {var_name}: ")
    while True:
        try:
            var = complex(input_var)
            break
        except ValueError:
            input_var = input(f"Please enter a valid value for {var_name}: ")
    return var


def take_input_general(var_name: str) -> float:
    """Take in a value as an input with error catching These should be floating point numbers

    :param var_name: a string name for variable
    :returns: the inputted variable

    """
    input_var = input(f"Enter a value for {var_name}: ")
    while True:
        try:
            var = float(input_var)
            break
        except ValueError:
            input_var = input(f"Please enter a valid value for {var_name}: ")
    return var


def run_option_a():
    """Run option a"""
    print("Finding arctan(x) with N iterations.")
    _ = input("Press anything to continue...")
    x = take_input_general("x")
    N = take_input_n()
    print(
        f"The answer of arctan({x}) is {round_with_sigfigs(10,find_arctan(x, N))} to 10 s.f."
    )


def run_option_b():
    """Run option b"""
    print("Finding arctan(x) for -2 <= x <= 2.")
    _ = input("Press anything to continue...")
    N = take_input_n()
    _ = generate_table(N, True)
    diffs = np.zeros(20)
    print("Testing the precision for N from 1 to 20.")
    user_input = input(
        "Do you want to print a table of results each time? (y/N): ",
    ).lower()
    while user_input not in ["y", "n", ""]:
        user_input = input("Unknown option. Try again:")
    _ = input("Press anything to continue...")
    for num in range(1, 21):
        if user_input == "y":
            print(f"For {num} iterations")
        diffs[num - 1] = generate_table(num, user_input == "y")
    print("\nSummary for iterations and average difference around x = 1:")
    padding = "│".ljust(4)
    draw_boxes("top", 1, 19)
    print(
        padding
        + "Iterations".ljust(16)
        + padding
        + "Avg diff (6dp)".ljust(16)
        + padding
    )
    draw_boxes("mid", 1, 19)
    for idx, val in enumerate(diffs):
        print(
            padding
            + str(idx + 1).ljust(16)
            + padding
            + str(round_with_sigfigs(6, val)).ljust(16)
            + padding
        )
    draw_boxes("bot", 1, 19)
    _ = input("Now plotting a graph of this. Press anything to continue...")
    plt.scatter(list(map(str, list(range(1, 21)))), diffs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Difference between approximated and built-in methods")
    plt.show()


def run_option_c():
    """Run option c"""
    print(
        "Estimating pi = arctan(1)/4. 742718 iterations needed for 7 significant figures."
    )
    _ = input("Press anything to continue...")
    N = 742718  # I just kept adjusting until I got the desired difference
    calc_pi = 4 * find_arctan(1, N)
    diff = abs(np.pi - calc_pi)
    print(
        f"The answer with N = {N}\nApproximated value = {calc_pi}",
        f"\nActual value = {np.pi}\nDifference = {diff}",
    )


def run_option_d():
    """Run option d"""
    print(
        "Estimating pi using a more clever way. 17 iterations needed for 12 decimal places."
    )
    _ = input("Press anything to continue...")
    N = 17  # I just kept adjusting until I got the desired difference
    calc_pi = 4 * (
        find_arctan(1 / 2, N) + find_arctan(1 / 5, N) + find_arctan(1 / 8, N)
    )
    diff = abs(np.pi - calc_pi)
    print(
        f"The answer with N = {N}\nApproximated value = {calc_pi}",
        f"\nActual value = {np.pi}\nDifference = {diff}",
    )


def run_option_e():
    """Run option e"""
    print(
        "Root finding for 0-4th degree polynomials",
        "Format is a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0",
    )
    _ = input("Press anything to continue...")
    a_0 = take_input_general("a0")
    a_1 = take_input_general("a1")
    a_2 = take_input_general("a2")
    a_3 = take_input_general("a3")
    a_4 = take_input_general("a4")
    print(f"Polynomial chosen: {a_4}x^4 + {a_3}x^3 + {a_2}x^2 + {a_1}x + {a_0}")
    print(
        "Insert a guess to start the numerical method",
    )
    x_0 = take_input_complex("x_0 (initial guess)")
    print(
        "Insert a precision value to stop once | x_i - x_f | <= delta",
    )
    delta = take_input_general("delta")
    print(
        f"The answer is {iterate_newton_raphson([a_0, a_1, a_2, a_3, a_4], x_0, delta)[0]}"
    )


def draw_boxes(location: str, repeats: int, width: int):
    """Draw boxes using ascii characters

    :param location: top, mid or bot
    :param repeats: how many columns
    :param width: the width of each column

    """
    line, characters = "─", []
    if location == "top":
        characters = ["┌", "┬", "┐"]
    elif location == "mid":
        characters = ["├", "┼", "┤"]
    elif location == "bot":
        characters = ["└", "┴", "┘"]
    print(
        characters[0]
        + (line * width + characters[1]) * repeats
        + line * width
        + characters[2]
    )


def format_many_roots(answers: list):
    """Print table for multiple roots finding method

    :param answers: a list of answers

    """
    padding = "│".ljust(4)
    draw_boxes("top", 2, 35)
    print(
        padding
        + "Initial guess".ljust(32)
        + padding
        + "Root found (9 dp)".ljust(32)
        + padding
        + "No. of iteration".ljust(32)
        + padding
    )
    draw_boxes("mid", 2, 35)
    roots = []
    for line in answers:
        if line[1] is None:
            continue
        # round and collect all roots found
        val = complex(line[1])
        val_real = round_with_sigfigs(6, round_with_decimal(7, np.real(val)))
        val_imag = round_with_sigfigs(6, round_with_decimal(7, np.imag(val)))
        roots.append(complex(val_real, val_imag))
        # round table values for printing
        line[0] = complex(
            round_with_decimal(3, np.real(line[0])),
            round_with_decimal(3, np.imag(line[0])),
        )
        line[1] = complex(
            round_with_decimal(9, np.real(line[1])),
            round_with_decimal(9, np.imag(line[1])),
        )
        # convert all values to string type and add some right padding
        line = [val.strip("()").ljust(32) for val in list(map(str, line))]
        print(padding + f"{padding}".join(line) + padding)
    draw_boxes("bot", 2, 35)
    # convert all values of roots to string type and print
    print("\nSummary of roots found, to 6 significant figures")
    print("\t\t".join(list(map(str, set(roots)))))


def run_option_f():
    """Run option f"""
    print(
        "Multiple roots finding for 0-4th degree polynomials.",
        "\nFormat is a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0",
    )
    _ = input("Press anything to continue...")
    a_0 = take_input_general("a0")
    a_1 = take_input_general("a1")
    a_2 = take_input_general("a2")
    a_3 = take_input_general("a3")
    a_4 = take_input_general("a4")
    print(f"Polynomial chosen: {a_4}x^4 + {a_3}x^3 + {a_2}x^2 + {a_1}x + {a_0}")
    print(
        "Range for initial guesses, applies to both real and imaginary parts of guesses."
    )
    x_min = take_input_general("minimum x value")
    x_max = take_input_general("maximum x value")
    increment = take_input_general("value to increment x_0 by")
    print(
        f"Range picked: {x_min} + i*({x_min}...{x_max}) ... {x_max} + i({x_min}...{x_max})"
    )
    answers = find_many_roots([a_0, a_1, a_2, a_3, a_4], x_min, x_max, increment)
    format_many_roots(answers)


def run_option_g():
    """Run option g"""
    print("Testing for x^4 + x^3 - 12x^2 - 2x + 10")
    _ = input("Press anything to continue...")
    answers = find_many_roots([10, -2, -12, 1, 1], -4, 4, 0.8)
    format_many_roots(answers)


def round_with_decimal(decimal_places: int, value: float) -> float:
    """Round a float to the nearest dp provided without precision error
    using quantize() from Decimal class

    :param dp: number of decimal places
    :param value: the float to round
    :returns: the answer as a float

    """
    reference = "1." + "0" * decimal_places
    return float(Decimal(str(value)).quantize(Decimal(reference)))


def round_with_sigfigs(sig_figs: int, value: float) -> float:
    """Round a float to the nearest sf

    :param sig_figs: the number of significant figures
    :param value: the float to round
    :returns: the answer as a float

    """
    return float(f"{value:.{sig_figs}g}")


def iterate_newton_raphson(variables: list, x_prev: complex, delta: float) -> tuple:
    """Find the root of a polynomial using the Newton-Raphson method

    :param variables: a list of coefficients
    :param x_prev: first value x_0
    :param delta: the minimum accuracy
    :param is_alt: perform task (e) if false, (f) if true
    :returns: the root and the number of iterations performed

    """
    funct = ply.Polynomial(variables)
    x_next = 0
    max_iterations = 1000
    iteration = 0
    while True:
        deriv = funct.deriv(1)(x_prev)
        if deriv == 0 or max_iterations == 0:
            return (None, None)
        x_next = x_prev - funct(x_prev) / deriv
        iteration += 1
        if np.abs(x_prev - x_next) <= delta:
            return (complex(f"{x_next:.15g}"), iteration)
        max_iterations -= 1
        x_prev = x_next


def find_many_roots(
    variables: list, x_min: float, x_max: float, increment: float
) -> list:
    """Find many roots using Newton-Raphson

    :param variables: a list of coefficients
    :param x_min: minimum x_0 value
    :param x_max: maximum x_0 value
    :param increment: increment x_0 by this value
    :returns: a list of number

    """
    x_val_imag = x_min
    x_val_real = x_min
    roots = []
    delta = 0.000009
    # iterate through all complex values in range before going to the next real value
    while x_val_real <= x_max:
        while x_val_imag <= x_max:
            x_val = complex(x_val_real, x_val_imag)
            answer = iterate_newton_raphson(variables, x_val, delta)
            roots.append(
                [
                    x_val,
                    answer[0],
                    answer[1],
                ]
            )
            x_val_imag = round_with_decimal(3, x_val_imag + increment)
        x_val_imag = x_min
        x_val_real = round_with_decimal(3, x_val_real + increment)
    return roots


def run_option_h():
    """Run option h, which prints the help"""
    print("(a) find arctan of a number and with N number of iterations")
    print(
        "(b) print a table of values of arctan(x) for -2 <= x <= 2 using both approximation and",
        "\n\tbuilt-in method with N number of iterations",
    )
    print("(c) evaluate pi to 7 significant digits using naive method as used in (a)")
    print("(d) evaluate pi to 12 decimal places using another identity")
    print(
        "(e) to use the Newton-Raphson method to approximate a root of a polynomial, takes in 5",
        "\n\tcoefficients a0 -> a4, initial guess, and accuracy",
        "\n\tCan guess complex roots",
    )
    print(
        "(f) to use the Newton-Raphson method to approximate roots of a polynomial, takes in 5",
        "\n\tcoefficients a0 -> a4, min and max values for initial guesses, and a value to",
        "\n\tincrement the guesses by",
        "\n\tWork with complex roots",
    )
    print("(g) test method in (f) with x^4+x^3-12x^2-2x+10")


def main():
    """Driver code"""
    user_input = "0"
    while user_input != "q":
        user_input = input(
            'Enter a choice, "a", "b", "c", "d", "e", "f", "g", "h" for help, or "q" to quit: '
        )
        print("You entered the choice: ", user_input)
        print(f"You have chosen part ({user_input})")
        if user_input == "a":
            run_option_a()
        elif user_input == "b":
            run_option_b()
        elif user_input == "c":
            run_option_c()
        elif user_input == "d":
            run_option_d()
        elif user_input == "e":
            run_option_e()
        elif user_input == "f":
            run_option_f()
        elif user_input == "g":
            run_option_g()
        elif user_input == "h":
            run_option_h()
        elif user_input != "q":
            print("This is not a valid choice.")
    print("You have chosen to finish - goodbye.")


if __name__ == "__main__":
    main()
