#!/usr/bin/env python3
"""
Exercise 1
"""
from decimal import Decimal
import numpy as np
from numpy.polynomial import polynomial as ply


def find_arctan(x: float, N: int) -> float:
    """Calculate the arctan of x using numerical methods. Takes in

    :param x: the value to find arctan of
    :param N: number of iterations

    """
    if is_magnitude_in_range_1(x):
        return sum_to_arctan(x, N)
    return use_other_arctan(x, N)


def is_magnitude_in_range_1(x: float) -> bool:
    """
    Checks the magnitude of x

    :param x: the value to find arctan of
    :returns: a boolean
    """
    return np.abs(x) <= 1


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
    """Generate values for generate_table()

    :returns: a list of values

    """
    increment = 0.1
    return np.arange(-2, 2 + increment, increment)


def generate_table(N: int):
    """Find the arctan of x_values and compare it to built-in/numpy methods, and print out a table

    :param N: the number of iterations

    """
    x_values = generate_values()
    orig = np.zeros(len(x_values))
    comp = np.zeros(len(x_values))
    diff = np.zeros(len(x_values))
    for idx, val in enumerate(x_values):
        orig[idx], comp[idx] = find_arctan(val, N), np.arctan(val)
        diff[idx] = np.abs(orig[idx] - comp[idx])
    print("Value\tApproximation\t\tBuilt-in\t\tDifference")
    padding = "\t"
    for idx, val in enumerate(x_values):
        print(
            f"{Decimal(val).quantize(Decimal('1.00'))}",
            padding,
            f"{orig[idx]:.16f}",
            padding,
            f"{comp[idx]:.16f}",
            padding,
            f"{diff[idx]:.16e}",
        )


def take_input_n() -> int:
    """Take in N as an input with error catching

    :returns: a positive integer

    """
    input_n = input("Enter a value for N (positive integer): ")
    while True:
        try:
            N = int(input_n)
            assert N > 0
            break
        except Exception:
            input_n = input("Please enter a positive integer: ")
    return N


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
    print("You have chosen part (a)")
    x = take_input_general("x")
    N = take_input_n()
    print(f"The answer is {find_arctan(x, N)}")


def run_option_b():
    """Run option b"""
    print("You have chosen part (b)")
    N = take_input_n()
    generate_table(N)


def run_option_c():
    """Run option c"""
    print("You have chosen part (c)")
    N = 742718 # I just kept adjusting until I get the desired difference
    calc_pi = 4 * find_arctan(1, N)
    diff = abs(np.pi - calc_pi)
    print(
        f"The answer with N = {N}\nApproximated value = {calc_pi}",
        f"\nActual value = {np.pi}\nDifference = {diff}",
    )


def run_option_d():
    """Run option d"""
    print("You have chosen part (d)")
    N = 17
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
    print("You have chosen part (e)")
    a_0 = take_input_general("a0")
    a_1 = take_input_general("a1")
    a_2 = take_input_general("a2")
    a_3 = take_input_general("a3")
    a_4 = take_input_general("a4")
    x_0 = take_input_general("x_0 (initial guess)")
    delta = take_input_general("delta")
    print(
        f"The answer is {iterate_newton_raphson([a_0, a_1, a_2, a_3, a_4], x_0, delta)[0]}"
    )


def run_option_f():
    """Run option f"""
    print("You have chosen part (f)")
    a_0 = take_input_general("a0")
    a_1 = take_input_general("a1")
    a_2 = take_input_general("a2")
    a_3 = take_input_general("a3")
    a_4 = take_input_general("a4")
    x_min = take_input_general("minimum x value")
    x_max = take_input_general("maximum x value")
    increment = take_input_general("value to increment x_0 by")
    answers = find_many_roots([a_0, a_1, a_2, a_3, a_4], x_min, x_max, increment)
    print("Initial guess\t\tRoot found\t\tNo. of iteration")
    for line in answers:
        print("\t\t".join(line))


def iterate_newton_raphson(variables: list, x_prev: float, delta: float) -> tuple:
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
        x_next = x_prev - funct(x_prev) / funct.deriv(1)(x_prev)
        iteration += 1
        if np.abs(x_prev - x_next) <= delta:
            return (float(f"{x_next:.15g}"), iteration)
        max_iterations -= 1
        if max_iterations == 0:
            return (None, None)
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
    x_val = x_min
    roots = []
    delta = 0.000009
    while True:
        if x_val > x_max:
            break
        answer = iterate_newton_raphson(variables, x_val, delta)
        if answer[0] is not None:
            roots.append(
                [
                    f"{x_val:.6f}",
                    str(Decimal(answer[0]).quantize(Decimal("1.000000000000"))).ljust(
                        16
                    ),
                    str(answer[1]),
                ]
            )
        elif answer[0] is None:
            roots.append([f"{x_val:.6f}", answer[0], answer[1]])
        x_val += increment
        x_val = float(Decimal(x_val).quantize(Decimal("1.000")))
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
    )
    print(
        "(f) to use the Newton-Raphson method to approximate roots of a polynomial, takes in 5",
        "\n\tcoefficients a0 -> a4, min and max values for initial guesses, and a value to",
        "\n\tincrement the guesses by",
    )


def main():
    """Driver code"""
    user_input = "0"

    while user_input != "q":
        user_input = input(
            'Enter a choice, "a", "b", "c", "d", "e", "f", "h" for help, or "q" to quit: '
        )
        print("You entered the choice: ", user_input)

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

        elif user_input == "h":
            run_option_h()

        elif user_input != "q":
            print("This is not a valid choice.")

    print(" You have chosen to finish - goodbye.")


if __name__ == "__main__":
    main()
