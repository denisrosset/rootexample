from math import cos
from typing import Any, Callable, Mapping, Optional, Tuple


def inverse_quadratic_interpolation_step(
    a: float, b: float, c: float, fa: float, fb: float, fc: float
) -> float:
    """
    Computes an approximation for a zero of a 1D function from three function values

    Note:
        The values ``fa``, ``fb``, ``fc`` need all to be distinct.

    See `<https://en.wikipedia.org/wiki/Inverse_quadratic_interpolation>`_

    Args:
        a: First x coordinate
        b: Second x coordinate
        c: Third x coordinate
        fa: f(a)
        fb: f(b)
        fc: f(c)

    Returns:
        An approximation of the zero
    """
    L0 = (a * fb * fc) / ((fa - fb) * (fa - fc))
    L1 = (b * fa * fc) / ((fb - fa) * (fb - fc))
    L2 = (c * fb * fa) / ((fc - fa) * (fc - fb))
    return L0 + L1 + L2


def secant_step(a: float, b: float, fa: float, fb: float) -> float:
    """
    Computes an approximation for a zero of a 1D function from two function values

    Note:
        The values ``fa`` and ``fb`` need to have a different sign.

    Args:
        a: First x coordinate
        b: Second x coordinate
        fa: f(a)
        fb: f(b)

    Returns:
        An approximation of the zero
    """
    return b - fb * (b - a) / (fb - fa)


def bisection_step(a: float, b: float) -> float:
    """
    Computes an approximation for a zero of a 1D function from two function values

    Note:
        The values ``f(a)`` and ``f(b)`` (not needed in the code) need to have a different sign.

    Args:
        a: First x coordinate
        b: Second x coordinate

    Returns:
        An approximation of the zero
    """
    return min(a, b) + abs(b - a) / 2


def test_convergence(
    a: float, b: float, fa: float, fb: float, settings: Mapping[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Checks convergence of a root finding method

    Args:
        a: Contrapoint
        b: Best guess
        fa: f(a)
        fb: f(b)
        settings: Tolerance settings

    Returns:
        Whether the root finding converges to the specified tolerance and why
    """
    if b == 0:
        return (True, "Exact root found")
    x_delta = abs(a - b)
    if x_delta <= settings["x_abs_tol"]:
        return (True, "Met x_abs_tol criterion")
    if x_delta / max(a, b) <= settings["x_rel_tol"]:
        return (True, "Met x_rel_tol criterion")
    y_delta = abs(fa - fb)
    if y_delta <= settings["y_abs_tol"]:
        return (True, "Met y_abs_tol criterion")
    if y_delta / max(a, b) <= settings["y_rel_tol"]:
        return (True, "Met y_rel_tol criterion")
    return (False, None)


def init_state(f: Callable[[float], float], a: float, b: float) -> Mapping[str, Any]:
    """
    Initializes a state from an interval that brackets a root of the given function

    Args:
        f: Function to find the root of
        a: First x coordinate
        b: Second x coordinate

    Returns:
        The initial state
    """
    fa = f(a)
    fb = f(b)
    # check the first invariant
    assert fa * fb <= 0, "Root not bracketed"
    if abs(fa) < abs(fb):
        # force the second invariant
        b, a = a, b
        fb, fa = fa, fb
    c, fc = a, fa
    d, fd = a, fa
    return {
        "a": a,  # Contrapoint
        "b": b,  # Current iterate, best root approximation known so far
        "c": c,  # Previous iterate
        "d": d,  # Iterate before the previous iterate
        "fa": fa,  # f(a)
        "fb": fb,  # f(b)
        "fc": fc,  # f(c)
        "last_step": None,  # Type of previous step
        "iter": 1,  # Current iteration number (1-based)
    }


def brent_step(
    f: Callable[[float], float], state: Mapping[str, Any], delta: float
) -> Mapping[str, Any]:
    """
    Performs a step of Brent's method

    Note:
        The state has the following invariants.

        - ``fa = f(a)`` and ``fb = f(b)`` have opposite signs

        - ``abs(fb) <= abs(fa)`` so that ``b`` is the best guess

    Args:
        f: Function to find the root of
        state: Previous state
        delta: x absolute tolerance

    Returns:
        New state
    """
    a, b, c, d = state["a"], state["b"], state["c"], state["d"]
    fa, fb, fc = state["fa"], state["fb"], state["fc"]
    last_step = state["last_step"]
    iter = state["iter"]
    step: Optional[str] = None
    if fa != fc and fb != fc:
        s = inverse_quadratic_interpolation_step(a, b, c, fa, fb, fc)
        step = "quadratic"
    else:
        s = secant_step(a, b, fa, fb)
        step = "secant"
    perform_bisection = False
    if not ((3 * a + b) / 4 <= s and s <= b):
        perform_bisection = True
    elif last_step == "bisection" and abs(s - b) >= abs(b - c) / 2:
        perform_bisection = True
    elif last_step != "bisection" and abs(a - b) >= abs(c - d) / 2:
        perform_bisection = True
    elif last_step == "bisection" and abs(b - c) < delta:
        perform_bisection = True
    elif last_step != "bisection" and abs(c - d) < delta:
        perform_bisection = True
    if perform_bisection:
        s = bisection_step(a, b)
        step = "bisection"
    fs = f(s)
    d = c
    c = b
    fc = fb
    # check which point to replace to maintain (a,b) have different signs
    if f(a) * f(s) < 0:
        b = s
        fb = fs
    else:
        a = s
        fa = fs
    # keep b as the best guess
    if abs(fa) < abs(fb):
        b, a = a, b
        fb, fa = fa, fb
    return {
        "a": a,  # Contrapoint
        "b": b,  # Current iterate, best root approximation known so far
        "c": c,  # Previous iterate
        "d": d,  # Iterate before the previous iterate
        "fa": fa,  # f(a)
        "fb": fb,  # f(b)
        "fc": fc,  # f(c)
        "last_step": step,  # Type of previous step
        "iter": iter + 1,  # Current iteration number (1-based)
    }


def brent(f: Callable[[float], float], a: float, b: float, settings: Mapping[str, Any]) -> float:
    """
    Finds the root of a function using Brent's method, starting from an interval enclosing the zero
    Args:
        f: Function to find the root of
        a: First x coordinate enclosing the root
        b: Second x coordinate enclosing the root
        settings: Algorithm settings

    Returns:
        The approximate root
    """
    state = init_state(f, a, b)
    converged = None
    reason = None

    def print_state(s: Mapping[str, Any]):
        """
        Prints information about an iteration
        """
        dx = abs(s["a"] - s["b"])
        dy = abs(s["fa"] - s["fb"])
        ls: Optional[str] = s["last_step"]
        if ls is None:
            ls = ""
        print(f"{s['iter']}\t{s['b']:.3e}\t{s['fb']:.3e}\t{dx:.3e}\t{dy:.3e}\t{ls}")

    if settings["verbose"]:
        print("Iter\tx\t\tf(x)\t\tdelta(x)\tdelta(f(x))\tstep")
        print_state(state)

    while not converged:
        state = brent_step(f, state, settings["x_abs_tol"])
        converged, reason = test_convergence(
            state["a"], state["b"], state["fa"], state["fb"], settings
        )
        if settings["verbose"]:
            print_state(state)

    if settings["verbose"]:
        assert reason is not None
        print(reason)

    return state["b"]


my_settings = {
    "x_rel_tol": 1e-12,
    "x_abs_tol": 1e-12,
    "y_rel_tol": 1e-12,
    "y_abs_tol": 1e-12,
    "verbose": True,
}


print(brent(cos, 0.0, 3.0, my_settings))
