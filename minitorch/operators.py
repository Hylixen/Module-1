"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Iterable
from functools import reduce
#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Basic multilpy of two numbers x, y"""
    return x * y


# - id
def id(x: float) -> float:
    """Identity output x"""
    return x


# - add
def add(x: float, y: float) -> float:
    """Add input x and y"""
    return x + y


# - neg
def neg(x: float) -> float:
    """Reverse sign of input x"""
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    """Check x < y"""
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Check x == y"""
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Return maximum of x and y"""
    if x > y:
        return x
    return y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Check ||x - y|| < 1e-2"""
    abs = math.fabs(x - y)
    return abs < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    r"""Sigmod activation:
    $sigmod(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    """
    if x < 0:
        x = math.exp(x)
        return x / (1 + x)
    x = math.exp(-x)
    return 1 / (1 + x)


# - relu
def relu(x: float) -> float:
    r"""$relu(x) = x$ if x > 0 else $relu(x) = 0$"""
    if x >= 0:
        return x
    return 0


# - log
def log(x: float) -> float:
    """Log operator of x"""
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Exp operator of x"""
    return math.exp(x)


# - log_back
def log_back(x: float, cof: float) -> float:
    """Computes the derivative of log times a second arg"""
    return cof / x


# - inv
def inv(x: float) -> float:
    """Computes the reciprocal"""
    return 1 / x


# - inv_back
def inv_back(x: float, cof: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -cof / (x * x)


# - relu_back
def relu_back(x: float, cof: float) -> float:
    """Computes the derivative of relu function times a second arg"""
    if x > 0:
        return cof
    return 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
def negList(x: Iterable[float]) -> Iterable[float]:
    """Negate a list"""
    return [v for v in map(neg, x)]


# - addLists : add two lists together
def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Add two lists together"""
    if len(x) != len(y):
        raise Exception("equal len of x, y required for addLists")
    return [add(v1, v2) for v1, v2 in zip(x, y)]


# - sum: sum lists
def sum(x: Iterable[float]) -> float:
    """Sum lists"""
    if len(x) < 1:
        return 0
    return reduce(add, x)


# - prod: take the product of lists
def prod(x: Iterable[float]) -> float:
    """Take the product of lists"""
    if len(x) < 1:
        raise Exception("Error: got empty list for prod.")
    return reduce(mul, x)


# TODO: Implement for Task 0.3.
