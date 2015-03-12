
# coding: utf-8

# # A study of a simple parabolic motion
# 
# This python notebook will explore how python and its various tools (NumPy, SciPy, Matplotlib, etc) can be used to study physics problems, using a very simple and well known example: the parabolic motion. This notebook will show:
# 
# * how to plot a mathematical function
# * how to do simple numerical computations
# * how to use symbolic calculations
# 
# First of all, let's include some modules:


# import Numpy, for dealing with data arrays
import numpy as np
# import matplotlib, for plotting figures
import matplotlib.pyplot as plt
# import the mathematical library
import math


# ## Modeling the trajectory
# 
# The trajectory equation for the parabolic motion is well known, and it is:
# 
# $$
# y = x \tan(\theta) - \frac{g}{2 v_0^2 (\cos{\theta})^2} x^2
# $$
# 
# It is pretty simple to represent it in python:

theta = math.radians(20) # in radians (converted from degrees)
v_0 = 30  # in m/s
g = 9.81  # in m/s**2

x = np.arange(0, 60, 0.1)  # array of x coordinates in meters
# the trajectory equation
y = x * math.tan(theta) - (g / (2 * v_0**2 * math.cos(theta)**2)) * x**2


# To plot it, we create a figure with a certain size, plot `x` and `y`, and also set up some axis labels:

plt.figure(figsize=(10,5))  # the figure
plt.plot(x, y)  # plotting y in function of x
plt.xlabel("x (m)")  # x label
plt.ylabel("y (m)")  # y label
plt.show()  # actually creating the figure


# We can plot the trajectory for various values of $\theta$. To do this in an efficienty way, we define a function that returns the parabolic trajectory, taking as inputs the values of $x$, $g$, $v_0$ and $\theta$


def get_y(x, a, v_0, t):
    """
    Simple function to compute y in a parabolic motion
    
    :param x: a numpy array with the X coordinates in meters
    :param a: the acceleration in m/s**2
    :param v_0: the initial velocity in m/s
    :param theta: the initial theta angle in radians
    """
    
    theta = t
    y = x * math.tan(theta) - (a / (2 * v_0**2 * math.cos(theta)**2)) * x**2
    return y


# Then we create a list of values for $\theta$, and plot the various trajectories

thetas = [t for t in np.arange(0, math.pi / 2., 0.1)]
x = np.arange(0, 100, 0.1)  # in meters

plt.figure(figsize=(10, 10))
for t in thetas:
    y = get_y(x, g, v_0, t=t)
    # we want to plot only y>0
    plt.plot(x[y>0], y[y>0], label="#theta = " + str(t))
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    
plt.legend(loc="best")
plt.show()


# defining the class
class ParabolicF(object):
    """
    Simple class defining a parabolic trajectory
    """
        
    # the special initialization method:
    def __init__(self, a, v_0, t):
        """
        Setting the parameters for the trajectory function
        """
        self.a = a
        self.v_0 = v_0
        self.t = t
    
    # the method called when the object is called as a function
    def __call__(self, x):
        """
        Returns the y values for a parabolic trajectory, given x
        """
        y = x * math.tan(self.t) - (self.a / (2 * v_0**2 * math.cos(self.t)**2)) * x**2
        return y

# creating an object
f = ParabolicF(g, v_0, math.pi / 3.)

from scipy import optimize
results = []  # list that will contain our results
thetas = [t for t in np.arange(0, math.pi / 2., 0.0001)]
# loop on thetas
for t in thetas:
    f1 = ParabolicF(g, v_0, t)  # creating the object
    root = optimize.root(f1, [0, 1000], )  # getting the roots
    results.append(root.x[-1])  # using only the last root, as we know the first one is zero

# printing the maximum value, and the correspondant theta
# np.argmax is a NumPy utility that returns the index of the maximum value
print "Maximum is %.2f meters for theta=%.2f degrees" % (np.max(results), np.degrees(thetas[np.argmax(results)]))


# ## Numerical computation of maximum range
from scipy import misc

# defining our parabilic function, this time with fixed parameters
def f_range(theta):
    v_0 = 30
    g = 9.81
    x = ( v_0**2 * math.sin(2 * theta)) / g
    return x

# a thin-grained list of thetas
thetas = np.arange(0, math.pi / 2., 0.0001)
results = []
# the loop over derivatives
for t in thetas:
    results.append(misc.derivative(f_range, t, dx=1e-6,))
    
# creates a numpy array with all the theta, f' couples
r = np.array(zip(thetas, results))

# find the element closest to 0, which is where range is maximum
min_val = r[np.isclose(r[:, 1], 0, atol=1e-03)]

print "Maximum range is %.2f for %.2f degrees" % (f_range(min_val[:, 0][0]), np.degrees(min_val[:, 0])[0])


# ## Full analytical computation of maximum range
# 
# Python is not just brute force computations... you can also do **symbolic computations**, using `SymPy`. For not too complex functions, an analytical derivative can be computed, and then using `scipy.optimize.root` we can find its roots.

import sympy

# define theta as a special SymPy symbol
theta_sym = sympy.Symbol("theta_sym")

# the function to be derived. NB: we have to use sympy.sin instead of math.sin, as we are performin symbolic computations
x_sym = ( v_0**2 * sympy.sin(2 * theta_sym)) / g

# the derivative operation
xprime = x_sym.diff(theta_sym)
print "The derivative is: ", xprime

# lambdify transforms SymPy expressions in lambda functions, that we can use to do further computations
fprime = sympy.lambdify(theta_sym, xprime, 'numpy')
fzero = sympy.lambdify(theta_sym, x_sym, 'numpy')

# let's find the roots
thetas2 = np.arange(0, math.pi / 2., 0.01)
roots = optimize.root(fprime, 0.5, method="hybr").x
print "The maximum range is %.2f for theta %.2f degrees: " % (fzero(roots[0]), np.degrees(roots)[0])

# and plotting again the derivative
plt.figure(figsize=(10, 5))
plt.plot(thetas2, fprime(thetas2))
plt.xlabel("theta (radians)")
plt.ylabel("x_prime(theta)")
plt.title("First-order derivative of range")
plt.show()

