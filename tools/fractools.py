import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def koch_snowflake(x, y, n_iter, theta = np.pi/3):

    """
    Given a set of x and y points, loops over pairs
    of points and applies the following transformation:
    the line is split into 3 segments of equal length,
    and the middle segment is bent into two sides of
    an equilateral triangle.
    """

    # Loop over a number of iterations
    for n in range(n_iter):

        # Loop over pairs of points
        xn, yn = [], []
        for i in range(len(x) - 1):

            # Generate two new points along the line
            a = [(2*x[i] + x[i+1])/3, (2*y[i] + y[i+1])/3]
            b = [(x[i] + 2*x[i+1])/3, (y[i] + 2*y[i+1])/3]

            # Calculate the point to be rotated, between a and b, depending on theta
            scl = 1 - 1/(2*np.cos(theta))
            c = [(x[i]*(1 + scl) + x[i+1]*(2 - scl))/3,
                 (y[i]*(1 + scl) + y[i+1]*(2 - scl))/3]

            # Calculate the point of the equilateral triangle
            n = [(c[0] - a[0])*np.cos(theta) - (c[1] - a[1])*np.sin(theta) + a[0],
                 (c[0] - a[0])*np.sin(theta) + (c[1] - a[1])*np.cos(theta) + a[1]]
            
            # Append new points except the final one to avoid duplicates
            xn.append([x[i], a[0], n[0], b[0]])
            yn.append([y[i], a[1], n[1], b[1]])

        # Append the final point and flatten the arrays
        xn.append([x[-1]])
        yn.append([y[-1]])
        x, y = np.hstack(xn), np.hstack(yn)

    # Make copies of the above set and rotate by 120 degrees then translate into position
    rot1 = 2*np.pi/3
    x1 = (x - x[0])*np.cos(rot1) - (y - y[0])*np.sin(rot1) + x[0]
    y1 = (x - x[0])*np.sin(rot1) + (y - y[0])*np.cos(rot1) + y[0]
    x1 -= (x1[-1] - x1[0])
    y1 -= (y1[-1] - y1[0])

    # Repeat for the final edge
    rot2 = 4*np.pi/3
    x2 = (x - x[-1])*np.cos(rot2) - (y - y[-1])*np.sin(rot2) + x[-1]
    y2 = (x - x[-1])*np.sin(rot2) + (y - y[-1])*np.cos(rot2) + y[-1]
    x2 += (x2[-1] - x2[0])
    y2 += (y2[-1] - y2[0])

    # Return the arrays combined together
    return np.hstack([x, x2, x1]), np.hstack([y, y2, y1])

def sierpinski_triangle(ax, triangle, depth, **kwargs):

    """
    A recursive function that plots the Sierpinski triangle fractal.
    This proceeds by taking an array of shape (3, 2) defining three
    points of an equilateral triangle, and then finding the midpoint
    of each side. Since the function is recursive, one can apply the
    same procedure to each sub-triangle, up until the specified depth.
    """

    # Draw the triangle as a patch if the final depth is reached
    if depth == 0:
        ax.add_patch(plt.Polygon(triangle, **kwargs))

    # Otherwise, iterate again
    else:
        t0 = [triangle[0], (triangle[0] + triangle[1])/2, (triangle[0] + triangle[2])/2]
        t1 = [triangle[1], (triangle[1] + triangle[0])/2, (triangle[1] + triangle[2])/2]
        t2 = [triangle[2], (triangle[2] + triangle[0])/2, (triangle[2] + triangle[1])/2]
        sierpinski_triangle(ax, t0, depth - 1, **kwargs)
        sierpinski_triangle(ax, t1, depth - 1, **kwargs)
        sierpinski_triangle(ax, t2, depth - 1, **kwargs)

def n_transform(x, y, z, n):

    """
    N-th power transformation used to approximate a
    3D complex ('triplex') number for iterations in
    generating a mandelbulb.
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    # Apply transformation to get new coordinates
    xn = (r**n) * np.sin(n*theta) * np.cos(n*phi)
    yn = (r**n) * np.sin(n*theta) * np.sin(n*phi)
    zn = (r**n) * np.cos(n*theta)

    # Return new (x, y, z) separately
    return xn, yn, zn

def mandelbulb(x, y, z, n, max_iter = 10, threshold = 2):

    """
    This is a 3D extension of the Mandelbrot set, which
    approximates a transformation of z -> z^n + c iteratively,
    until a max_iter loops are reached or the radial size of the
    output is larger than the threshold (divergent).
    """

    # Initialise output for (x, y, z) points
    output = []
    for xi in tqdm(x):
        for yi in y:
            edge = False # outermost point in (x, y) plane
            for zi in z:

                # Initialise z at the origin and iteration counter
                x0, y0, z0 = 0, 0, 0
                n_iter = 0

                # Keep iterating until max_iter or we reach the edge
                while True:

                    # Apply the nth order transformation
                    xn, yn, zn = n_transform(x0, y0, z0, n)

                    # Update the (x, y, z) coordinates and iteration counter
                    x0 = xn + xi
                    y0 = yn + yi
                    z0 = zn + zi
                    n_iter += 1

                    # Break if the boundary threshold is exceeded
                    # and discard the point
                    if np.sqrt(xn**2 + yn**2 + zn**2) > threshold:

                        # If this is equivalent to the edge,
                        # the next point won't be
                        if edge:
                            edge = False
                        break

                    # Break if we reach the maximum iterations
                    if n_iter > max_iter:

                        # If edge was not yet reached, this must be it;
                        # reset the flag and store the original point
                        if not edge:
                            edge = True
                            output.append([xi, yi, zi])
                        break
                    
    # Split the output into three arrays for (x, y, z) points
    output = np.array(output)
    return output[:, 0], output[:, 1], output[:, 2]

# Class of complex numbers
class complex:

    def __init__(self, a = 0, b = 0):

        # Set coefficients for (a + b*i) based on constructor
        self.a = a # real coefficient
        self.b = b # imaginary coefficient

    def __repr__(self):

        """
        Defines representation for this object as (a + bi).
        """

        return f'({self.a:.2f} + {self.b:.2f}i)'

    def __add__(self, c):

        """
        Adds two complex numbers together as z1 + z2.
        """

        return complex(self.a + c.a, self.b + c.b)

    def __mul__(self, c):

        """
        Multiplies two complex numbers together as z1 * z2.
        """

        return complex(self.a*c.a - self.b*c.b, self.a*c.b + self.b*c.a)
    
    def __abs__(self):

        """
        Returns the magnitude of the complex object.
        """

        return np.sqrt(self.a**2 + self.b**2)
    
    @staticmethod
    def iterate(z, c, max_iter = 100, threshold = 5):

        """
        For a complex number z, performs the z -> z**2 + c
        transformation for n_iter iterations, and determines whether
        the number converges or diverges based whether a given
        value exceeds the provided threshold.
        """

        # Loop over iterations
        for i in range(max_iter):
            
            # Exit loop if deemed to diverge
            if abs(z) > threshold:
                return i
            
            # Otherwise iterate again
            z = z*z + c

        # Return a small number if convergent
        return 1