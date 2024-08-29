import numpy as np
import matplotlib.pyplot as plt

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