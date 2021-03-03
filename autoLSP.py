import numpy as np
import matplotlib.pyplot as plt


'''
Implementation of Liu et al's Automatic Least Squares Projection method
'''


'''
Given a point p and a projection vector n
Project it onto C by minimizing 
E(p) = sum_i w_i || p* - p_i || ^ 2
'''
def directedProjection(p, n, pts, weights):
    # Ensure given inputs are valid
    np.testing.assert_equal(p.shape, n.shape)
    np.testing.assert_equal(len(pts), len(weights))

    dim = len(p)
    c_0 = weights.sum()
    c = np.array([ (pts[:, d] * weights).sum() for d in range(dim)])
    return (np.dot(c, n) / c_0 - np.dot(p, n))

# Compute the projection direction n
def projDir(pts, weights, p):
    np.testing.assert_equal(len(pts), len(weights))

    dim = len(p)
    c_0 = weights.sum()
    c = np.array([ (pts[:, d] * weights).sum() for d in range(dim)]) 
    m = (c / c_0) - p
    return m / np.linalg.norm(m)

# One step of the LSP method
def projectStep(p, pts):
    '''
    Input:
        p = point we want to project
        pts = points we are projecting onto
    Output:
        n = projection direction
        t = the value such that pnew = p + tn
        weights
    '''
    # Compute weights
    weights = np.array([1 / (1 + np.linalg.norm(p - p_i) ** 4) for p_i in pts])

    # Compute the projection direction
    n = projDir(pts, weights, p)

    # Compute the projection through Directed Projection
    t = directedProjection(p, n, pts, weights)
    return t, n, weights


def LSP(p, pts):
    MAX_STEPS = 5
    EPSILON = 0.1
    k = 0
    active = [True] * len(pts)
    t = 0

    while k < MAX_STEPS:
        tnew, n, weights = projectStep(p, pts[active])

        # Check and update t
        if abs(tnew - t) < EPSILON:
            break
        t = tnew

        # Update the active indices by looking at the weights
        w_max = weights.max()
        w_mean = weights.mean()
        w_lim = w_mean
        if k < 11:
            w_lim += (w_max - w_mean) / (12 - k)
        else:
            w_lim += (w_max - w_mean) / 2

        j = 0
        for i, _ in enumerate(pts):
            if active[i]:
                if weights[j] < w_lim:
                    active[i] = False
                j += 1

        if j == 0:
            break
        k += 1

        '''
        # Can uncomment this code to plot projection at each step
        q = p + t * n
        x_active = pts[:, 0][active]
        y_active = pts[:, 1][active]

        xnot = [pts[:, 0][i] for i in range(len(pts)) if not active[i]]
        ynot = [pts[:, 1][i] for i in range(len(pts)) if not active[i]]

        plt.scatter(x_active, y_active, c="red")
        plt.scatter(xnot, ynot, c="black")
        plt.scatter( [p[0]], [p[1]], c="blue")
        plt.scatter( [q[0]], [q[1]], c="cyan")
        plt.show()
        '''

    return p + t * n

if __name__ == "__main__":
    # Point cloud for the bean curve with noise
    noise = 0.5

    t = np.linspace(0, np.pi * 2, 500)
    r = 10
    x = r * np.cos(t) * (np.sin(t) ** 3 + np.cos(t) ** 3)
    y = r * np.sin(t) * (np.sin(t) ** 3 + np.cos(t) ** 3)

    x += np.random.uniform(low = -noise, high = noise, size = len(x))
    y += np.random.uniform(low = -noise, high = noise, size = len(y))

    z = np.stack( (x, y), 1)


    # Project a bunch of lines showing a point and its projection onto the bean curve


    plt.figure(figsize=(20,10))
    qs = np.random.uniform(-10, 15, size = (100, 2))

    rp = []
    for q in qs:
        r = LSP(q, z)
        rp.append(r)

    for i in range(len(rp)):
        r_x = rp[i][0]
        r_y = rp[i][1]
        plt.plot([qs[i][0], r_x], [qs[i][1], r_y])

    plt.scatter(x, y)

    plt.show()

