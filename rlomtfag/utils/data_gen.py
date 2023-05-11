import numpy as np

"""
Generate mock data for testing collevrative ability of RLOMTFAG.
"""

def two_moon_gen(num1, num2=None, sigma_noise=0.1, horizontal=0, vertical=0.1):
    if num2 is None:
        num2 = num1
    if sigma_noise is None:
        sigma_noise = 0.1
    if horizontal is None and vertical is None:
        level = 0.3
        upright = 0.1
    else:
        level = 0.32 + horizontal
        upright = 0.15 + vertical

    t = np.linspace(np.pi, 0, num1)
    input1 = np.zeros((num1, 2))
    input1[:, 0] = np.cos(t) + np.random.randn(num1) * sigma_noise - level
    input1[:, 1] = np.sin(t) + np.random.randn(num1) * sigma_noise - upright

    t = np.linspace(np.pi, 2 * np.pi, num2)
    input2 = np.zeros((num2, 2))
    input2[:, 0] = np.cos(t) + np.random.randn(num2) * sigma_noise + level
    input2[:, 1] = np.sin(t) + np.random.randn(num2) * sigma_noise + upright

    result = np.vstack((input1, input2))

    return result


def three_ring_gen(n, noise1=0.035, fea_n=1, noise2=0.1):
    # generate three ring data, each row is a data

    n1 = int(1/10 * n)
    n2 = int(3/10 * n)
    n3 = int(6/10 * n)

    curve = 2.5

    # 2-D data
    r = 0.2
    t = np.random.uniform(0, 0.8, n1)
    x = r * np.sin(curve * np.pi * t) + noise1 * np.random.randn(n1)
    y = r * np.cos(curve * np.pi * t) + noise1 * np.random.randn(n1)
    z = 5 * noise2 * np.random.randn(n1, fea_n)
    data1 = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z), axis=1)

    r = 0.6
    curve = 2.5
    t = np.random.uniform(0, 0.8, n2)
    x = r * np.sin(curve * np.pi * t) + noise1 * np.random.randn(n2)
    y = r * np.cos(curve * np.pi * t) + noise1 * np.random.randn(n2)
    z = 5 * noise2 * np.random.randn(n2, fea_n)
    data2 = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z), axis=1)

    r = 1
    curve = 2.5
    t = np.random.uniform(0, 0.8, n3)
    x = r * np.sin(curve * np.pi * t) + noise1 * np.random.randn(n3)
    y = r * np.cos(curve * np.pi * t) + noise1 * np.random.randn(n3)
    z = 5 * noise2 * np.random.randn(n3, fea_n)
    data3 = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z), axis=1)

    x = np.concatenate((data1, data2, data3), axis=0)
    y = np.concatenate((np.ones(n1), 2 * np.ones(n2), 3 * np.ones(n3)), axis=0)

    if fea_n == 1:
        x = x[:, 0:2]

    return x, y, n1, n2, n3


if __name__ == '__main__':
    pass
