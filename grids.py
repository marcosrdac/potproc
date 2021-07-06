import numpy as np

def get_grid_45():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')
    sources = np.asarray([[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]])
    return grid, sources

def get_grid_60():
    grid = np.loadtxt('data/60anomaly3d')
    sources = np.asarray([[30,70]])
    return grid, sources
