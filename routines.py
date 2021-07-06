import numpy as np
from numpy.linalg import norm


def get_versor(d, i, deg=False):
    if deg:
        d, i = np.radians(d), np.radians(i)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_d, sin_d = np.cos(d), np.sin(d)
    return np.asarray((cos_i * cos_d, cos_i * sin_d, sin_i))


def dft(grid, shift=True):
    grid = np.fft.fft2(grid)
    if shift:
        grid = np.fft.fftshift(grid, axes=(0, 1))
    return grid


def idft(grid, shift=True):
    if shift:
        grid = np.fft.ifftshift(grid, axes=(0, 1))
    grid = np.fft.ifft2(grid)
    return grid


def dftfreq(grid, dx=1, dy=1, shift=True):
    freq_x_ax = np.fft.fftfreq(grid.shape[0], d=dx)
    freq_y_ax = np.fft.fftfreq(grid.shape[1], d=dy)
    freq_x, freq_y = np.meshgrid(freq_x_ax, freq_y_ax, indexing='ij')
    freq = np.stack((freq_x, freq_y), axis=-1)
    if shift:
        freq = np.fft.fftshift(freq, axes=(0, 1))
    return freq


def apply_filter(phi, grid):
    grid_t = dft(grid)
    if phi.ndim == 2:
        result = idft(phi * grid_t).real
    if phi.ndim == 3:
        result = np.empty(phi.shape)
        for d in range(result.shape[2]):
            result[:, :, d] = idft(phi[:, :, d] * grid_t).real
    return result


def dftgrad(k):
    k_norm = norm(k, axis=-1)
    i_kx, i_ky = 1j * k[:, :, 0], 1j * k[:, :, 1]
    phi = np.stack((i_kx, i_ky, k_norm), axis=-1)
    return phi


def phase_change_filt(k,
                      m_old=(0, 45),
                      f_old=(0, 0),
                      m_new=(0, 90),
                      f_new=(0, 90)):
    '''Phase change filter. Directions as (azimuth, inclination)'''
    directions = (m_old, f_old, m_new, f_new)
    m_old, f_old, m_new, f_new = (get_versor(d, i) for (d, i) in directions)

    grad = dftgrad(k)

    theta_m_new = np.dot(grad, m_new)
    theta_f_new = np.dot(grad, f_new)
    theta_m_old = np.dot(grad, m_old)
    theta_f_old = np.dot(grad, f_old)

    phi = (theta_m_new * theta_f_new) / (theta_m_old * theta_f_old)
    phi[~np.isfinite(phi)] = 1
    return phi


def rtp_filt(k, m=(0, 45), f=(0, 0)):
    return phase_change_filt(k, m_old=m, f_old=f, m_new=(0, 90), f_new=(0, 90))


def rte_filt(k, m=(0, 45), f=(0, 0)):
    return phase_change_filt(k, m_old=m, f_old=f, m_new=(0, 0), f_new=(0, 0))


def upward_cont_filt(k, dz):
    k_norm = norm(k, axis=-1)
    return np.exp(-k_norm * dz)


def downward_cont_filt(grid, dz):
    return upward_cont_filt(grid, -dz)


def dz_filt(k):
    return norm(k, axis=-1)


def analytic_signal_filt(k):
    analytic_signal = dftgrad(k)
    analytic_signal[..., -1] *= 1j
    return analytic_signal


def asa(grid, k):
    phi = analytic_signal_filt(k)
    return norm(apply_filter(phi, grid), axis=-1)


def butterworth_filt(k, cutoff, level):
    return 1 / (1 + (k / cutoff)**level)

def butterworth(grid, k, cutoff, level=2):
    phi_x = butterworth_filt(k[:,:,0], cutoff=cutoff, level=level)
    phi_y = butterworth_filt(k[:,:,1], cutoff=cutoff, level=level)
    butterworth = apply_filter(phi_x, grid)
    butterworth = apply_filter(phi_y, butterworth)
    return butterworth


def tilt_angle(grid, k):
    grad = apply_filter(dftgrad(k), grid)
    Dz = grad[:, :, -1]
    Dh = norm(grad[:, :, :2], axis=-1)
    return np.arctan(Dz/Dh)

def asta(grid, k):
    tilt = tilt_angle(grid, k)
    return asa(tilt, k)


def pseudogravity_filt(k, m=(0, 45), f=(0, 0)):
    gamma = 6.6743015e-11
    Cm = 1e-7
    rho = 1
    M = 1
    K = gamma / Cm * rho / M
    return K * rtp_filt(k, m=m, f=f)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from grids import get_grid_45, get_grid_60
    from plots import plot, plot1, plot23


    # opening data
    grid, sources = get_grid_45()
    # grid, sources= get_grid_60()


    # preparing dft domain (k)
    k = dftfreq(grid, 1, 1)


    # filter to apply
    ## separable filters
    phi = rtp_filt(k, m=(0, 45), f=(0, 0))
    # phi = pseudogravity_filt(k, m=(0, 45), f=(0, 0))
    # phi = rte_filt(k, m=(0,45), f=(0,0))
    # phi = dz_filt(k)
    # phi = upward_cont_filt(k, 50)
    # phi = downward_cont_filt(k, 5)
    # phi = butterworth_filt(k, 1, 8)

    grid_filt = apply_filter(phi, grid)


    ## other filters
    # grid_filt = butterworth(grid, k, cutoff=.01, level=2)
    # grid_filt = asa(grid)
    # grid_filt = tilt_angle(grid, k)
    # grid_filt = asta(grid, k)

    plot23(grid, grid_filt)
