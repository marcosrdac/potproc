#!/bin/python3.7
#
# Code by: Marcos Conceição
# E-mail:  marcosrdac@gmail.com
# GitHub:  https://github.com/marcosrdac

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns


def plot(grid,
         fig,
         ax,
         contourf=True,
         four=None,
         phase=None,
         title=None,
         vmin=None,
         vmax=None,
         log=False):
    ax.set_title(title)
    if four:
        dfour = np.fft.fft2(grid)
        dfour = np.fft.fftshift(dfour)
        if phase:
            grid = np.angle(dfour)
        else:
            grid = np.abs(dfour)
        y = np.arange(-grid.shape[0] // 2, -grid.shape[0] // 2 + grid.shape[0])
        x = np.arange(-grid.shape[1] // 2, -grid.shape[1] // 2 + grid.shape[1])
    else:
        y = np.arange(grid.shape[0])
        x = np.arange(grid.shape[1])
    x, y = np.meshgrid(x, y)

    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if contourf:
        ax.invert_yaxis()
        contourf = ax.contourf(x, y, grid, norm=norm)
        cbar = fig.colorbar(contourf, ax=ax)
    else:
        if log:
            sns.heatmap(grid,
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                        norm=norm,
                        cbar=False)
        else:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax)


def get_versor(d, i):
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_d, sin_d = np.cos(d), np.sin(d)
    return np.asarray((cos_i * cos_d, cos_i * sin_d, sin_i))


def pc(grid, md=0, mi=45, fd=0, fi=0, mnd=90, mni=0, fnd=90, fni=0):
    md, mi, fd, fi, mnd, mni, fnd, fni = np.radians(
        (md, mi, fd, fi, mnd, mni, fnd, fni))

    # getting versors
    mh = get_versor(md, mi)
    fh = get_versor(fd, fi)
    mnh = get_versor(mnd, mni)
    fnh = get_versor(fnd, fni)

    gridfour = np.fft.fft2(grid)
    gridfour = np.fft.fftshift(gridfour)

    filt = np.ones(gridfour.shape, dtype=np.complex64)
    for k2 in range(-grid.shape[1] // 2,
                    -grid.shape[1] // 2 + grid.shape[1] + 1):
        for k1 in range(-grid.shape[0] // 2,
                        -grid.shape[0] // 2 + grid.shape[0] + 1):
            knorm = np.linalg.norm((k1, k2))
            k = np.array([k1 * 1j, k2 * 1j, knorm])
            Tmn = np.dot(k, mnh)
            Tfn = np.dot(k, fnh)
            Tm = np.dot(k, mh)
            Tf = np.dot(k, fh)
            if Tm * Tf != 0:
                filt[k1, k2] = (Tmn * Tfn) / (Tm * Tf)
    filt = np.fft.fftshift(filt)

    gridfour *= filt
    gridfour = np.fft.ifftshift(gridfour)
    grid = np.fft.ifft2(gridfour).real
    return grid


def rtp(grid, md=0, mi=45, fd=0, fi=0):
    return (pc(grid, md, mi, fd, fi, mnd=0, mni=90, fnd=0, fni=90))


def rte(grid, md=0, mi=45, fd=0, fi=0):
    return (pc(grid, md, mi, fd, fi, mnd=0, mni=90, fnd=0, fni=0))


def pseud(grid, md=0, mi=45, fd=0, fi=0):
    mnd = 0
    mni = 90
    fnd = 0
    fni = 90

    md, mi, fd, fi, mnd, mni, fnd, fni = np.radians(
        [md, mi, fd, fi, mnd, mni, fnd, fni])
    Cm = 1e-7
    gamma = 6.6743015e-11
    rho = 1
    M = 1
    K = gamma / Cm * rho / M

    # getting versors
    mh = get_versor(md, mi)
    fh = get_versor(fd, fi)
    mnh = get_versor(mnd, mni)
    fnh = get_versor(fnd, fni)

    gridfour = np.fft.fft2(grid)
    gridfour = np.fft.fftshift(gridfour)

    filt = np.ones(gridfour.shape, dtype=np.complex64)
    for k2 in range(-grid.shape[1] // 2,
                    -grid.shape[1] // 2 + grid.shape[1] + 1):
        for k1 in range(-grid.shape[0] // 2,
                        -grid.shape[0] // 2 + grid.shape[0] + 1):
            knorm = np.linalg.norm([k1, k2])
            k = np.array([k1 * 1j, k2 * 1j, knorm])
            Tmn = np.dot(k, mnh)
            Tfn = np.dot(k, fnh)
            Tm = np.dot(k, mh)
            Tf = np.dot(k, fh)
            if Tm * Tf != 0:
                filt[k1, k2] = (Tmn * Tfn) / (Tm * Tf * knorm)
    filt = np.fft.fftshift(filt)

    gridfour *= filt
    gridfour = np.fft.ifftshift(gridfour)
    grid = np.fft.ifft2(gridfour).real
    return (grid)


def uc(grid, dz):
    n, m = grid.shape

    gridfour = np.fft.fft2(grid)
    gridfour = np.fft.fftshift(gridfour)

    filt = np.empty(gridfour.shape)
    for k2 in range(-m // 2, -m // 2 + m + 1):
        for k1 in range(-n // 2, -n // 2 + n + 1):
            knorm = np.linalg.norm([k1, k2])
            filt[k1, k2] = np.exp(-knorm * dz)
    filt = np.fft.fftshift(filt)

    gridfour = np.fft.ifftshift(gridfour * filt)
    grid = np.fft.ifft2(gridfour).real
    return (grid)


def asa(grid):
    asgrid = grid + 1j * hilbert(grid, axis=0)
    return (np.abs(asgrid))


def plot1(grid,
          sources=None,
          prop='$\Delta T$',
          title='',
          figname=None,
          scale=True):
    fig, axes = plt.subplots()

    if sources:
        sources = np.array(sources)
        sourcex = sources[:, 0]
        sourcey = sources[:, 1]
    else:
        sourcex = np.array([])
        sourcey = np.array([])

    plot(grid, fig, axes, vmin=grid.min(), vmax=grid.max())
    axes.scatter(sourcex, sourcey, c='white', s=3)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if figname:
        fig.savefig(figname + '.png')
    # plt.show()
    plt.close(fig)


def plot23(grid,
           grid_filt,
           sources=None,
           prop='$\Delta T$',
           title='',
           figname=None,
           scale=True):
    # ,constrained_layout=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    if sources:
        sources = np.array(sources)
        sourcex = sources[:, 0]
        sourcey = sources[:, 1]
    else:
        sourcex = np.array([])
        sourcey = np.array([])

    plot(grid,
         fig,
         axes[0, 0],
         title=prop + '\n(domínio do espaço)',
         vmin=grid.min(),
         vmax=grid.max())
    axes[0, 0].scatter(sourcex, sourcey, c='white', s=3)
    plot(grid,
         fig,
         axes[0, 1],
         title='Amp. do ' + prop + ' transformado\n(domínio do n. de onda)',
         four=True,
         log=True)
    plot(grid,
         fig,
         axes[0, 2],
         title='Fase do ' + prop + ' transformado\n(domínio do n. de onda)',
         four=True,
         phase=True)
    if scale:
        plot(grid_filt,
             fig,
             axes[1, 0],
             title='$\Delta T$ filtrado\n(domínio do espaço)')
    else:
        plot(grid_filt,
             fig,
             axes[1, 0],
             title='$\Delta T$ filtrado\n(domínio do espaço)',
             vmin=grid.min(),
             vmax=grid.max())
    axes[1, 0].scatter(sourcex, sourcey, c='white', s=3)
    plot(grid_filt,
         fig,
         axes[1, 1],
         title='Amp. do ' + prop + ' filtrado\n(domínio do n. de onda)',
         four=True,
         log=True)
    plot(grid_filt,
         fig,
         axes[1, 2],
         title='Fase do ' + prop + ' filtrado\n(domínio do n. de onda)',
         four=True,
         phase=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if figname:
        fig.savefig(figname + '.png')
    plt.show()
    plt.close(fig)


def plot_rtp(grid, md=0, mi=45, fd=0, fi=0, sources=[]):
    grid_filt = rtp(
        grid,
        md,
        mi,
        fd,
        fi,
    )
    title = 'Redução ao polo\n' \
        r'$m_{dec}=$'+f'{md}'+r'$^o$, ' \
        r'$m_{inc}=$'+f'{mi}'+r'$^o$'
    figname = f'rtp_md{md}_mi{mi}'
    plot23(grid, grid_filt, title=title, figname=figname, sources=sources)


def plot_rte(grid, md=0, mi=45, fd=0, fi=0, sources=[]):
    grid_filt = rte(
        grid,
        md,
        mi,
        fd,
        fi,
    )
    title = 'Redução ao equador\n' \
        r'$m_{dec}=$'+f'{md}'+r'$^o$, ' \
        r'$m_{inc}=$'+f'{mi}'+r'$^o$'
    figname = f'rte_md{md}_mi{mi}'
    plot23(grid, grid_filt, title=title, figname=figname, sources=sources)


def plot_uc(grid, dz=.1, sources=[]):
    grid_filt = uc(grid, dz=dz)
    title = 'Continuação para cima\n' \
            r'$\Delta z$='+f'{dz:.2f}km'
    figname = f'uc_dz{dz}'
    plot23(grid,
           grid_filt,
           title=title,
           figname=figname,
           sources=sources,
           scale=False)


def plot_dc(grid, dz=.1, sources=[]):
    grid_filt = uc(grid, dz=-dz)
    title = 'Continuação para baixo\n' \
            r'$\Delta z$='+f'{dz:.2f}km'
    figname = f'dc_dz{dz}'
    plot23(grid,
           grid_filt,
           title=title,
           figname=figname,
           sources=sources,
           scale=True)


def plot_asa(grid, figid='', sources=[]):
    grid_filt = asa(grid)
    title = 'Sinal analítico'
    figname = 'asa' + figid
    plot23(grid, grid_filt, title=title, figname=figname, sources=sources)


def plot_pseud(grid, md=0, mi=45, fd=0, fi=0, sources=[]):
    grid_filt = pseud(
        grid,
        md,
        mi,
        fd,
        fi,
    )
    title = 'Pseudogravidade\n' \
        r'$m_{dec}=$'+f'{md}'+r'$^o$, ' \
        r'$m_{inc}=$'+f'{mi}'+r'$^o$'
    figname = f'pseud_md{md}_mi{mi}'
    plot23(grid, grid_filt, title=title, figname=figname, sources=sources)


def plotrtp45():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')

    # for (md, mi) in [(0,45)]:
    md = 0
    mi = 45
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_rtp(grid, sources=sources)


def plotrte45():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')

    # for (md, mi) in [(0,45)]:
    md = 0
    mi = 45
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_rte(grid, sources=sources)


def plotrtp60():
    grid = np.loadtxt('data/60anomaly3d')

    # for (md, mi) in [(0,45)]:
    md = 0
    mi = 60
    sources = [[30, 70]]
    plot_rtp(grid, md=md, mi=mi, sources=sources)


def plotrte60():
    grid = np.loadtxt('data/60anomaly3d')

    # for (md, mi) in [(0,45)]:
    md = 0
    mi = 60
    sources = [[30, 70]]
    plot_rte(grid, md=md, mi=mi, sources=sources)


def plotuc45(dz=.1):
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_uc(grid, dz, sources=sources)


def plotdc45(dz=.1):
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_dc(grid, dz, sources=sources)


def plotasa45():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')

    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_asa(grid, figid='45', sources=sources)


def plotasa60():
    grid = np.loadtxt('60anomaly3d')
    sources = [[30, 70]]
    plot_asa(grid, figid='60', sources=sources)


def plotpseud45():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')

    md = 0
    mi = 45
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_pseud(grid, md=md, mi=mi, sources=sources)


def plotpseud60():
    grid = np.loadtxt('data/60anomaly3d')
    md = 0
    mi = 60
    sources = [[30, 70]]
    plot_pseud(grid, md=md, mi=mi, sources=sources)


def plot145():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot1(grid,
          title='Anomalia causada por várias fontes\n' +
          '$m_{dec}=0^o, m_{inc}=45^o$',
          figname='plot45',
          sources=sources)


def plot160():
    grid = np.loadtxt('data/60anomaly3d')
    sources = [[30, 70]]
    plot1(grid,
          title='Anomalia causada por várias fontes\n' +
          '$m_{dec}=0^o, m_{inc}=60^o$',
          figname='plot60',
          sources=sources)


def plotwrongrtp45():
    grid = np.loadtxt('data/45anomaly3d')
    grid += np.loadtxt('data/45anomaly3d_1515')
    grid += np.loadtxt('data/45anomaly3d_0015')
    grid += np.loadtxt('data/45anomaly3d_-20-20')
    grid += np.loadtxt('data/45anomaly3d_20-20')

    md = 45
    mi = 70
    sources = [[50, 50], [65, 50], [65, 65], [30, 30], [30, 70]]
    plot_rtp(grid, mi=mi, md=md, sources=sources)


plot145()
plot160()

plotuc45(.1)
plotuc45(.3)
plotuc45(.5)
plotuc45(.7)

plotdc45(.3)
plotdc45(.1)

plotrtp45()
plotrtp60()
plotrte45()
plotrte60()

plotpseud45()
plotpseud60()

plotasa45()
plotasa60()

plotwrongrtp45()
