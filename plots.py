import numpy as np
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


def plot1(grid,
          sources=None,
          prop='$\Delta T$',
          title='',
          figname=None,
          scale=True):
    fig, axes = plt.subplots()

    if sources is not None:
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

    if sources is not None:
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
