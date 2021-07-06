from routines import *
from os.path import join
from grids import get_grid_45, get_grid_60
from plots import plot, plot1, plot23


def plot_rtp(grid, k, m=(0, 45), f=(0,0), sources=[], figname=None):
    grid_filt = apply_filter(rtp_filt(k, m=m, f=f), grid)
    title = 'Redução ao polo\n' \
        r'$m_{dec}=$'+f'{m[0]}'+r'$^o$, ' \
        r'$m_{inc}=$'+f'{m[1]}'+r'$^o$'
    plot23(grid, grid_filt, title=title, figname=figname, sources=sources)

def plot_rte(grid, k, m=(0, 45), f=(0,0), sources=[], figname=None):
    grid_filt = apply_filter(rtp_filt(k, m=m, f=f), grid)
    title = 'Redução ao equador\n' \
        r'$m_{dec}=$'+f'{m[0]}'+r'$^o$, ' \
        r'$m_{inc}=$'+f'{m[1]}'+r'$^o$'
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


# def plot_uc(grid, dz=.1, sources=[]):
    # grid_filt = uc(grid, dz=dz)
    # title = 'Continuação para cima\n' \
            # r'$\Delta z$='+f'{dz:.2f}km'
    # figname = f'uc_dz{dz}'
    # plot23(grid,
           # grid_filt,
           # title=title,
           # figname=figname,
           # sources=sources,
           # scale=False)


# def plot_dc(grid, dz=.1, sources=[]):
    # grid_filt = uc(grid, dz=-dz)
    # title = 'Continuação para baixo\n' \
            # r'$\Delta z$='+f'{dz:.2f}km'
    # figname = f'dc_dz{dz}'
    # plot23(grid,
           # grid_filt,
           # title=title,
           # figname=figname,
           # sources=sources,
           # scale=True)


# def plot_asa(grid, figid='', sources=[]):
    # grid_filt = asa(grid)
    # title = 'Sinal analítico'
    # figname = 'asa' + figid
    # plot23(grid, grid_filt, title=title, figname=figname, sources=sources)


# def plot_pseud(grid, md=0, mi=45, fd=0, fi=0, sources=[]):
    # grid_filt = pseud(
        # grid,
        # md,
        # mi,
        # fd,
        # fi,
    # )
    # title = 'Pseudogravidade\n' \
        # r'$m_{dec}=$'+f'{md}'+r'$^o$, ' \
        # r'$m_{inc}=$'+f'{mi}'+r'$^o$'
    # figname = f'pseud_md{md}_mi{mi}'
    # plot23(grid, grid_filt, title=title, figname=figname, sources=sources)

if __name__ == '__main__':
    # opening data
    grid, sources = get_grid_45()
    # grid, sources = get_grid_60()


    # preparing dft (grid_t) and dft domain (k)
    k = dftfreq(grid, 1, 1)

    m = (0, 45)
    f = (0, 0)
    plot_rtp(grid, k, m=m, f=f, sources=sources, figname=join('pix',f'rtp_m_{m[0]}_{m[1]}_f_{f[0]}_{f[1]}'))
    # plot_rte(grid, k, m=m, f=f, sources=sources, figname = join('pix',f'rte_md{md}_mi{mi}'))
