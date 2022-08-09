import pylab as pl
from collections import OrderedDict
import sys, glob
from math import pi, acos, log
import plotsetting as ps
import numpy as np
import fitfun as ff
import matplotlib.colors as colors
import cmasher as cmr
import fitfun as ff
from matplotlib.backends.backend_pdf import PdfPages

def exactG (V, tp):
    if V != 0:
        return 0.5 * pi / acos(-0.5*V)
    else:
        return 4*tp**2/(1+tp**2)**2

def get_para (fname, key, typ, last=False, n=1):
    with open(fname) as f:
        for line in f:
            if key in line:
                val = list(map(typ,line.split()[-n:]))
                if n == 1: val = val[0]
                if not last:
                    return val
        return val

def get_hop_t (fname):
    L_lead = get_para (fname, 'L_lead', int)
    L_device = get_para (fname, 'L_device', int)
    L = 2*L_lead + L_device
    ts = np.full (L, np.nan)
    with open(fname) as f:
        for line in f:
            if 'H left lead' in line:
                part = 'L'
            elif 'H right lead' in line:
                part = 'R'
            elif 'H dev' in line:
                part = 'S'
            elif 'Hk, t' in line:
                if part == 'L': offset = 0
                elif part == 'S': offset = L_lead
                elif part == 'R': offset = L_lead + L_device
                tmp = line.split()
                i = int(tmp[-3])
                t = float(tmp[-1])
                ts[i+1+offset] = t
            elif 't_contactL' in line:
                t = float(line.split()[-1])
                ts[L_lead] = t
            elif 't_contactR' in line:
                t = float(line.split()[-1])
                ts[L_lead+L_device] = t
    return ts

def get_data (fname):
    #dmrgdir = get_para (fname, 'input_dir', str)
    #dmrgfile = glob.glob ('../gs/*.out')[0]
    L_lead = get_para (fname, 'L_lead', int)
    L_device = get_para (fname, 'L_device', int)
    L = 2*L_lead + L_device
    Nstep = get_para (fname, 'step =', int, last=True)
    iC = get_para (fname, 'charge site', int)
    hopt = get_hop_t (fname)
    maxnC = get_para (fname, 'maxCharge', int)
    t_lead = get_para (fname, 't_lead', float)

    jLs = np.full (Nstep, np.nan)
    jRs = np.full (Nstep, np.nan)
    ns = np.full ((Nstep,L+1), np.nan)
    Ss = np.full ((Nstep,L+1), np.nan)
    dims = np.full ((Nstep,L), np.nan)
    nCs = np.full ((Nstep,2*maxnC+1), np.nan)
    EEs = np.full (Nstep+1, np.nan)
    nns = dict()
    with open(fname) as f:
        for line in f:
            line = line.lstrip()
            if line.startswith ('step ='):
                tmp = line.split()
                step = int(tmp[-1])
            elif line.startswith('I L/R ='):
                tmp = line.split()
                jL = float(tmp[-2]) * 2*pi * t_lead
                jR = float(tmp[-1]) * 2*pi * t_lead
                jLs[step-1] = jL
                jRs[step-1] = jR
            elif line.startswith('*den '):
                tmp = line.split()
                i = int(tmp[1])
                n = float(tmp[-1])
                ns[step-1,i-1] = n
            elif line.startswith('*entS'):
                tmp = line.split()
                i = int(tmp[1])
                S = float(tmp[-1])
                Ss[step-1,i-1] = S
            elif line.startswith('*m'):
                tmp = line.split()
                i = int(tmp[1])
                m = float(tmp[-1])
                dims[step-1,i-1] = m
            elif line.startswith('*nC'):
                tmp = line.split()
                n = int(tmp[-2])
                nC = float(tmp[-1])
                nCs[step-1,n+maxnC] = nC
            elif line.startswith('EE ='):
                EEs[step] = float(line.split()[-1])
            elif line.startswith('Initial EE'):
                EEs[0] = float(line.split()[-1])
            elif line.startswith('nn '):
                tmp = line.split()
                i,j,nn = int(tmp[1]), int(tmp[2]), float(tmp[-1])
                if (i,j) not in nns:
                    nns[i,j] = np.full (Nstep, np.nan)
                else:
                    nns[i,j][step-1] = nn
    # jLs: current for the link that is left to the scatterer
    # jRs: current for the link that is right to the scatterer
    # ns: Occupation
    # Ss: entanglement entropy
    # dims: bond dimension
    # nCs: distribution on the charge site
    return Nstep, L, jLs, jRs, ns, Ss, dims, nCs, EEs, nns

def plot_prof (ax, data, dt, label=''):
    Nstep, L = np.shape(data)
    sc = ax.imshow (data, origin='lower', extent=[1, L, dt, Nstep*dt], aspect='auto')
    cb = pl.colorbar (sc)
    ax.set_xlabel ('site')
    ax.set_ylabel ('time')
    cb.ax.set_ylabel (label)

def plot_time_slice (ax, data, n, xs=[], label='', **args):
    Nstep, L = np.shape(data)
    itv = Nstep // n
    if len(xs) == 0:
        xs = range(1,L+1)
    for d in data[::itv,:]:
        ax.plot (xs, d, **args)
    ax.plot (xs, data[-1,:], label=label, **args)

def get_basis (fname):
    ens, segs = [],[]
    with open(fname) as f:
        for line in f:
            if 'Orbitals: name, ki, energy' in line:
                for line in f:
                    tmp = line.split()
                    if not tmp[0][:-1].isdigit():
                        return np.array(ens), np.array(segs)
                    ens.append (float(tmp[3]))
                    segs.append (tmp[1])

def extrap_current (ts, Il, Ir, plot=False):
    n = 100
    ts = np.reciprocal(ts)[-n:]
    Il = Il[-n:]
    Ir = Ir[-n:]
    if plot:
        f,ax = pl.subplots()
        ax.plot (ts, Il, marker='.', ls='None', label='left')
        ax.plot (ts, Ir, marker='.', ls='None', label='right')
        fitx, fity, stddev, fit = ff.myfit (ts, Il, order=1, ax=ax, refit=True)
        fitx, fity, stddev, fit = ff.myfit (ts, Ir, order=1, ax=ax, refit=True)


if __name__ == '__main__':
    files = [i for i in sys.argv[1:] if i[0] != '-']
    for fname in files:
        print (fname)
        pdfall = PdfPages (fname.replace('.out','')+'.pdf')

        en_basis, segs = get_basis (fname)

        # Get data
        Nstep, L, jLs, jRs, ns, Ss, dims, nCs, EEs, nns = get_data (fname)
        dt = get_para (fname, 'dt', float)
        m = get_para (fname, 'Largest link dim', int)
        ts = dt * np.arange(1,Nstep+1)

        # n profile
        f,ax = pl.subplots()
        plot_prof (ax, ns, dt, 'Density')
        ax.set_title ('$m='+str(m)+'$')
        ps.set(ax)
        pdfall.savefig(f)

        f,ax = pl.subplots()
        ii = segs == 'L'
        plot_time_slice (ax, ns[:,ii], n=5, marker='.', ls='None', label='L', xs=en_basis[ii])
        ii = segs == 'R'
        plot_time_slice (ax, ns[:,ii], n=5, marker='x', ls='None', label='R', xs=en_basis[ii])
        ii = segs == 'S'
        plot_time_slice (ax, ns[:,ii], n=5, marker='+', ls='None', label='S', xs=en_basis[ii])
        iis = np.where (ii)[0]
        for x in iis:
            ax.axvline (en_basis[x], ls='--', c='gray', alpha=0.5)
        ii = segs == 'C'
        plot_time_slice (ax, ns[:,ii], n=5, marker='*', ls='None', label='C', xs=en_basis[ii])
        ax.set_xlabel ('Energy')
        ax.set_ylabel ('Occupation')
        ax.legend()
        ps.set(ax)
        pdfall.savefig(f)

        # Scatterer occupation
        slabel = dict()
        slabel[iis[0]+1] = '$|0\\rangle$'
        slabel[iis[1]+1] = '$|1\\rangle$'
        f,ax = pl.subplots()
        for i in iis:
            ax.plot (ts, ns[:,i], label=slabel[i+1])
        ax.set_xlabel('Time')
        ax.set_ylabel('Occupation')
        ax.legend()
        ps.set(ax)
        pdfall.savefig(f)

        # S profile
        f,ax = pl.subplots()
        plot_prof (ax, Ss, dt, 'Entropy')
        ax.set_title ('$m='+str(m)+'$')
        ps.set(ax)
        pdfall.savefig(f)

        f,ax = pl.subplots()
        plot_time_slice (ax, Ss, n=3)
        ax.set_xlabel ('Site')
        ax.set_ylabel ('Entropy')
        ps.set(ax)
        pdfall.savefig(f)

        # Bond dimension vs. MPS bond
        f,ax = pl.subplots()
        plot_time_slice (ax, dims, n=5)
        ax.set_xlabel ('MPS bond')
        ax.set_ylabel ('Bond dimension')
        ps.set(ax)
        pdfall.savefig(f)

        # Bond dimension vs. time
        max_dims = np.amax (dims, axis=1)
        f,ax = pl.subplots()
        ax.plot (ts, max_dims, marker='.')
        ax.set_xlabel ('Time')
        ax.set_ylabel ('Bond dimension')
        ps.set(ax)
        pdfall.savefig(f)

        # Charge site Occupation
        f,ax = pl.subplots()
        maxnC = get_para (fname, 'maxCharge', int)
        cs = range(-maxnC,maxnC+1)
        for i in range(len(cs)):
            ax.plot (ts, nCs[:,i], label='n='+str(cs[i]));
        ax.set_xlabel ('Time')
        ax.set_ylabel ('occupation')
        ax.legend()
        ps.set(ax)

        # Current vs time
        f,ax = pl.subplots()
        muL = get_para (fname, 'mu_biasL', float)
        muR = get_para (fname, 'mu_biasR', float)
        Vb = muR - muL
        Il = jLs / Vb
        Ir = jRs / Vb
        ax.plot (ts, Il, label='left')
        ax.plot (ts, Ir, label='right')
        ax.set_xlabel ('Time')
        ax.set_ylabel ('Current')
        ax.legend()
        ps.set(ax)
        pdfall.savefig(f)

        # Entanglement entropy
        f,ax = pl.subplots()
        ax.plot (np.insert(ts, 0, 0.), EEs, marker='.')
        ax.set_xlabel ('Time')
        ax.set_ylabel ('Entanglement entropy')

        yticks = [log(3**0.5), log(2**0.5), log(2), log(4)]
        for yt in yticks:
            ax.axhline (yt, ls='--', c='k')
        ax2 = ax.twinx()
        ax2.set_ylim (ax.get_ylim())
        ax2.set_yticks (yticks, ['$\log(\sqrt{3})$','$\log(\sqrt{2})$','$\log(2)$','$\log(4)$'])
        ps.set(ax)
        pdfall.savefig(f)

        # Occupation correlation
        f,ax = pl.subplots()
        for i,j in nns:
            corr = nns[i,j] - ns[:,i-1]*ns[:,j-1]
            ax.plot (ts, corr, label=slabel[i]+','+slabel[j])
        ax.set_xlabel ('Time')
        ax.set_ylabel ('Occupation correlation')
        ax.legend()
        ps.set(ax)
        pdfall.savefig(f)

        pdfall.close()
    pl.show()
