
import sys, os
from pathlib import Path

# before going to the project root
SVDIR = os.getcwd() + "/figures"
# print(SVDIR)
os.makedirs(SVDIR, exist_ok=True)

# act as if we at the project root
sys.path[:0] = [str(Path(sys.path[0]).parents[1])]

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
## DESIGN PLOT GRAPHICS

prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
colors = matplotlib.colormaps['tab10'].colors
colors = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish purple
    "#F0E442",  # Yellow
    "#56B4E9"   # Sky blue
]
colors = ['blue', 'red', 'green', '#d55e00', 'grey', 'purple', 'cyan', '#56be49', 'e69f00', 'black']

## DESIGN PLOT GRAPHICS
lw = 0.02
axlw = lw
lnlw = lw
fsz = 0.1
fsztick = 0.2 #"xx-small"
fszaxlbl = "xx-small"
fszlgnd = "xx-small"
pdsz = 0.25
majticksz = 0.2
minticksz = majticksz/2
figw = (10/3)*0.1
figh = figw*0.36 # pz *0.67

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'''
\usepackage{color} 
\usepackage{amsmath} 
\usepackage{amssymb} 
\usepackage{times} 
\usepackage{array} 
\usepackage{newtxtext}     % Modern Times Roman text font
\usepackage{microtype} % Fixes micro-spacing, tracking, and character protrusion

% --- Custom Spacing Variables ---
\scriptspace=0pt   % removes the accidental horizontal gap after subscripts
\renewcommand{\arraystretch}{0.25} % for clean vertical space between cells
''')


def leading_and_exponent(x: float): 
    if x == 0: return 0, 0 
    a = abs(x) 
    # # use scientific-format string to avoid log10 rounding issues 
    # s = "{:.15e}".format(a) # e.g. "2.000000000000000e+06" 
    # mantissa_str, exp_str = s.split('e') 
    # leading = int(mantissa_str.lstrip('0.').lstrip('.')) 
    # # first nonzero digit 
    # exponent = int(exp_str) 
    # # ensure leading is a single digit 1-9 (handles cases like 0.0... ) 
    # if leading == 0: # fallback: compute via arithmetic 
    exponent = math.floor(math.log10(a)) 
    leading = int(a / (10 ** exponent)) 
    if x >= 0: 
        return leading, exponent 
    else:
        return -leading, exponent

def cbinsearch(val,l=0, h=500):

    val = abs(val)
    if val == 1:
        # id = 0
        return 0
    
    id = None
    while l <= h:
        m = (l+h)//2
        if val >= 1:
            eval = 10**m
            if eval >= val:
                id = m
                h = m-1
            else:
                l = m+1
        elif val > 0:
            eval = 10**(-m)
            if val >= 10**(-m) :
                id = -m
                h = m-1
            else:
                l = m+1
        elif np.isnan(val) or np.isinf(val):
            id = None
    # print(val, id)

    return id

def logpow_tick_values(tick_val, pos=0):
    """
    logplot tick vals
    """
    # print('val', tick_val)
    if not np.isinf(tick_val) or np.isnan(tick_val):
        # print(tick_val)
        if tick_val == 0:
            return str(r'$\mathrm{\mathsf{\hbox{0}}}$')
        else:
            id = cbinsearch(tick_val)
            if id is not None:
                # fntsel = r'\fontsize{0.5}{0.05}\selectfont'
                fntsel = r'\fontsize{0.3}{1}\selectfont'
                new_tick_format = r'\hbox{10}^{\hbox{'+fntsel+f'{id}'+'}}'
                if tick_val < 0:
                    new_tick_format = r'\hbox{10}^{\hbox{'+fntsel+f'-{id}'+'}}'
            else:
                new_tick_format = tick_val
            
            # make new_tick_format into a string value
            new_tick_format =  str(r'$\mathrm{\mathsf{'+ new_tick_format + r'}}$')
    else:
        new_tick_format = tick_val

    return new_tick_format

def reformat_large_tick_values(tick_val, pos=0):
    """
    https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    elif int(tick_val) == 0:
        new_tick_format = int(tick_val)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)
    
    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]
            
    return new_tick_format
  
def reformat_small_tick_values(tick_val, pos=1):
    """
    Formats small tick values
    """
    negsign = False
    sgn = str(tick_val)[0]
    if sgn == '-':
        negsign = True
    
    # fix for values greater than 0.01
    if tick_val < 10:
        
        estr = 'e-'
        ise = estr in str(tick_val)
        if not ise:
            estr = 'E-'
            ise = estr in str(tick_val)
            
        if ise:
            val, sigits = str(tick_val).split(estr)
            val = round(float(val),1)
            rdigits = str(val).split('.')
            if rdigits[-1] == '0':
                val = rdigits[0]
            if float(sigits) == 0:
                new_tick_format = f"{val}"
            else:
                new_tick_format = f"{val}E-{int(float(sigits))}"            
        else:
            tick_val = float(tick_val)/(10**2)
            
            estr = 'e-'
            ise = estr in str(tick_val)
            if not ise:
                estr = 'E-'
                ise = estr in str(tick_val)
                
            if ise:
                val, sigits = str(tick_val).split(estr)
                val = round(float(val),1)
                rdigits = str(val).split('.')
                if rdigits[-1] == '0':
                    val = rdigits[0]
                if int(float(sigits)-2) == 0:
                    new_tick_format = f"{val}"
                else:
                    new_tick_format = f"{val}E-{int(float(sigits)-2)}"
                    
            else:  
                sigits = len(str(tick_val))-2
                val = tick_val
                if sigits > 0:
                    sv = str(tick_val)
                    cnt = -1
                    for chr in sv.split('0'):
                        cnt+=1
                        if chr not in ['','.','-']:
                            val = chr
                            break
                    sigits = cnt
                    try:
                        digs = len(val)-1
                        val = round(float(val)/(10**digs),1)
                        rdigits = str(val).split('.')
                        if rdigits[-1] == '0':
                            val = rdigits[0]
                        if sigits-2 <= 0:
                            new_tick_format = f"{val}"
                        else:
                            new_tick_format = f"{val}E-{sigits-2}"
                    except:
                        new_tick_format = tick_val
                else:
                    new_tick_format = tick_val
    else:
        new_tick_format = reformat_large_tick_values(tick_val, pos)
            
    if negsign: 
        new_tick_format = '-' + new_tick_format
        
    return new_tick_format



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "legend.fontsize":fszlgnd,
    "legend.handlelength": 0.5,
    "legend.handletextpad": 0.25,
    "legend.columnspacing":0.62,
    "font.size": fsz,    
    "axes.titlesize":fszaxlbl,
    "axes.titlepad":0.5,
    "axes.labelsize":fszaxlbl,
    "axes.labelpad":0.5,
    "xtick.labelsize": fsztick,
    "ytick.labelsize":fsztick,
    "axes.linewidth": axlw,
    "lines.linewidth": lnlw,
    "lines.markersize":lnlw,
    "xtick.major.size": majticksz,
    "xtick.major.width": lw,
    "xtick.major.pad": pdsz,
    "xtick.minor.size": minticksz,
    "xtick.minor.width": lw,
    "xtick.minor.pad": pdsz,
    "ytick.major.size": majticksz,
    "ytick.major.width": lw,
    "ytick.major.pad": pdsz,
    "ytick.minor.size": minticksz,
    "ytick.minor.width": lw,
    "ytick.minor.pad": pdsz,    
    "figure.figsize": (figw, figh),
    "figure.labelsize": "small",
    "figure.dpi": 4000,
    "figure.constrained_layout.use":True,
    "savefig.dpi": "figure",
    "savefig.bbox": "standard", # standard or tight
})

# fmt axes
def addticks(ticks, min, max):
    tmp = [min, ]
    for tk in ticks:
        tmp.append(tk)
    tmp.append(max)
    return tmp


def addnewticks(ticks, newtks):
    for tk in ticks:
        if tk not in newtks:
            newtks.append(tk)
    return newtks

def remnegticks(ticks):
    tmp = []
    for tk in ticks:
        if tk >= 0:
            tmp.append(tk)
    return tmp

deflgnd =dict(loc='best', ncols=1, borderaxespad=0., fancybox=False, edgecolor='black', frameon=False, alignment='center', prop={'size': 0.67}, handlelength=0.3, handletextpad=0.15, columnspacing=0.5, labelspacing=-0.33)

lgndkw1 = dict(loc='lower right', ncols=1, borderaxespad=0.1, fancybox=False, edgecolor='black', frameon=False, alignment='left', prop={'size': 0.67}, handlelength=0.3, handletextpad=0.15, columnspacing=0.25, labelspacing=-0.45,)

lgndkw2 = dict(loc='lower center', ncols=1, borderaxespad=0.1, fancybox=False, edgecolor='black', frameon=False, alignment='left', prop={'size': 0.67}, handlelength=0.3, handletextpad=0.15, columnspacing=0.25, labelspacing=-0.45,)

lgndkw3 = dict(loc='lower left', ncols=1, borderaxespad=0.1, fancybox=False, edgecolor='black', frameon=False, alignment='left', prop={'size': 0.67}, handlelength=0.3, handletextpad=0.15, columnspacing=0.25, labelspacing=-0.45,)

def fmtaxes2(ax, h, xlbl=None, ylbl=None, ylims=None, xlims=None, remylims=False, mgxy=[0.1,0.1], lblpad=[0, 0.1], padxy=[-0.33,0.05], lgndkw=deflgnd):

    if xlbl is not None:
        xlbl = fntscalerl+xlbl
    if ylbl is not None:
        ylbl = fntscalerl+ylbl
        
    yticks = ax.get_yticks()
    if ylims is not None and remylims is False:
        ax.set_yticks(addticks(yticks, ylims[0], ylims[1]))

    xticks = ax.get_xticks()
    ax.set_xticks(addticks(xticks, xlims[0], xlims[1]))

    ylabels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox')
                for label in ax.get_ymajorticklabels()]
    xlabels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox')
                for label in ax.get_xmajorticklabels()]
    ax.yaxis.set_ticks(ax.get_yticks(), ylabels)
    ax.xaxis.set_ticks(ax.get_xticks(), xlabels)

    lw = 0.2
    spinelw = 0.2*lw
    fx, fy = 0.5, 0.5
    for spine in ax.spines.values():
        spine.set_linewidth(0.2*lw)

    ax.xaxis.set_tick_params(which='major', labelsize=fx-0.5,length=0.1, width=spinelw,pad=padxy[0])
    ax.xaxis.set_tick_params(which='minor', labelsize=fx-0.5,length=0.05, width=spinelw,pad=padxy[0])   
    if xlbl is not None:
        ax.set_xlabel(xlbl, fontsize=0.5*fx, labelpad=lblpad[0])

    ax.yaxis.set_tick_params(which='major', labelsize=fy-0.5,length=0.15, width=spinelw,pad=padxy[1])    
    ax.yaxis.set_tick_params(which='minor', labelsize=fy-0.5,length=0.05, width=spinelw,pad=padxy[1])   
    if ylbl is not None:
        ax.set_ylabel(ylbl, fontsize=0.5*fy, labelpad=lblpad[1], loc='center', rotation=90)

    if h:
        lgnd = ax.legend(handles=h, **lgndkw)
        # text alignment
        frame = lgnd.get_frame()
        frame.set_linewidth(spinelw)
        lgndverticaltxtalignment = 0.1
        for handle in lgnd.legend_handles:
            handle.set_ydata([lgndverticaltxtalignment]*len(handle.get_xdata()))
    
    if ylims is not None:
        ax.yaxis.set_ticks(ax.get_yticks())
        ax.set_ylim(bottom=ylims[0]-1e-4, top=ylims[1] )
    # print(min(xlims[0], ax.get_xticks()[0]))
    if xlims is not None:
        ax.set_xlim(left=xlims[0], 
        right=min(xlims[1], ax.get_xticks()[-1]) )
    # pla.set_xmargin(ax, left=0)  
    ax.margins(x=mgxy[0], y=mgxy[1], tight=True)




# ---------------------------------------------------------
# BVPs
from bvp_lrs_np import x8Dome, x8M, x4, x3, x2, glin, grcos
# ---------------------------------------------------------

# ---------------------------------------------------------
# Grid
# ---------------------------------------------------------
# Generate grid
def grid(tau): return np.linspace(0, 1, tau)
N = 5000
r = grid(N)

# ---------------------------------------------------------
# Figure layout
# ---------------------------------------------------------
fntscaler = r'\fontsize{0.3}{0.5}\selectfont '    
fntscalerx = r'\fontsize{0.4}{0.5}\selectfont ' 
fntscalery = r'\fontsize{0.4}{0.5}\selectfont ' 
fntscalerl = r'\fontsize{0.25}{0.3}\selectfont ' 

fig, axs = plt.subplots(2,1, figsize=((10/3)*0.03, 0.07*2), dpi=1200, tight_layout=True, gridspec_kw={'height_ratios': [1, 1]})

ax = axs.ravel()
ax_main = ax[0]
ax_dome = ax[1]

plt.subplots_adjust(hspace=0.3)

connections = [] 
handles1 = []
handles2 = []

# Config.
m1, m2, m3 = 0., 0., 0.3
e1, e2, e3 = 0., 0., 0.
c = 0.85

p=1

def _order_proj(m1, m2, m3, e1, e2, e3, c):
    """Keep the 8-point BVP parameters in a valid monotone ordering."""
    m1 = 0.33*m1
    m2 = np.clip(0.66*m2, min=m1)
    m3 = np.clip(0.8*m3, min=m2)

    e1 = np.clip(0.2*e1, min=0, max=m2 - m1)
    e2 = np.clip(0.1*e2, min=0, max=m3 - m2)
    e3 = np.clip(0.1*e3, min=0, max=1 - m3 - 1e-3)

    c = 0.7 + (1-0.7) * (1 - c)

    return m1, m2, m3, e1, e2, e3, c

def _order_projd(m1, m2, m3, e1, e2, e3, c):
    """Keep the 8-point BVP parameters in a valid monotone ordering."""
    m1 = 0.33*m1
    m2 = np.clip(0.66*m2, min=0.33)
    m3 = np.clip(0.8*m3, min=0.7)

    e1 = np.clip(0.2*e1, min=0, max=m2 - m1)
    e2 = np.clip(0.1*e2, min=0, max=m3 - m2)
    e3 = np.clip(0.1*e3, min=0, max=1 - m3 - 1e-3)

    c = 0.7 + (1-0.7) * c

    return m1, m2, m3, e1, e2, e3, c


def addplt(r, m1, m2, m3, e1, e2, e3, c):

    args = (m1, m2, m3, e1, e2, e3, c)
    m1, m2, m3, e1, e2, e3, c = _order_proj(*args)
    x = x8M(r, m1, m2, m3, e1, e2, e3, c)
    yl = glin(x, p=p)
    yr = grcos(x, p=p)

    # =====================================================
    # TOP: FINAL SCHEDULE
    # =====================================================
    lnlbls = ["normal shape", "regularized shape"]
    lnlbls[:] = [fntscalerl + " " + lbl for lbl in lnlbls]


    h, = ax_main.plot(r, yl, color='red', lw=1*lw, label=lnlbls[0])
    handles1.append(h)
    h, = ax_main.plot(r, yr, color='blue', lw=1*lw, label=lnlbls[1])
    handles1.append(h)

    ax_main.axhline(1, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')
    # ax_main.axhline(0, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')

    axlbls = [
    r'$\phi(t)$', 
    fr"$\begin{{array}}{{l}} \qquad \qquad \qquad r(t) \, \\\\[0em] {{m}}_1=\text{{{m1:.2f}}}, \,{{\varepsilon}}_1=\text{{{e1:.2f}}} \,\\\\[0em] {{m}}_2=\text{{{m2:.2f}}}, \,{{\varepsilon}}_2=\text{{{e2:.2f}}} \,\\\\[0em] {{m}}_3=\text{{{m3:.2f}}}, \, {{\varepsilon}}_3=\text{{{e3:.2f}}}, \, c=\text{{{c:.2f}}} \end{{array}}$",
    fr"$\begin{{array}}{{c}} \text{{8-point BVPs }}\end{{array}}$",
    r'Normalized horizon, $r(t)$'
    ]
    axlbls[:] = [fntscalerl + " " + lbl for lbl in axlbls]

    ylbl = axlbls[0]
    xlbl = axlbls[1]
    ax_main.set_title(axlbls[2], fontsize=fszaxlbl, pad=0.1)

    fmtaxes2(ax_main, handles1, xlbl, ylbl, ylims=(0,1.05), xlims=(0,1), remylims=True, lblpad=[-0.5, 0.1], lgndkw=lgndkw2)

    ax_main.axis('on')

    ax_main.axvline(m1, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_main.axvline(m2, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_main.axvline(m3, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')   
    ax_main.axvline(m1+e1, c='k', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_main.axvline(m2+e2, c='k', alpha=0.5, zorder=4, lw=lw, ls='-.') 
    ax_main.axvline(m3+e3, c='k', alpha=0.5, zorder=4, lw=lw, ls='-.')   

    # =====================================================
    # AX-xbvp
    # =====================================================

    m1, m2, m3, e1, e2, e3, c = _order_projd(*args)
    xd = x8Dome(r, m1, m2, m3, e1, e2, e3, c)
    yld = glin(xd, p=p)
    yrd = grcos(xd, p=p)    

    axlbls = [
    r'$\phi(t)$', 
    fr"$\begin{{array}}{{l}} \qquad \qquad \qquad r(t) \, \\\\[0em] {{m}}_1=\text{{{m1:.2f}}}, \,{{\varepsilon}}_1=\text{{{e1:.2f}}} \,\\\\[0em] {{m}}_2=\text{{{m2:.2f}}}, \,{{\varepsilon}}_2=\text{{{e2:.2f}}} \,\\\\[0em] {{m}}_3=\text{{{m3:.2f}}}, \, {{\varepsilon}}_3=\text{{{e3:.2f}}}, \, c=\text{{{c:.2f}}} \end{{array}}$",
    fr"$\begin{{array}}{{c}} \text{{8-point BVP, }} {{x}}_8\bigl(r(t);m,\varepsilon\bigr)\end{{array}}$",
    r'Normalized horizon, $r(t)$'
    ]
    axlbls[:] = [fntscalerl + " " + lbl for lbl in axlbls]

    h, = ax_dome.plot(r, yld, color='red', lw=1*lw, label=lnlbls[0])
    handles2.append(h)
    h, = ax_dome.plot(r, yrd, color='blue', lw=1*lw, label=lnlbls[1])
    handles2.append(h)

    ax_dome.axhline(1, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')

    ylbl = axlbls[0]
    xlbl = axlbls[1]
    fmtaxes2(ax_dome, handles2, xlbl, ylbl, ylims=(0,1.05), xlims=(0,1), remylims=True, lblpad=[-0.5, 0.1], lgndkw=lgndkw2)

    ax_dome.axvline(m1, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_dome.axvline(m2, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_dome.axvline(m3, c='silver', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_dome.axvline(m1+e1, c='k', alpha=0.5, zorder=4, lw=lw, ls='-.')
    ax_dome.axvline(m2+e2, c='k', alpha=0.5, zorder=4, lw=lw, ls='-.')  
    ax_dome.axvline(m3+e3, c='k', alpha=0.5, zorder=4, lw=lw, ls='-.')   


addplt(r, m1, m2, m3, e1, e2, e3, c)
# =====================================================
# SAVING
# =====================================================
fig.tight_layout(pad=0.01)
svdr = f"{SVDIR}"
os.makedirs(svdr, exist_ok=True)
fnm = f"bvp8_plt"

# upscale from 3600 to 9600 dpi for publication quality
fig.savefig(f"{svdr}/{fnm}.png", dpi=9600, format='png',  bbox_inches='tight', pad_inches=0.001)
plt.close(fig)
# plt.show()

# ---------------------------------------------------------
# Animation update
# ---------------------------------------------------------
def update(frame):

    # Print a verbose update for every frame
    # print(f"Rendering frame {frame}...")

    # -------------------------------------------------
    # CLEAR OLD CONNECTIONS and AXES
    # -------------------------------------------------
    
    for hnd in handles1: hnd.remove()
    handles1.clear()
    for hnd in handles2: hnd.remove()
    handles2.clear()    


    for ax in [ax_main, ax_dome]: ax.clear()

    # animated
    def evolve(frame, fps=30):
        x = (frame%fps)/fps
        return 1 - np.cos(np.pi*x)**2

    m1 = evolve(frame, fps=960)
    m2 = evolve(frame, fps=720)
    m3 = evolve(frame, fps=180)
    e1 = evolve(frame, fps=720)
    e2 = evolve(frame, fps=360)
    e3 = evolve(frame, fps=180)
    c = evolve(frame, fps=30)

    addplt(r, m1, m2, m3, e1, e2, e3, c)

# ---------------------------------------------------------
# Run animation
# ---------------------------------------------------------
ani = FuncAnimation(fig, update, frames=360, interval=80)

# Save:
ani.save(svdr+"/bvp_8_anim.gif", writer="pillow", dpi=9600, progress_callback=lambda i, total: print(f'Saved frame {i+1}/{total}', end='\r'))

# plt.show()

