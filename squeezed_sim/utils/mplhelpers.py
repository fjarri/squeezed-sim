# --------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------

import matplotlib as mpl


default_backend = dict(
    name='Agg',
    params={
        'font.family': 'sans-serif',
        'text.usetex': False
    })

helvetica_backend = dict(
    name='pgf',
    params={
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False, # don't setup fonts from rc parameters
        "pgf.preamble": [
            # preamble copied from fonts.tex
            r"\usepackage[scaled]{helvet}",
            r"\renewcommand{\familydefault}{\sfdefault}"
            r"\usepackage[T1]{fontenc}"
            r"%\usepackage[mathrmOrig,mathitOrig]{sfmath}"
            ]
    })

libertine_backend = dict(
    name='pgf',
    params={
        "text.usetex": True,
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False, # don't setup fonts from rc parameters
        "pgf.preamble": [
            # preamble copied from fonts.tex
            r"\usepackage{fontspec}",
            r"\defaultfontfeatures{Ligatures=TeX, Numbers={OldStyle,Proportional}, " +
                r"Scale=MatchLowercase, Mapping=tex-text}"
            r"\usepackage[oldstyle]{libertine}", # Linux Libertine fonts for text
            r"\usepackage{unicode-math}",  # unicode math setup
            r"\setmathfont{Tex Gyre Pagella Math}", # Tex Gyre as main math font
            r"\setmathfont[range={\mathcal,\mathbfcal},StylisticSet=1]{Latin Modern Math}"
            r"\newfontfamily\abbrevfont[Numbers=Uppercase]{Linux Libertine O}"
            r"\newcommand{\abbrev}[1]{{\abbrevfont\expandafter\MakeUppercase\expandafter{#1}}}"
            ]
    })


luatex_backend = dict(
    name='pgf',
    params={
        "text.usetex": True,
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False, # don't setup fonts from rc parameters
        "pgf.preamble": [
            # preamble copied from fonts.tex
            r"\usepackage[scaled]{helvet}",
            r"\usepackage{unicode-math}",  # unicode math setup
            ]
    })


def setup_backend(backend=None):
    if backend is None:
        backend = default_backend
    backend_name = backend['name']
    backend_params = backend['params']

    mpl.use(backend_name)
    mpl.rcParams.update(backend_params)


setup_backend(default_backend) # have to call it before importing pyplot


# --------------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------------

from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


colors_xml = """
<!--
   Color Palette by Color Scheme Designer
-->
<palette>
    <url>http://colorschemedesigner.com/#0742hsOsOK-K-</url>
    <colorspace>RGB; </colorspace>

    <colorset id="primary" title="Primary Color">
        <color id="primary-1" nr="1" rgb="E63717" r="230" g="55" b="23"/>
        <color id="primary-2" nr="2" rgb="8F4B3F" r="143" g="75" b="63"/>
        <color id="primary-3" nr="3" rgb="6D1303" r="109" g="19" b="3"/>
        <color id="primary-4" nr="4" rgb="F9826D" r="249" g="130" b="109"/>
        <color id="primary-5" nr="5" rgb="F9BBB0" r="249" g="187" b="176"/>
    </colorset>
    <colorset id="secondary-a" title="Secondary Color A">
        <color id="secondary-a-1" nr="1" rgb="E6A317" r="230" g="163" b="23"/>
        <color id="secondary-a-2" nr="2" rgb="8F753F" r="143" g="117" b="63"/>
        <color id="secondary-a-3" nr="3" rgb="6D4B03" r="109" g="75" b="3"/>
        <color id="secondary-a-4" nr="4" rgb="F9CC6D" r="249" g="204" b="109"/>
        <color id="secondary-a-5" nr="5" rgb="F9E1B0" r="249" g="225" b="176"/>
    </colorset>
    <colorset id="secondary-b" title="Secondary Color B">
        <color id="secondary-b-1" nr="1" rgb="1F409A" r="31" g="64" b="154"/>
        <color id="secondary-b-2" nr="2" rgb="303D61" r="48" g="61" b="97"/>
        <color id="secondary-b-3" nr="3" rgb="041649" r="4" g="22" b="73"/>
        <color id="secondary-b-4" nr="4" rgb="7392E6" r="115" g="146" b="230"/>
        <color id="secondary-b-5" nr="5" rgb="AABAE6" r="170" g="186" b="230"/>
    </colorset>
    <colorset id="complement" title="Complementary Color">
        <color id="complement-1" nr="1" rgb="11A844" r="17" g="168" b="68"/>
        <color id="complement-2" nr="2" rgb="2E6942" r="46" g="105" b="66"/>
        <color id="complement-3" nr="3" rgb="02501C" r="2" g="80" b="28"/>
        <color id="complement-4" nr="4" rgb="66E992" r="102" g="233" b="146"/>
        <color id="complement-5" nr="5" rgb="A5E9BC" r="165" g="233" b="188"/>
    </colorset>
</palette>
"""


class AttrDict(defaultdict):

    def __init__(self):
        return defaultdict.__init__(self, lambda: AttrDict())

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def load_colors():
    root = ET.XML(colors_xml)
    result = AttrDict()

    names = {'primary': 'red', 'secondary-a': 'yellow',
        'secondary-b': 'blue', 'complement': 'green'}
    types = ['main', 'dark', 'darkest', 'light', 'lightest']

    for elem in root:
        if elem.tag != 'colorset':
            continue

        name = elem.attrib['id']
        r = result[names[name]]

        for color in elem:
            ctype = types[int(color.attrib['id'][-1]) - 1]
            r[ctype] = tuple(int(color.attrib[c]) / 255. for c in ['r', 'g', 'b'])

    return result

color = load_colors()


cm_negpos = LinearSegmentedColormap.from_list(
    "negpos",
    [
        color.blue.dark,
        color.blue.main, color.blue.light,
        (1., 1., 1.),
        color.red.light, color.red.main,
        color.red.dark]
    )

cm_zeropos = LinearSegmentedColormap.from_list(
    "zeropos",
    [
        (0.0, (1.0, 1.0, 1.0)),
        (2.5/15, color.blue.main),
        (5.0/15, (117/255., 192/255., 235/255.)),
        (10.0/15, color.yellow.light),
        (12.5/15, color.red.main),
        (1.0, color.red.dark)
    ]
    )
cm_zeropos.set_under(color='white')


# --------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------

import numpy
import matplotlib.pyplot as plt


class rcparams:

    def __init__(self, **kwds):
        self.new_params = kwds

    def __enter__(self):
        self.old_params = {}
        self.missing_params = set()
        for key, val in self.new_params.items():
            if key in mpl.rcParams:
                self.old_params[key] = mpl.rcParams[key]
            else:
                self.missing_params.add(key)
            mpl.rcParams[key] = val

    def __exit__(self, *args):
        for key in self.missing_params:
            del mpl.rcParams[key]
        for key, val in self.old_params.items():
            mpl.rcParams[key] = val


dash = {
    '-': [],
    '--': (6, 3),
    '-.': (5, 3, 1, 3),
    ':': (0.5, 2),
    '--.': (6, 3, 6, 3, 1, 3),
    '-..': (6, 3, 1, 3, 1, 3),
}


A4_figure = {
    'font.size': 8,
    'lines.linewidth': 1.0,
    'lines.dash_capstyle': 'round',
    'legend.fontsize': 'medium',

    # axes
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.pad': 3,
    'xtick.minor.pad': 3,
    'ytick.major.pad': 3,
    'ytick.minor.pad': 3,

    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    }


rcparams(**A4_figure).__enter__()


def figure(width=1, aspect=None):
    column_width_inches = 8.6 / 2.54 # PRL single column
    if aspect is None:
        aspect = (numpy.sqrt(5) - 1) / 2

    fig_width = column_width_inches * width
    fig_height = fig_width * aspect # height in inches

    return plt.figure(figsize=[fig_width, fig_height])


def close(fig):
    plt.close(fig)


def aspect_modifier(subplot):
    ys = subplot.get_ylim()
    xs = subplot.get_xlim()
    return (xs[1] - xs[0]) / (ys[1] - ys[0])


def text(subplot, x, y, s):
    subplot.text(
        subplot.get_xlim()[0] + (subplot.get_xlim()[1] - subplot.get_xlim()[0]) * x,
        subplot.get_ylim()[0] + (subplot.get_ylim()[1] - subplot.get_ylim()[0]) * y,
        s)


def crop_bounds(x, y_low, y_high, extent):

    xmin, xmax, ymin, ymax = extent

    for imin in range(x.size):
        if x[imin] > xmin:
            break
    for imax in range(imin, x.size):
        if x[imax] > xmax or y_high[imax] < ymin or y_low[imax] > ymax:
            break

    x = x[imin:imax]
    y_low = y_low[imin:imax]
    y_high = y_high[imin:imax]

    return x, numpy.array([max(ymin, y) for y in y_low]), numpy.array([min(ymax, y) for y in y_high])
