---
authors:
- name: Keith Arnaud
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
  orcid: 0000-0001-8977-8916
  website: https://asd.gsfc.nasa.gov/Keith.Arnaud/home.html
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-11-17'
execution:
  cal-files:
    xmm-ccf: false
    chandra: false
    xspec-models: true
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: heasoft
  language: python
  name: heasoft
title: PyXspec - simultaneous fitting
---

# PyXspec – simultaneous fitting

## Learning Goals

By the end of this tutorial, you will be able to:

- Load multiple X-ray spectra into separate PyXspec data groups.
- Fit a spectral model simultaneously to multiple spectra.
- Control whether model parameters are allowed to vary separately for different spectra (untied), or if they must be the same (tied).
- Visualize simultaneously fitted spectra, models, and residuals using PyXspec and matplotlib.

## Introduction

This PyXspec tutorial follows on from the previous '{doc}`PyXspec basics <pyxspec-basics>`'
demonstration (which we strongly encourage you to go through first), and focuses on how you
can analyze multiple spectra at once.

XSPEC has the very useful facility of allowing models to be fitted simultaneously to
more than one data file. It is even possible to group files together and to fit
different models simultaneously. Reasons for fitting in this manner include:

- The same target has been observed at several epochs but, although the source stays constant, the response matrix has changed. When this happens, the data files cannot be added together; they have to be fitted separately. Fitting the data files simultaneously yields tighter constraints.
- The same target is observed with different instruments. All the instruments on Suzaku, for example, observe in the same direction simultaneously. As far as XSPEC is concerned, this is just like the previous case: two data files with two responses fitted simultaneously with the same model.
- Different targets are observed, but the user wants to fit the same model to each data file with some parameters shared and some allowed to vary separately. For example, if we have a series of spectra from a variable AGN, we might want to fit them simultaneously with a model that has the same, common photon index but separately vary the normalization and absorption.

Other scenarios are possible – the important thing is to recognize the flexibility of XSPEC in this regard.

As an example, we will look at a case of fitting the same model to two different data
files but where not all the parameters are identical. Just as in the {doc}`first PyXspec tutorial <pyxspec-basics>`
we will make use of an old dataset, as it provides a simpler illustration than more modern data.

In particular, we will use two observations of Centaurus X-3 (alternatively known as Cen X-3), a very
well-studied high-mass X-ray binary (HMXRB), and which was the first X-ray pulsar ever discovered.

The observations in question were taken by Ginga (originally known as ASTRO-C), a Japanese Space Agency (JAXA) X-ray
telescope that was launched in 1987, using the 'Large Area Proportional Counter' (LAC) instrument. Ginga's
observations of Cen X-3 were taken in 1989 and cover pre- and post-eclipse emission (note that there are more
Ginga observations than we use in this tutorial which cover other phases).

Cen X-3's flux level before eclipse was much lower than the level after eclipse.

<span style="color:purple">Here, we'll use PyXspec to see whether
spectra from these two phases can be fitted with the same model, which differs only in
the amount of absorption.</span>

The spectra and corresponding response matrix were obtained from the HEASARC and are available from
[the full XSPEC walkthrough dataset](https://heasarc.gsfc.nasa.gov/FTP/software/xspec/walkthrough).


### Inputs

- Ginga-LAC pre-eclipse (low-state) spectrum file for Centaurus X-3 – `losum.pha`
- Ginga-LAC post-eclipse (high-state) spectrum file for Centaurus X-3 – `hisum.pha`
- Corresponding Ginga-LAC response file – `ginga_lac.rsp`

### Outputs

- Diagnostic plots showing the simultaneously fitted spectra, models, and residuals for both data groups
- Best-fit model parameters for the linked and partially decoupled models

### Runtime

As of 17th November 2026, this notebook takes ~30-seconds to run to completion on Fornax using the 'small' server with 8GB RAM/2 cores.

## Imports

```{code-cell} python
import contextlib
import os
from collections.abc import Sequence
from urllib.request import urlretrieve

import numpy as np
import xspec as xs
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def plot_fit_residual_spec(
    plot_data: dict | Sequence[dict],
    sp_color: str | Sequence[str] = ("teal", "magenta", "navy", "darkorange"),
    sp_marker: str | Sequence[str] = ("x", "o", "D", "+"),
    mod_color: str | Sequence[str] = (
        "crimson",
        "black",
        "darkorchid",
        "peru",
    ),
    mod_ls: str | Sequence[str | tuple] = (
        "solid",
        "dashed",
        (0, (3, 1, 1, 1)),
        "dotted",
    ),
    mod_lw: float | Sequence[float] = 1.8,
    res_color: str | Sequence[str] | None = None,
    x_lims: tuple[float, float] | None = None,
    y_lims: tuple[float, float] | None = None,
    inst_name: str | Sequence[str] | None = None,
    stepped_model: bool = True,
    mod_expr: str | Sequence[str] | None = None,
    x_scale: str = "log",
    y_scale: str = "log",
    fig_size: tuple[float, float] | None = None,
):
    """
    A convenience function used to plot one or more spectra, fitted models, and
    residuals, at various points in this demonstration. The required input is either
    a single dictionary (of the style constructed in `pyxspec-basics.md`) or a
    sequence of such dictionaries when plotting multiple simultaneously fitted
    spectra/groups.

    Limited customization of the output figure is offered, but this is not intended
    as a truly general-purpose plotting function, more as a possible inspiration
    for your own versions.

    :param dict | Sequence[dict] plot_data: Dictionary (or sequence of
        dictionaries) containing all information necessary to produce the fitted
        spectrum and residual visualization for each spectrum/group.
    :param str | Sequence[str] sp_color: Matplotlib color (or sequence of colors)
        to use for the spectral data points of each dataset.
    :param str | Sequence[str] sp_marker: Matplotlib marker (or sequence of markers)
        to use for the spectral data points of each dataset.
    :param str | Sequence[str] mod_color: Matplotlib color (or sequence of colors)
        to use for the fitted model line of each dataset.
    :param str | Sequence[str | tuple] mod_ls: Matplotlib linestyle (or
        sequence of linestyles) to use for the fitted model line of each dataset.
    :param float | Sequence[float | tuple] mod_lw: Matplotlib linewidth (or
        sequence of linewidths) to use for the fitted model line of each dataset.
    :param str | Sequence[str] | None res_color: Matplotlib color (or sequence
        of colors) to use for the residual data points. Defaults to matching
        `sp_color` if None.
    :param tuple[float, float] | None x_lims: Optional limits on which parts
        of the x-axis to plot. Must be a two-element tuple containing the lower and
        then the upper limit.
    :param tuple[float, float] | None y_lims: Optional limits on which parts
        of the y-axis to plot. Must be a two-element tuple containing the lower and
        then the upper limit.
    :param str | Sequence[str] | None inst_name: Optionally, a
        mission/instrument or dataset label (or sequence of labels, one per dictionary
        in `plot_data`) to add to the legend label given to the spectral data points.
    :param bool stepped_model: Controls whether the fitted model is plotted as a
        staircase (to match XSPEC's plotting style) or as a smooth line. Default is
        True, resulting in a staircase.
    :param str | Sequence[str] | None mod_expr: Optionally, the 'expression' of
        the fitted model (or sequence of expressions/labels) - to be added to its
        legend label.
    :param str x_scale: Scaling that should be applied to the x-axis, accepts all
        matplotlib-supported scales. Default is "log".
    :param str y_scale: Scaling that should be applied to the y-axis, accepts all
        matplotlib-supported scales. Default is "log".
    :param tuple[float, float] | None fig_size: Defines the size of the figure produced
        by this function. Default is None, in which case (7.5, 6.5) is used.
    """
    # Normalize a single dictionary input into a one-element list so that single-
    #  and multi-spectrum plotting follow the exact same code path.
    if isinstance(plot_data, dict):
        plot_data_list = [plot_data]
    elif isinstance(plot_data, (list, tuple)) and len(plot_data) > 0:
        plot_data_list = list(plot_data)
    else:
        raise TypeError(
            "plot_data must be a dictionary or a non-empty list of dictionaries."
        )

    num_spec = len(plot_data_list)

    # Some basic checks to make sure each plot data dictionary is in the right format
    req_ents = [
        "energy",
        "energy_delta",
        "rate",
        "rate_err",
        "model",
        "residual",
        "residual_err",
        "energy_step",
    ]
    for idx, p_data in enumerate(plot_data_list):
        if any([en not in p_data for en in req_ents]):
            raise KeyError(
                f"Plot data entry {idx} must contain the following keys: "
                f"{', '.join(req_ents)}"
            )

    # Basic validity check on any axis limits
    if x_lims is not None and (len(x_lims) != 2 or np.diff(x_lims) < 0):
        raise ValueError(
            "Passed x-axis limits must be a two-element tuple, with the first "
            "entry less than the second."
        )
    if y_lims is not None and (len(y_lims) != 2 or np.diff(y_lims) < 0):
        raise ValueError(
            "Passed y-axis limits must be a two-element tuple, with the first "
            "entry less than the second."
        )

    # Helper to broadcast scalar or sequence styling arguments across datasets
    def _broadcast_arg(arg_val, n_items: int):
        if isinstance(arg_val, (list, tuple)):
            return [arg_val[i % len(arg_val)] for i in range(n_items)]
        return [arg_val] * n_items

    sp_colors = _broadcast_arg(sp_color, num_spec)
    sp_markers = _broadcast_arg(sp_marker, num_spec)
    mod_colors = _broadcast_arg(mod_color, num_spec)
    mod_lstyles = _broadcast_arg(mod_ls, num_spec)
    mod_lwidths = _broadcast_arg(mod_lw, num_spec)
    res_colors = sp_colors if res_color is None else _broadcast_arg(res_color, num_spec)
    mod_exprs = _broadcast_arg(mod_expr, num_spec)
    inst_names = _broadcast_arg(inst_name, num_spec)

    if fig_size is None:
        fig_size = (7.5, 6.5)

    fig, ax_arr = plt.subplots(
        nrows=2, figsize=fig_size, height_ratios=(3, 1.5), sharex=True
    )
    # Shrink the vertical gap between the panels to zero
    fig.subplots_adjust(hspace=0)

    # First axis (the large, top-most one) is where we will plot the spectrum
    #  data points, and fitted model lines.
    spec_ax = ax_arr[0]
    spec_ax.minorticks_on()
    spec_ax.tick_params(which="both", direction="in", top=True, right=True)

    res_ax = ax_arr[1]
    res_ax.minorticks_on()
    res_ax.tick_params(which="both", direction="in", top=True, right=True)

    for idx, p_data in enumerate(plot_data_list):
        cur_inst = inst_names[idx]
        cur_mod_expr = mod_exprs[idx]

        # Determine what the label for spectrum data points should be based on input
        #  instrument/dataset name
        if cur_inst is None:
            sp_label = "Spectral data" if num_spec == 1 else f"Group {idx + 1} data"
        elif num_spec > 1 and isinstance(inst_name, str):
            sp_label = f"{cur_inst} (Group {idx + 1}) data"
        else:
            sp_label = f"{cur_inst} data"

        # Same as above, but for the model label
        if cur_mod_expr is None:
            mod_label = (
                "Fitted model" if num_spec == 1 else f"Group {idx + 1} fitted model"
            )
        elif num_spec > 1 and isinstance(mod_expr, str):
            mod_label = f"Group {idx + 1} fitted model ({cur_mod_expr})"
        else:
            mod_label = f"Fitted model ({cur_mod_expr})"

        # First we plot the spectrum data points, including the count rate uncertainty,
        #  and the size of each energy bin as error bars.
        spec_ax.errorbar(
            p_data["energy"],
            p_data["rate"],
            xerr=p_data["energy_delta"],
            yerr=p_data["rate_err"],
            fmt=sp_markers[idx],
            capsize=1.5,
            label=sp_label,
            color=sp_colors[idx],
        )

        # If the user has requested that the model fit be shown as a 'stepped' line
        #  (i.e. what the standard XSPEC plots look like), we have to use the 'stairs'
        #  method rather than the standard 'plot' method.
        if stepped_model:
            spec_ax.stairs(
                p_data["model"],
                p_data["energy_step"],
                baseline=None,
                fill=False,
                color=mod_colors[idx],
                linestyle=mod_lstyles[idx],
                alpha=0.85,
                label=mod_label,
                linewidth=mod_lwidths[idx],
                zorder=3,
            )
        # Otherwise, the model will be plotted as a smooth line.
        else:
            spec_ax.plot(
                p_data["energy"],
                p_data["model"],
                color=mod_colors[idx],
                linestyle=mod_lstyles[idx],
                label=mod_label,
                alpha=0.85,
                linewidth=mod_lwidths[idx],
                zorder=3,
            )

        res_ax.errorbar(
            p_data["energy"],
            p_data["residual"],
            xerr=p_data["energy_delta"],
            yerr=p_data["residual_err"],
            fmt=sp_markers[idx],
            capsize=1.5,
            color=res_colors[idx],
        )

    # We allow the user to set specific x and y axis limits when they call this
    #  function - if they have passed limits, we enforce them here (the residual
    #  axis will inherit the limits as well, because we set sharex=True when
    #  we defined the figure.
    if x_lims is not None:
        spec_ax.set_xlim(x_lims)
    if y_lims is not None:
        spec_ax.set_ylim(y_lims)

    # The user may pass the y and x scale values separately; the default for
    #  both is log.
    spec_ax.set_yscale(y_scale)
    # Alter the formatting of the labels so that they are 0.1, 0.01, 0.001 etc.
    spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
    # And make sure to set the y-axis label
    spec_ax.set_ylabel(
        r"Spectrum [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
    )

    spec_ax.legend(fontsize=12)

    res_ax.axhline(0, color="goldenrod", linestyle="dashed")

    res_ax.set_xlabel("Energy [keV]", fontsize=15)
    res_ax.set_ylabel(
        r"Residuals [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
    )

    # Users may pass x and y scale values separately, the default for both is log.
    res_ax.set_xscale(x_scale)
    res_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
    res_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

    plt.show()
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
SRC_NAME = "Cen X-3"

REMOTE_DATA_ROOT_URL = "https://heasarc.gsfc.nasa.gov/FTP/software/xspec/walkthrough/"
REQUIRED_FILES = ["losum.pha", "hisum.pha", "ginga_lac.rsp"]

# We are producing multiple takes on the same plot, so we define custom colors, marker
#  shapes, and figure size here to make changing them over all plots easier
MARKER_COLORS = ["tab:cyan", "magenta"]
MARKER_SHAPES = ["x", "d"]
FIGURE_SIZE = (9.5, 7.5)
```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# ------------- Setting how many cores we can use --------------
# We use a service called CircleCI to execute, test, and validate these notebooks
#  as we're writing and maintaining them. Unfortunately we have to treat the
#  determination of the number of cores we can use differently, as the
#  'os.cpu_count()' call will return the number of cores of the host machine, rather
#  than the number that have actually been allocated to us.
if "CIRCLECI" in os.environ and bool(os.environ["CIRCLECI"]):
    # Here we read the CPU quota (total CPU time allowed) and the CPU period (how
    #  long the scheduling window is) from a cgroup (a linux kernel feature) file.
    # Dividing one by t'other provides the number of cores we've been allocated.
    with open("/sys/fs/cgroup/cpu.max", "r") as cpu_maxo:
        quota, period = cpu_maxo.read().strip().split()
        NUM_CORES = int(quota) // int(period)

# If you, the reader, are running this notebook yourself, this is the
#  part that is relevant to you - you can override the default number of cores
#  used by setting this variable to an integer value.
else:
    NUM_CORES = None

# Determines the number of CPU cores available
total_cores = os.cpu_count()

# If NUM_CORES is None, then we use the number of cores returned by 'os.cpu_count()'
if NUM_CORES is None:
    NUM_CORES = total_cores
# Otherwise, NUM_CORES has been overridden (either by the user, or because we're
#  running on CircleCI, and we do a validity check.
elif not isinstance(NUM_CORES, int):
    raise TypeError(
        "If manually overriding 'NUM_CORES', you must set it to an integer value."
    )
elif isinstance(NUM_CORES, int) and NUM_CORES > total_cores:
    raise ValueError(
        f"If manually overriding 'NUM_CORES', the value must be less than or "
        f"equal to the total available cores ({total_cores})."
    )
# --------------------------------------------------------------

# -------------- Set paths and create directories --------------
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = "../../../_data/PyXspec/Ginga"
else:
    ROOT_DATA_DIR = "PyXspec/Ginga"

# Ensure that the specified directory is absolute
ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)
# --------------------------------------------------------------

# ------------- Download demonstration data files --------------
# Download the necessary files from the XSPEC walkthrough dataset.
for req_file in REQUIRED_FILES:
    ret = urlretrieve(
        os.path.join(REMOTE_DATA_ROOT_URL, req_file),
        os.path.join(ROOT_DATA_DIR, req_file),
    )
# --------------------------------------------------------------

# We set XSPEC's plot device to "/null", which means that it won't output figures at
#  all - instead we'll construct our own figures with matplotlib, extracting data from
#  PyXspec.
xs.Plot.device = "/null"
```

***

## 1. Handling multiple spectra and their models


### Loading multiple spectra into separate 'data groups'

This kind of fitting relies on introducing an extra dimension, the group, to the
indexing of the data files. The files in each group share the same model but not
necessarily the same parameter values, which may be shared as common to all the
groups or varied separately from group to group. Although each group may contain more
than one file, there is only one file in each of the two groups in this example.

```{important}
Unlike in the first '{doc}`PyXspec basics <pyxspec-basics>`' tutorial, where we defined
a PyXspec `Spectrum` object directly to load our spectrum, here we are using `xs.AllData()`
to load our spectra all at once.

This is a less 'Pythonic' interface, but mirrors the syntax used by command-line XSPEC.

Unlike loading using `Spectrum`, `xs.AllData()` **does not** return a `Spectrum`
object; instead it returns Python's null object, `None`.
```

The spectral files we need for this demonstration were downloaded in
the [Global Setup: Configuration](#configuration) section. As they were not downloaded to the
same directory as the notebook, so we will briefly change our working directory (using
the `chdir` context manager) as we load them.

Of course, we could pass full spectrum paths rather than changing
directories, but the 'RESPFILE' entry in each spectrum's header is a path relative
to the location of the spectrum file. As such, if we didn't change directories, PyXspec
would have been unable to find and automatically load the response file.

We load our two spectra into PyXspec, with `1:1 losum.pha` reading the _pre-eclipse_
spectrum into **group one: file number one**, and `2:2 hisum.pha` reading the
_post-eclipse_ spectrum into **group two: file number two**:

```{code-cell} python
with contextlib.chdir(ROOT_DATA_DIR):
    xs.AllData("1:1 losum.pha 2:2 hisum.pha")
```

```{tip}
The call to `AllData(...)` above should display a summary of the spectra currently in
use (unless you have PyXspec's 'chatter' set to $<10$ using `xs.Xset.chatter=`).

You can also run `xs.AllData.show()` to see that summary at any time (again, if chatter
has been set high enough).
```

### File and data group numbers

Here, the first group makes up the file `losum.pha`, which contains the spectrum of all
the low-flux, pre-eclipse, emission. The second group makes up the second file, `hisum.pha`, which
contains all the high-flux, post-eclipse, emission.

Note that the file number is "absolute" in the sense that it is independent of the group
number. For instance, if you wanted three files in each of the two groups:
- <span style="color: #1f77b4; font-weight: bold;">Group One</span> - `lo1.pha` (<span style="color: #d62728; font-weight: bold;">file 1</span>), `lo2.pha` (<span style="color: #2ca02c; font-weight: bold;">file 2</span>), and `lo3.pha` (<span style="color: #9467bd; font-weight: bold;">file 3</span>)
- <span style="color: #ff7f0e; font-weight: bold;">Group Two</span> - `hi1.pha` (<span style="color: #8c564b; font-weight: bold;">file 4</span>), `hi2.pha` (<span style="color: #e377c2; font-weight: bold;">file 5</span>), and `hi3.pha` (<span style="color: #17becf; font-weight: bold;">file 6</span>)

You would load them as:

<div class="highlight"><pre><span></span><code>xs.AllData("<span style="color: #1f77b4; font-weight: bold;">1</span>:<span style="color: #d62728; font-weight: bold;">1</span> lo1.pha <span style="color: #1f77b4; font-weight: bold;">1</span>:<span style="color: #2ca02c; font-weight: bold;">2</span> lo2.pha <span style="color: #1f77b4; font-weight: bold;">1</span>:<span style="color: #9467bd; font-weight: bold;">3</span> lo3.pha <span style="color: #ff7f0e; font-weight: bold;">2</span>:<span style="color: #8c564b; font-weight: bold;">4</span> hi1.pha <span style="color: #ff7f0e; font-weight: bold;">2</span>:<span style="color: #e377c2; font-weight: bold;">5</span> hi2.pha <span style="color: #ff7f0e; font-weight: bold;">2</span>:<span style="color: #17becf; font-weight: bold;">6</span> hi3.pha")</code></pre></div>


### Setting up

We want the x-axis of every spectrum we visualize throughout this tutorial to be in terms of
energy, in units of keV, so we make sure to set that up here:

```{code-cell} python
xs.Plot.xAxis = "keV"
```

```{note}
We set XSPEC's plot device to "/null" in the [Global Setup: Configuration](#configuration) section, which
means that it won't output figures - instead we construct our own figures with matplotlib, extracting data
from PyXspec.
```


### Examining the spectra

If you're working in a Jupyter notebook and are likely to be making multiple versions
of an XSPEC plot, we recommend storing plot data in dictionaries, as we do below.
This helps reduce the risk of accidentally re-using variable names and overwriting their
existing values, or using plot data from a previous figure. As Jupyter notebooks can
be run out of order, or have cells re-executed, this is an important consideration.

We used this same approach to storing plotting information in the {doc}`first PyXspec tutorial <pyxspec-basics>`,
where we also talked more about the types of plotting data you can extract from PyXspec, and how to do it.

In _this_ tutorial, we have two spectra in two different data groups – that changes how we retrieve the
plotting information slightly. You can see below that when we call PyXspec functions (such as `xs.Plot.x(...)`), we
are passing a `plotGroup=` argument - the value passed is the index number (remembering XSPEC indexing starts
at **one not zero**) of the data group who's plotting information we wish to retrieve (**1** for the
pre-eclipse spectrum and **2** for the post-eclipse).

You'll note that we also pass `plotWindow=1` to every `xs.Plot.<>` call - this is because we are only
asking PyXspec to generate plotting data for one 'window', the _data_ (`xs.Plot("data")`). In
<a href="#vis-all-tied">the next section</a>, where we have fitted a model and want to plot
the residuals as well, we will have a second plot window to retrieve data from:

```{code-cell} python
# Fetch the plotting information for the spectra
xs.Plot("data")

raw_spec_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(plotGroup=grp_id, plotWindow=1)),
        "energy_delta": np.array(xs.Plot.xErr(plotGroup=grp_id, plotWindow=1)),
        "rate": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=1)),
        "rate_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=1)),
        "x_label": xs.Plot.labels(plotWindow=1)[0],
        "y_label": xs.Plot.labels(plotWindow=1)[1],
        "title": xs.Plot.labels(plotWindow=1)[2],
    }
    raw_spec_plot_data.append(grp_data)
```

Now, we can visualize our two spectra before we ignore any channels or fit any models:

<a id="fig-raw-spectra"></a>
```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.75))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

raw_labels = ("Ginga-LAC (pre-eclipse)", "Ginga-LAC (post-eclipse)")

for idx, p_data in enumerate(raw_spec_plot_data):
    sp_label = raw_labels[idx] if idx < len(raw_labels) else f"Group {idx + 1} data"
    plt.errorbar(
        p_data["energy"],
        p_data["rate"],
        xerr=p_data["energy_delta"],
        yerr=p_data["rate_err"],
        fmt=MARKER_SHAPES[idx % len(MARKER_SHAPES)],
        capsize=1.5,
        label=sp_label,
        color=MARKER_COLORS[idx % len(MARKER_COLORS)],
    )

plt.xscale("log")
plt.yscale("log")

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
ax.xaxis.set_minor_formatter(
    FuncFormatter(lambda inp, _: "{:g}".format(inp) if inp >= 1 else "")
)
ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.xlabel(raw_spec_plot_data[0]["x_label"], fontsize=15)
plt.ylabel(
    r"Spectrum [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
```

Looking at the spectra above, there are a few features we notice immediately:
1. Cen X-3 in its _pre-eclipse_ phase had much lower overall flux than in its _post-eclipse_ phase.
2. There is a prominent emission line evident in the _pre-eclipse_ spectrum, right around the energy we would expect for the iron complex. It seems very wide, which is easily explained by the poor (relative to modern missions) spectral/energy resolution of Ginga-LAC.
3. Otherwise, the spectra are quite smooth, with little first-pass visual evidence of other emission lines, absorption edges, or absorption lines.
4. The observed flux drops markedly below ${\sim}1.5-2.0$ keV, which is likely because Ginga-LACs recommended lower bound energy for analysis is 1.5 keV ([Angelini L. and Allen, J.](https://heasarc.gsfc.nasa.gov/docs/ginga/ginga.html)).
5. Somewhat relatedly, the signal-to-noise above ${\sim} 20$ keV is very low, and those portions of the spectra may hamper our fit.

### Excluding parts of the spectra from our fit

As we alluded to above, the whole energy range covered by our Ginga-LAC spectral files
is not necessarily the energy range we wish to perform our analysis in. The lowest
energy that can be analyzed in reality is ${\sim}1.5$ keV; this corresponds roughly to
**channel 3** (1.16–1.74 keV).

The low-signal-to-noise upper energies of our spectra don't seem like they will contribute
much to our fits, so we will drop everything above and including **channel 37** (23.26–24.43 keV).

We covered the ignore command in the '{doc}`PyXspec basics <pyxspec-basics>`' tutorial, and
won't repeat that discussion of the whys and wherefores of ignoring channels. There is one
difference worth mentioning, though – we have multiple spectra loaded.

PyXspec has the capability to ignore different channels of different spectra; this works
on **a file-by-file basis**, rather than at the level of data groups. When _no file IDs
are supplied_ the `xs.AllData.ignore(...)` call applies the channel range to every loaded
spectrum (the input to this function mirrors the syntax of command-line XSPEC):

```{code-cell} python
xs.AllData.ignore("1-3 37-48")
```

You can use the file number (see [our previous discussion](#file-and-data-group-numbers))
to specify which spectrum an ignore command should apply to. For instance, if you wanted
to ignore channels 1–3 and 37–48 for spectrum one, but only ignore channel 37–48 for
spectrum two, you could run `xs.AllData.ignore("1:1-3 37-48 2: 37-48")`.

We can also apply different ignore commands to different ranges of file IDs. Imagine
for a moment that we loaded **four** separate spectra, rather than two, and wanted to
ignore different channels for the first two spectra than for the second two; you could
run `xs.AllData.ignore("1-2:1-3 37-48 3-4: 37-48")`.


### Excluding 'bad' channels

We also remove any channels from our spectra that were previously marked as bad:

```{code-cell} python
xs.AllData.ignore("bad")
```

### Choosing and setting up a model

The model we'll use at first to fit the two files is an absorbed power law with a
high-energy cut-off:

- **tbabs** – The Tuebingen-Boulder interstellar-medium (ISM) absorption model, which is used to model the absorption of X-rays by the ISM. It is a multiplicative model that down weights emissive models to account for the absorption of their emission at different energies.
- **highecut** – This multiplicative model has a constant value of **1** up to the 'cutoff energy'. Beyond that the model value begins to decrease with increasing energy, with the rate of decrease controlled by an 'e-folding energy' parameter.
- **pow** – A shorthand name for one of XSPEC's simplest models; `powerlaw`.

We do not assign the return of the `xs.Model(...)` definition to a variable (as we did
in the {doc}`first PyXspec tutorial <pyxspec-basics>`); PyXspec has set up two separate
instances of the model we specified, one per data group, and only the first is returned.

PyXspec's `AllModels()` model manager will be used to access the model instances later
on in this tutorial.

```{code-cell} python
xs.Model("tbabs * highecut(pow)")
```

<br>

Notice how the summary of the model, displayed immediately above, is different now that
we have two groups, as opposed to one (compared with the summaries we saw in the
{doc}`first PyXspec tutorial <pyxspec-basics>`).

We can see that though there are **10** model parameters in total, they are split so that
**5** are associated with each data group.

You might also notice that the second data group's model parameters are **linked to those
of the first** – this is made evident by the value column entries for group two ending in
`= p<N>`, where **N** is the parameter ID (first column) of the corresponding parameter in
the first group's model.

```{tip}
Setting up the model above should also display an overview of its parameters currently in
use (unless you have PyXspec's 'chatter' set too low, see [the previous note on viewing a summary of loaded spectra](#loading-multiple-spectra-into-separate-data-groups)).

You can also run `xs.AllModels.show()` to see that summary at any time.
```


## 2. Fitting the exact same model to both spectra

```{admonition} Hypothesis
:class: hint

We'll start with the (clearly incorrect, as the flux levels are very different) hypothesis
that both spectra represent the exact same emission/absorption state, with the same
physical processes behind them.
```

<a id="setup-all-tied"></a>
### Setting up

```{code-cell} python
xs.Fit.nIterations = 50
```

````{tip}
At this point we _could_ configure PyXspec to use multiple cores when performing the model fit:

```python
xs.Xset.parallel.leven = NUM_CORES
```

_However_, the data and models we're using in this demonstration are so simple (meaning
very few data points in the spectra, and very few parameters in the models) that making
XSPEC use multiple cores for the model fit would actually _make the fit take longer_.
````

<a id="fit-all-tied"></a>
### Performing the fit

Our first demonstration is to fit the model exactly as we defined it, meaning that we've
left the model configuration just as it was when it was declared. This means:
- We're using default starting values for fit parameters.
- Every parameter in the second (post-eclipse) spectrum's model is tied to its counterpart in the first spectrum's model.

We run the fit (of both spectra) by calling `xs.Fit.perform()`:

```{code-cell} python
xs.Fit.perform()
```

<br>

Looking through the fit summary shown above (you may need to scroll if the output cell
has been partially collapsed), we can see that all the fitted parameters for the second
data group (i.e. the post-eclipse spectrum) are identical to those for the first data group.

We can also see that **this is an absolutely dreadful fit** – with a total fit statistic $\gt 2\times10^{5}$, and 61 degrees of freedom (DOF), leading to a reduced-$\chi^{2}$ of $\gt 3000$.

From this, we can safely say that our first hypothesis was, as we thought, doomed from the start. These two spectra _do not_ represent Cen X-3 at the exact same emission phase.

<a id="vis-all-tied"></a>
### Visualizing the fitted model

To emphasize just how bad this first fit was, we will visualize it. Just as in the
<a href="#fig-raw-spectra">first plot of our spectra</a> (without model fits), we store the
plotting information in dictionaries. Unlike that first visualization, we also
wish to plot the model residuals, so we call `xs.Plot("data resid")` (as opposed to
`xs.Plot("data")`), and extract the residual information from `plotWindow=2`:

```{code-cell} python
# Fetch the plotting information for the spectra, models, and model residuals
xs.Plot("data resid")

all_tied_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(plotGroup=grp_id, plotWindow=1)),
        "energy_delta": np.array(xs.Plot.xErr(plotGroup=grp_id, plotWindow=1)),
        "rate": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=1)),
        "rate_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=1)),
        "model": np.array(xs.Plot.model(plotGroup=grp_id, plotWindow=1)),
        "residual": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=2)),
        "residual_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=2)),
    }
    grp_data["energy_step"] = np.append(
        grp_data["energy"] - grp_data["energy_delta"],
        grp_data["energy"][-1] + grp_data["energy_delta"][-1],
    )
    all_tied_plot_data.append(grp_data)
```

Over the course of this tutorial we're going to be plotting the spectra, fitted models, and
model residuals several times. As such, we've implemented a small convenience function in the
[Global Setup: Functions](#functions) section near the beginning of the notebook - `plot_fit_residual_spec()`.

This function is very similar to the one we defined in the {doc}`first PyXspec tutorial <pyxspec-basics>`, but
in this case we've adapted it so that it can plot _multiple_ spectra, models, and residuals, rather than just one.

We encourage you to look through the function definition to get an idea of how you could create similar figures.

Note that we are plotting these spectra and models with **linear** scaling applied to the y-axis:

<a id="fig-all-tied"></a>
```{code-cell} python
# Call the plotting convenience function
plot_fit_residual_spec(
    all_tied_plot_data,
    inst_name=["Ginga-LAC (pre-eclipse)", "Ginga-LAC (post-eclipse)"],
    mod_expr=["1: Defaults", "2: Defaults"],
    y_scale="linear",
    fig_size=FIGURE_SIZE,
    sp_color=MARKER_COLORS,
    sp_marker=MARKER_SHAPES,
)
```

```{important}
In this circumstance, the model lines produced by XSPEC are identical, because the models for both
spectra have identical parameters **and** the response file used by both spectra is the same. If
these two spectra were from different instruments, or even from two different points in the lifetime
of the _same_ instrument, there would be two distinct (though not that distinct) model lines shown.

This is because XSPEC models are 'folded through' the responses of the instruments whose data they are fitting.
```

## 3. Fitting the model with nH allowed to vary separately for both spectra

```{admonition} Hypothesis
:class: hint

Let us suppose that we have a theory that the difference between the two spectra **is a significant change in absorption**.
```

<a id="setup-untied-nh"></a>
### Setting up

We redeclare the model that we chose [earlier in the tutorial](#choosing-and-setting-up-a-model) to wipe the previous fit:

```{code-cell} python
xs.Model("tbabs * highecut(pow)")
```

<br>

What we want to do now is to **remove the link between the nH parameters of the first and
second data groups**; looking at the model summary above, you should be able to see that
the parameter IDs for those nH variables are **1** and **6**.

In PyXspec, there are several ways to break a parameter link, or 'untie', in XSPEC terminology:
1. Call the `<model variable>.<component model>.<parameter name>.untie()` method (e.g. `spec_two_mod.TBabs.nH.untie()`) to unlink a specific parameter. It is important to note that XSPEC 'ties' are **directional**; for instance, in the model summary above, the second model's parameters are tied to the first models. If you called `untie()` on any of the first model's parameters, nothing would change.
2. Be less targeted and call the `<model variable>.untie()` (e.g. `spec_two_mod.untie()`); this will untie _every parameter of that model_.
3. Assign a value to a parameter that is currently tied to another parameter (e.g. `spec_two_mod.TBabs.nH = 1.0`); as a linked parameter does not have a value of its own (instead it mirrors the value of the parameter it is linked to), this implicitly breaks the link.

In this case, we're going to use the third method – assigning a value to the nH parameter of the
second data group model. We're also going to assign a much larger nH starting value to the
pre-eclipse spectrum model (group 1), as our current hypothesis is that we can account for
the differences between spectra by modeling significantly more absorption.

First, we have to retrieve the PyXspec model instances for the first and second data groups; this is
as simple as passing their data group ID (remembering that **XSPEC indexes from 1**, whereas
**Python indexes from 0**) to `xs.AllModels(...)`:

```{code-cell} python
# The pre-eclipse spectrum's model (data group 1)
spec_one_mod = xs.AllModels(1)

# The post-eclipse spectrum's model (data group 2)
spec_two_mod = xs.AllModels(2)
```

Here, we'll enter the absorption column of the first group of data files (losum.pha; pre-eclipse)
as $100\times10^{22}\: \mathrm{cm}^{-2}$, leaving default values set for all the other parameters:

```{code-cell} python
spec_one_mod.TBabs.nH = 100.0
```

<br>

Now, when it comes to the second group of parameters, we enter a column of $1\times10^{22}\:
\mathrm{cm}^{-2}$, and again leave default values for the other parameters (this has untied
the second data group model's nH from the first data group model's:

```{code-cell} python
spec_two_mod.TBabs.nH = 1.0
```

<br>

```{seealso}
You can also access a model's parameters by passing the component index. For instance,
the `spec_two_mod.TBabs.nH = 1.0` call above could be written as `spec_two_mod(1).values = 1.0`.

However, **we do recommend accessing through component model and parameter names as in
the code cell above**, as it is much easier to read and recall what is being untied.
```

<a id="fit-untied-nh"></a>
### Performing the fit

We run the fit for our new definition of the model, with nH parameters untied and allowed
to vary independently, and data group 1's nH starting value much higher than data group 2's:

```{code-cell} python
xs.Fit.perform()
```

<br>

We can see that this version of the model fit achieved a total fit statistic
$\gt 8\times10^{4}$, and 60 degrees of freedom (DOF), leading to a
reduced-$\chi^{2}$ of $\gt 1300$.

While certainly an improvement on <a href="#fit-all-tied">our first attempt</a>, we cannot
expect the models to closely follow the spectra's data points when we visualize them
in the next subsection.

<a id="vis-untied-nh"></a>
### Visualizing the fitted model

```{code-cell} python
# Fetch the plotting information for the spectra, models, and model residuals
xs.Plot("data resid")

untied_nh_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(plotGroup=grp_id, plotWindow=1)),
        "energy_delta": np.array(xs.Plot.xErr(plotGroup=grp_id, plotWindow=1)),
        "rate": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=1)),
        "rate_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=1)),
        "model": np.array(xs.Plot.model(plotGroup=grp_id, plotWindow=1)),
        "residual": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=2)),
        "residual_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=2)),
    }
    grp_data["energy_step"] = np.append(
        grp_data["energy"] - grp_data["energy_delta"],
        grp_data["energy"][-1] + grp_data["energy_delta"][-1],
    )
    untied_nh_plot_data.append(grp_data)
```

<a id="fig-untied-nh"></a>
```{code-cell} python
plot_fit_residual_spec(
    untied_nh_plot_data,
    inst_name=["Ginga-LAC (pre-eclipse)", "Ginga-LAC (post-eclipse)"],
    mod_expr=["1: untied nH", "2: untied nH"],
    y_scale="linear",
    fig_size=FIGURE_SIZE,
    sp_color=MARKER_COLORS,
    sp_marker=MARKER_SHAPES,
)
```

## 4. Fitting the model with nH and powerlaw normalization allowed to vary separately for both spectra

```{admonition} Hypothesis
:class: hint

Perhaps scattering also plays a role in reducing the flux before eclipse?
```

Our assumption that only a change in absorption can account for the spectral
variation before and after eclipse is clearly wrong.

Perhaps scattering also plays a role in reducing the flux before eclipse.

This could be modeled (simply at first) by
allowing the normalization of the power law to be smaller before eclipse than after
eclipse.

To decouple tied parameters, we can change the parameter value in the second
group to a value – any value – different from that in the first group (changing the
value in the first group has the effect of changing both without decoupling).

<a id="setup-untied-nh-norm"></a>
### Setting up

Once again, we redeclare the model that we chose [earlier in the tutorial](#choosing-and-setting-up-a-model)
to wipe the previous fit:

```{code-cell} python
xs.Model("tbabs * highecut(pow)")
```

<br>

We repeat the same nH parameter setup process that we used in <a href="#setup-untied-nh">the last fit</a>:

```{code-cell} python
spec_one_mod = xs.AllModels(1)
spec_two_mod = xs.AllModels(2)

spec_one_mod.TBabs.nH = 100.0
spec_two_mod.TBabs.nH = 1.0
```

<br>

Now we come to what we're changing this time around – the powerlaw normalization. Specifically,
we're untying the normalization of the second data group model's powerlaw component so that
it can vary independently of the first data group's equivalent:

```{code-cell} python
spec_two_mod.powerlaw.norm = 1.0
```

<br>

<a id="fit-untied-nh-norm"></a>
### Performing the fit

We run the fit for our new definition of the model, with nH and powerlaw normalization
parameters untied and allowed to vary independently, and again with data group 1's nH
starting value much higher than data group 2's:

```{code-cell} python
xs.Fit.perform()
```

<br>

This decoupling improves the fit, taking us from a reduced-$\chi^{2}$ of $\gt 1300$ from
the <a href="#fit-untied-nh">last model fit</a> to a reduced-$\chi^{2}$ of $\gt 250$, but we
still clearly see that it **isn't a good fit**.

Indeed, this simple attempt to account for the spectral variability in terms of "blanket" cold absorption
and scattering does not work.

<span style="color:red">More sophisticated models involving additional components and partial absorption should be tried.</span>

<span style="color:red">Point out the fitted parameter differences between Group 1 (`losum`, $N_{\mathrm{H}} \approx 20.6 \times 10^{22}\ \mathrm{cm}^{-2}$, $\mathrm{norm} \approx 0.0596$) and Group 2 (`hisum`, $N_{\mathrm{H}} \approx 1.31 \times 10^{22}\ \mathrm{cm}^{-2}$, $\mathrm{norm} \approx 0.314$), and note that Group 1 (the pre-eclipse spectrum) contributes the vast majority of the remaining $\chi^2$ (~13,993 vs ~1,316).</span>

<a id="vis-untied-nh-norm"></a>
### Visualizing the fitted model

```{code-cell} python
# Fetch the plotting information for the spectra, models, and model residuals
xs.Plot("data resid")

untied_nh_plnorm_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(plotGroup=grp_id, plotWindow=1)),
        "energy_delta": np.array(xs.Plot.xErr(plotGroup=grp_id, plotWindow=1)),
        "rate": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=1)),
        "rate_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=1)),
        "model": np.array(xs.Plot.model(plotGroup=grp_id, plotWindow=1)),
        "residual": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=2)),
        "residual_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=2)),
    }
    grp_data["energy_step"] = np.append(
        grp_data["energy"] - grp_data["energy_delta"],
        grp_data["energy"][-1] + grp_data["energy_delta"][-1],
    )
    untied_nh_plnorm_plot_data.append(grp_data)
```

<a id="fig-untied-nh-norm"></a>
```{code-cell} python
plot_fit_residual_spec(
    untied_nh_plnorm_plot_data,
    inst_name=["Ginga-LAC (pre-eclipse)", "Ginga-LAC (post-eclipse)"],
    mod_expr=["1: untied nH & norm", "2: untied nH & norm"],
    fig_size=FIGURE_SIZE,
    sp_color=MARKER_COLORS,
    sp_marker=MARKER_SHAPES,
)
```

## Fitting a model with additional emission components

```{admonition} Hypothesis
:class: hint

That we are missing a low-temperature thermal emission component, and that there is an unmodelled Fe emission line around 6.4 keV.
```

<a id="setup-add-comps"></a>
### Setting up

Now

```{code-cell} python
xs.Model("tbabs*(highecut(pow) + gauss + bbody)")
```

<br>

```{code-cell} python
ORIGINAL_XS_CHATTER = xs.Xset.chatter
xs.Xset.chatter = 0

spec_one_mod = xs.AllModels(1)
spec_one_mod.TBabs.nH = 100.0

spec_two_mod = xs.AllModels(2)
spec_two_mod.TBabs.nH = 1.0

spec_two_mod.powerlaw.norm = 1.0

spec_one_mod.gaussian.LineE = 6.4
spec_one_mod.gaussian.LineE.frozen = True

spec_two_mod.gaussian.Sigma.untie()
spec_two_mod.gaussian.norm.untie()

xs.Xset.chatter = ORIGINAL_XS_CHATTER
xs.AllModels.show()
```

<br>

<a id="fit-add-comps"></a>
### Performing the fit

```{code-cell} python
xs.Fit.perform()
```

<br>

Our final fit is complete, and the reduced-$\chi^{2}$ is now ${\sim}50$, with 53 degrees of freedom.

Additionally, you will notice that the $\chi^2$ of data group 1 is ${\sim}2400$, and the
$\chi^2$ of data group 2 is ${\sim}200$; the **fit to the second, post-eclipse, spectrum
is considerably worse** than the fit to the first, pre-eclipse, data group.

<a id="vis-add-comps"></a>
### Visualizing the fitted model

```{code-cell} python
# Fetch the plotting information for the spectra, models, and model residuals
xs.Plot("data resid")

untied_nh_norm_gauss_bbody_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(plotGroup=grp_id, plotWindow=1)),
        "energy_delta": np.array(xs.Plot.xErr(plotGroup=grp_id, plotWindow=1)),
        "rate": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=1)),
        "rate_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=1)),
        "model": np.array(xs.Plot.model(plotGroup=grp_id, plotWindow=1)),
        "residual": np.array(xs.Plot.y(plotGroup=grp_id, plotWindow=2)),
        "residual_err": np.array(xs.Plot.yErr(plotGroup=grp_id, plotWindow=2)),
    }
    grp_data["energy_step"] = np.append(
        grp_data["energy"] - grp_data["energy_delta"],
        grp_data["energy"][-1] + grp_data["energy_delta"][-1],
    )
    untied_nh_norm_gauss_bbody_plot_data.append(grp_data)
```

Plotting our final model fits, we can see the impact of adding the two new emission
components – just by eye the lower energy part of the pre-eclipse spectrum is better
represented now we have an additional thermal component.

The visually obvious (more so in the pre-eclipse than the post-eclipse spectrum) emission
line around 6.4 keV is being modeled now that we have the additional gaussian component. Examining
the residuals of the fits, we can see that the emission line seems to be better modeled in the
pre-eclipse spectrum than in the post-eclipse:

<a id="fig-add-comps"></a>
```{code-cell} python
plot_fit_residual_spec(
    untied_nh_norm_gauss_bbody_plot_data,
    inst_name=["Ginga-LAC (pre-eclipse)", "Ginga-LAC (post-eclipse)"],
    mod_expr=["1: original + bbody + gauss", "2: original + bbody + gauss"],
    fig_size=FIGURE_SIZE,
    sp_color=MARKER_COLORS,
    sp_marker=MARKER_SHAPES,
)
```

### Calculating parameter uncertainties

```{code-cell} python
xs.Xset.parallel.error = NUM_CORES
```

```{code-cell} python
try:
    xs.Fit.error("2-3 12-13")
except Exception as err:
    print(err)
```

```{code-cell} python
xs.Fit.error("maximum 50 15")
```

### Highlighting the very low quality of the fit

```{code-cell} python
xs.Xset.parallel.goodness = NUM_CORES
xs.Fit.goodness(1000)
```

## About this notebook

Author: Keith Arnaud, XSPEC Lead, Associate Research Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-11-17

+++

### Additional Resources

Support: [XSPEC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xspec)

[Original PyXspec Jupyter Notebooks Repository](https://github.com/HEASARC/PyXspec-Jupyter-notebooks)

[Full XSPEC walkthrough dataset](https://heasarc.gsfc.nasa.gov/FTP/software/xspec/walkthrough.tar.gz)

[XSPEC Model Manual - `highecut`](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node294.html)

[XSPEC Model Manual - `tbabs`](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node318.html)

[XSPEC Model Manual - `pow`](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node251.html)


### Acknowledgements

### References

[Nagase F., Corbet R.H.D., Day C.S.R., Inoue H., Takeshima T., Yoshida K. Mihara T. (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...396..147N/abstract) - _GINGA Observations of Centaurus X-3_

[Angelini L. and Allen, J. - _HEASARC: The Ginga Observatory_ [ACCESSED 19-NOV-2026]](http://heasarc.gsfc.nasa.gov/docs/ginga/ginga.html)
