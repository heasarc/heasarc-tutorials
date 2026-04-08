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
date: '2026-04-08'
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
title: PyXspec basics - fitting models to data
---

# PyXspec basics - fitting models to data

## Learning Goals

By the end of this tutorial, you will be able to:

- Load X-ray a spectrum and response into PyXspec.
- Define and fit models to X-ray spectra and evaluate their quality.
- Calculate parameter errors and confidence contours.
- Use PyXspec and matplotlib to visualize spectra and model fits.

## Introduction

This example uses some very old spectral data. It's much simpler than more modern
observations and so can be used to better illustrate the basics of XSPEC analysis.

The data in question is a 20 ks EXOSAT 'Medium-Energy' (ME) observation of the
6-second period X-ray pulsar **1E1048.1-5937** by EXOSAT, taken in June 1985.

In this example, we'll conduct a general investigation of the spectrum from the
Medium Energy instrument, i.e. follow the same sort of steps as the original
investigators ([Seward F. D., Charles P. A., Smale A. P. 1986](https://ui.adsabs.harvard.edu/abs/1986ApJ...305..814S/abstract)).

The ME spectrum and corresponding response matrix were obtained from the HEASARC and
are available either as part of a [large collection of example data](https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/walkthrough.tar.gz) or directly
from the URLs defined in the [Global Setup: Constants](#constants) section.

### Inputs

- EXOSAT-ME spectrum file for 1E1048.1-5937 - s54405.pha
- Corresponding EXOSAT-ME response file - s54405.rsp

### Outputs

- Various diagnostic plots showing data, models, and residuals
- Best-fit model parameters with uncertainties
- Flux measurements and confidence ranges
- Upper limit on iron emission line equivalent width

### Runtime

As of 8th April 2026, this notebook takes ~1-minute to run to completion on Fornax using the 'small' server with 8GB RAM/2 cores.

## Imports

```{code-cell} python
import contextlib
import os
from time import sleep
from typing import Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import xspec as xs
from astropy.units import Quantity
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import chi2
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
    plot_data: dict,
    sp_color: str = "navy",
    mod_color: str = "firebrick",
    res_color: str = "navy",
    x_lims: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Tuple[float, float]] = None,
    inst_name: Optional[str] = None,
    stepped_model: bool = True,
    mod_expr: Optional[str] = None,
):
    """
    A convenience function used to plot the spectrum, fitted model, and residuals, at
    various points in this demonstration. The required input is a dictionary of
    the style constructed in various subsections of the 'alternative spectral models'
    section.

    Limited customization of the output figure is offered, but this is not intended
    as a truly general-purpose plotting function, more as a possible inspiration
    for your own versions.

    :param dict plot_data: Dictionary containing all information necessary to produce
        the fitted spectrum and residual visualization.
    :param str sp_color: Matplotlib color to use for the spectral data points.
    :param str mod_color: Matplotlib color to use for the fitted model staircase line.
    :param str res_color: Matplotlib color to use for the residual data points.
    :param Optional[Tuple[float, float]] x_lims: Optional limits on which parts
        of the x-axis to plot. Must be a two-element tuple containing the lower and
        then the upper limit.
    :param Optional[Tuple[float, float]] y_lims: Optional limits on which parts
        of the y-axis to plot. Must be a two-element tuple containing the lower and
        then the upper limit.
    :param str inst_name: Optionally, a mission/instrument name to add to the
        legend label given to the spectral data points.
    :param bool stepped_model: Controls whether the fitted model is plotted as a
        staircase (to match XSPEC's plotting style) or as a smooth line. Default is
        True, resulting in a staircase.
    :param str mod_expr: Optionally, the 'expression' of the fitted model - to be
        added to its legend label.
    """

    # Some basic checks to make sure the plot data is in the right format
    # These are what we need
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
    # Raise an error before we get started plotting if any entries are missing
    if any([en not in plot_data for en in req_ents]):
        raise KeyError(
            f"Plot data must contain the following keys: f{', '.join(req_ents)}"
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

    # Determine what the label for spectrum data points should be based on input
    #  instrument name
    sp_label = "Spectral data" if inst_name is None else f"{inst_name} data"

    # Same as above, but for the model label
    mod_label = "Fitted model" if mod_expr is None else f"Fitted model ({mod_expr})"

    fig, ax_arr = plt.subplots(
        nrows=2, figsize=(7, 6), height_ratios=(3, 1.5), sharex=True
    )
    # Shrink the vertical gap between the panels to zero
    fig.subplots_adjust(hspace=0)

    spec_ax = ax_arr[0]
    spec_ax.minorticks_on()
    spec_ax.tick_params(which="both", direction="in", top=True, right=True)

    spec_ax.errorbar(
        plot_data["energy"],
        plot_data["rate"],
        xerr=plot_data["energy_delta"],
        yerr=plot_data["rate_err"],
        fmt="+",
        capsize=1.5,
        label=sp_label,
        color=sp_color,
    )

    if stepped_model:
        spec_ax.stairs(
            plot_data["model"],
            plot_data["energy_step"],
            baseline=None,
            fill=False,
            color=mod_color,
            alpha=0.8,
            label=mod_label,
            linewidth=1.4,
        )
    else:
        spec_ax.plot(
            plot_data["energy"],
            plot_data["model"],
            color=mod_color,
            label="Fitted model",
            alpha=0.8,
        )

    if x_lims is not None:
        spec_ax.set_xlim(x_lims)
    if y_lims is not None:
        spec_ax.set_ylim(y_lims)

    spec_ax.set_yscale("log")
    spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

    spec_ax.set_ylabel(
        r"Spectrum [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
    )

    spec_ax.legend(fontsize=14)

    res_ax = ax_arr[1]
    res_ax.minorticks_on()
    res_ax.tick_params(which="both", direction="in", top=True, right=True)

    res_ax.errorbar(
        plot_data["energy"],
        plot_data["residual"],
        xerr=plot_data["energy_delta"],
        yerr=plot_data["residual_err"],
        fmt="+",
        capsize=1.5,
        color=res_color,
    )
    res_ax.axhline(0, color="goldenrod", linestyle="dashed")

    res_ax.set_xlabel("Energy [keV]", fontsize=15)
    res_ax.set_ylabel(
        r"Residuals [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
    )

    res_ax.set_xscale("log")
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
SRC_NAME = "1E1048.1-5937"

EXOSAT_ME_SP_BASE_URL = "https://nasa-heasarc.s3.amazonaws.com/exosat/data/me/spectra"
DEMO_SPEC_URL = os.path.join(EXOSAT_ME_SP_BASE_URL, "s54405.pha.Z")
DEMO_RESP_URL = os.path.join(EXOSAT_ME_SP_BASE_URL, "s54405.rsp.Z")
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
    ROOT_DATA_DIR = "../../../_data/PyXSPEC/EXOSAT"
else:
    ROOT_DATA_DIR = "PyXSPEC/EXOSAT"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)
# --------------------------------------------------------------

# ------------- Download demonstration data files --------------
# Download the spectrum and response required for this demonstration.
#  The return value is unimportant, and we only capture it in a variable to
#  avoid Jupyter notebooks from printing it, as these are the last lines of the cell.
# The EXOSAT-ME spectrum
ret = urlretrieve(
    DEMO_SPEC_URL, os.path.join(ROOT_DATA_DIR, os.path.basename(DEMO_SPEC_URL))
)
# The accompanying EXOSAT-ME response
ret = urlretrieve(
    DEMO_RESP_URL, os.path.join(ROOT_DATA_DIR, os.path.basename(DEMO_RESP_URL))
)
# --------------------------------------------------------------
```

***

## 1. Loading a spectrum into PyXspec

The spectral files we need for this demonstration should have been downloaded
in the [Global Setup: Configuration](#configuration) section.

We can read our spectrum file into a PyXspec `Spectrum` object, assigning it
to the `exo_me_spec`variable. **Most** PyXspec operations don't involve
direct interaction with individual spectrum objects, but we will use it to ignore
some channels later in this tutorial.

The spectrum file we are using for this demonstration has not been downloaded to the
same directory as the notebook, so we briefly change our working directory as we load
it.

Of course, we could have passed the full spectrum path rather than changing
directories, but the 'RESPFILE' entry in the spectrum's header is a path relative
to the location of the spectrum file.

If we hadn't changed directories, then PyXspec would have been unable to find and
automatically load the response file (though we could also have passed a response
file path to the optional `respfile` argument of the `Spectrum` constructor).

```{code-cell} python
with contextlib.chdir(ROOT_DATA_DIR):
    exo_me_spec = xs.Spectrum("s54405.pha")
```

## 2. Visualizing the data

One of the first things most users will want to do at this stage - even before fitting
a model - is to look at their data (something we strongly encourage as a first
step of _any_ analysis).

XSPEC can plot a wide variety of different information, all related in some way to
the underlying spectral data, fitted model(s), fit statistics, and the instrument
used to collect the data.

Even though we're using the Python interface to XSPEC and will be creating
visualizations using Matplotlib, we can still take advantage of the backend
XSPEC plotting functionality by retrieving the data necessary to create
a particular figure from PyXspec.

The exact visualization information that can be retrieved from PyXspec will depend on
what you're plotting, but here follow some examples of what can be fetched:
- X and Y data points.
- X and Y uncertainties.
- Axis labels
- Plot title

### What plots can XSPEC generate?

To list the types of plots that PyXspec can generate, we can run:

```{code-cell} python
xs.Plot("?")
```

### Setting XSPEC's plot device

Before we actually plot something, we need to set the 'plotting device' that XSPEC's
graphics library tries to output to. When running XSPEC 'traditionally' (i.e. through
the command line), then your choice of plot device will control whether a figure is
written to a file or displayed in a window.

Here, as we're going to be doing the plotting ourselves using matplotlib, we don't
really want either of those things to happen. As such, we set the plot device
to "/null":

```{code-cell} python
xs.Plot.device = "/null"
```

```{danger}
Skipping this step could result in many files being written to your current
directory, or the wholesale failure of all plotting in your notebook.
```

### Plotting a spectrum

The **data** plot option produces the most important of all XSPEC visualizations - a
spectrum! By default, spectra will be plotted as a function of
**instrument channel**, which is the most fundamental indication of the energy
of an X-ray event (which hopefully corresponds to a _photon_ hitting the detector).

However, if an instrument response has been loaded along with the spectral data (we
made sure of that [Section 1](#1-loading-a-spectrum-into-pyxspec)), then we can plot
the much more useful spectrum as a function of **energy**.

Note, however, that this won't happen automatically just because a response is
available; we have to specify the units of the energy axis explicitly:

```{code-cell} python
xs.Plot.xAxis = "keV"
```

As we've already mentioned, we're going to use the matplotlib Python module to
construct our figures and visualizations - it will give us a great deal of flexibility
and more control over the final plot than if we just used XSPEC's built-in plotting
functionality.

Our first task is to use PyXspec to produce the rate, energy, and uncertainty
information that makes up a spectrum visualization and then to retrieve the data
for later use.

Here we call the `Plot` method, passing **"data"** to specify which type of plot XSPEC
should produce (in this case a spectrum). Note that **preparing** and **retrieving** the
data necessary to visualize a PyXspec plot with Matplotlib are **two distinct steps**:

1. Running `Plot("data")` will recalcuate plot quantities based on the current data, noticed channels, model fit (if any), etc.
2. Calling `Plot.x()` or `Plot.y()` will **fetch** the most current calculated quantities.

You will also notice that PyXspec very handily provides the axis labels and title that
it would use if it were making the plot itself, we store those too:

```{code-cell} python
xs.Plot("data")

spec_plot_data = {
    "energy": np.array(xs.Plot.x()),
    "energy_delta": np.array(xs.Plot.xErr()),
    "rate": np.array(xs.Plot.y()),
    "rate_err": np.array(xs.Plot.yErr()),
    "x_label": xs.Plot.labels()[0],
    "y_label": xs.Plot.labels()[1],
    "title": xs.Plot.labels()[2],
}
```

```{tip}
If you're working in a Jupyter notebook, and are likely to be making multiple versions
of XSPEC plots, we recommend storing plot data in a dictionary, as we have
demonstrated above.

This helps reduce the risk of accidentally re-using variable names and overwriting their
existing values, or using plot data from a previous figure. As Jupyter notebooks can
be run out of order, or have cells re-executed, this is an important consideration.
```

Now we can use that information to construct a figure showing the spectrum, prior to
any fitting or energy limits, whilst making some small customizations to improve
the appearance and clarity of the plot.

In this particular case, this includes:
- Logging the energy axis; `plt.xscale("log")`
- Configuring axis ticks to point inwards and be present on the top and right of the figure; `plt.tick_params(which="both", direction="in", top=True, right=True)`
- Removing labels from minor ticks on the energy axis if the values are below 1 keV, to avoid colliding labels; `ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp) if inp >= 1 else ""))`

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(7, 4.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.errorbar(
    spec_plot_data["energy"],
    spec_plot_data["rate"],
    xerr=spec_plot_data["energy_delta"],
    yerr=spec_plot_data["rate_err"],
    fmt="+",
    label="EXOSAT-ME data",
    color="navy",
)

plt.xscale("log")

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
ax.xaxis.set_minor_formatter(
    FuncFormatter(lambda inp, _: "{:g}".format(inp) if inp >= 1 else "")
)

plt.xlabel(spec_plot_data["x_label"], fontsize=15)
plt.ylabel(spec_plot_data["y_label"], fontsize=15)

plt.title(spec_plot_data["title"], fontsize=16)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
```

## 3. Defining and fitting models

We are now ready to fit the data with a model. Models in XSPEC are specified using
the model command, followed by an algebraic expression of a combination of model
components. There are two basic kinds of model components: additive and multiplicative.

Additive components represent X-Ray sources of different kinds (e.g., a bremsstrahlung
continuum) and, after being convolved with the instrument response, prescribe the
number of counts per energy bin. Multiplicative components represent phenomena that
modify the observed X-Radiation (e.g. reddening or an absorption edge). They apply an
energy-dependent multiplicative factor to the source radiation before the result is
convolved with the instrumental response.

More generally, XSPEC allows three types of modifying components: convolutions and
mixing models in addition to the multiplicative type. Since there must be a source, there
must be at least one additive component in a model, but there is no restriction on the
number of modifying components.

Given the quality of our data, as shown by the plot, we'll choose an absorbed
power law. To set it up, we define an instance of the PyXspec `Model` class
and assign it to a variable - `abs_pl_mod`.

### Setting up a model object

```{code-cell} python
abs_pl_mod = xs.Model("tbabs(powerlaw)")
```

### Ignoring bad channels

We are not quite ready to fit the data (and obtain a better $\chi^2$), because not
all of the 125 PHA bins should be included in the fitting:
- Some are below the lower discriminator of the instrument and therefore do not contain valid data.
- Some have imperfect background subtraction at the margins of the pass band.
- Some may not contain enough counts for $\chi^2$ to be strictly meaningful.

To find out which channels to discard (ignore in XSPEC terminology), consult
mission-specific documentation that will include information about discriminator
settings, background subtraction problems, and other issues.

For the mature missions in the HEASARC archives, this information may have already been
encoded in the headers of the spectral files as a list of bad channels. To remove the
bad channels from the spectrum that we
[previously read into `exo_me_spec`](#1-loading-a-spectrum-into-pyxspec), we can use:

```{code-cell} python
xs.AllData.ignore("bad")
```

```{note}
PyXspec doesn't allow us to ignore "bad" channels for individual spectra
but does it for all loaded spectra. AllData is a special object which allows us to
perform operations on all current spectra.
```

### Renormalizing the model to our data

The current statistic is $\chi^2$ and is huge for the initial, default values - mostly
because the power law normalization is two orders of magnitude too large. This
particular problem is easily fixed using the renorm method:

```{code-cell} python
xs.Fit.renorm()
```

To show off our renormalized, but not yet fit, model, we'll use PyXspec and
matplotlib together to produce a two-panel figure. The top panel will show
the spectral data and the current state of the model, and the bottom panel will
show the **signed** $\Delta\chi^2$ (each $\Delta\chi^2$ point has the sign of
the corresponding residual point).

Just like in [Section 2](#plotting-a-spectrum), we can use the PyXspec `Plot` manager
object to calculate the information necessary to plot our spectrum, model, and signed
$\Delta\chi^2$.

Passing two choices to the `Plot` object generates a plot with vertically stacked
'plot windows' (a maximum of six choices can be passed at once). The plot data
calculated for each plot option can be accessed by passing an index to the
`plotWindow=...` argument of `Plot`'s various methods.

Remember that XSPEC (and thus PyXspec) uses 'one-based indexing' (as opposed to
Python's zero-based indexing), so to retrieve the data relevant to the bottom panel, we
need to pass `plotWindow=2`, and `plotWindow=1` for the top panel.

We read out the spectral rates and errors, energy bin centers and half-widths, the
current rates of the model at each energy bin center, and the signed $\Delta\chi^2$
values calculated for the bottom panel - storing them in a dictionary. Additionally, the
axis labels that XSPEC _would_ have used are also stored in the same dictionary.

In this instance we're going to imitate the appearance of an XSPEC fitted spectrum
plot, so rather than plotting the model as a smooth line, we'll display it as a
'staircase'. This being as a visual reminder that the model is only evaluated at the
centers of the energy bins.

For that, we calculate the edges of each energy bin by subtracting the energy bin
half-widths from the energy bin centers and appending a final bin edge
representing the last energy bin center plus its half-width:

```{code-cell} python
xs.Plot("data chi")

rn_mod_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "signed_chisq": np.array(xs.Plot.y(plotWindow=2)),
    "x_label": xs.Plot.labels(plotWindow=1)[0],
    "y_label": xs.Plot.labels(plotWindow=1)[1],
    "chisq_label": xs.Plot.labels(plotWindow=2)[1],
}

rn_mod_plot_data["energy_step"] = np.append(
    rn_mod_plot_data["energy"] - rn_mod_plot_data["energy_delta"],
    rn_mod_plot_data["energy"][-1] + rn_mod_plot_data["energy_delta"][-1],
)
```

```{attention}
We get a warning that the fit is not current because no fit has been performed
yet (renormalizing doesn't count, the reason for which will become obvious when you
see the figure).
```

As a brief aside, we can examine the XSPEC-generated label for the y-axis of the
upcoming figure's lower panel, as we won't actually be using it in our
visualization.

This is for practical purposes, as it is too long for the small amount of space we
give to the lower panel - but as you can see, the meanings are equivalent, and you
will also notice that XSPEC produces LaTeX-formatted labels suitable for use
with matplotlib:

```{code-cell} python
rn_mod_plot_data["chisq_label"]
```

Finally, we can make our visualization:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
fig, ax_arr = plt.subplots(nrows=2, figsize=(7, 6), height_ratios=(3, 1.5), sharex=True)
# Shrink the vertical gap between the panels to zero
fig.subplots_adjust(hspace=0)

spec_ax = ax_arr[0]
spec_ax.minorticks_on()
spec_ax.tick_params(which="both", direction="in", top=True, right=True)

spec_ax.errorbar(
    rn_mod_plot_data["energy"],
    rn_mod_plot_data["rate"],
    xerr=rn_mod_plot_data["energy_delta"],
    yerr=rn_mod_plot_data["rate_err"],
    fmt="+",
    capsize=1.5,
    label="EXOSAT-ME data",
    color="navy",
)

spec_ax.stairs(
    rn_mod_plot_data["model"],
    rn_mod_plot_data["energy_step"],
    baseline=None,
    fill=False,
    color="firebrick",
    alpha=0.8,
    label="Renormalized model",
    linewidth=1.4,
)

spec_ax.set_xscale("log")
spec_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
spec_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_yscale("log")
spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_ylabel(rn_mod_plot_data["y_label"], fontsize=15)

spec_ax.legend(fontsize=14)

chi_ax = ax_arr[1]
chi_ax.minorticks_on()
chi_ax.tick_params(which="both", direction="in", top=True, right=True)

chi_ax.stairs(
    rn_mod_plot_data["signed_chisq"],
    rn_mod_plot_data["energy_step"],
    baseline=None,
    fill=False,
    color="navy",
    linewidth=1.4,
)

chi_ax.axhline(0, color="goldenrod", linestyle="dashed")

chi_ax.set_xlabel(rn_mod_plot_data["x_label"], fontsize=15)
chi_ax.set_ylabel(
    r"$\frac{\rm{Residual}}{|\rm{Residual}|} \: \times \: \Delta\chi^2$", fontsize=15
)

plt.show()
```


### Ignoring channels based on energy

Forty channels were rejected in [a previous section](#ignoring-bad-channels) because
they were flagged as **bad** - but do we need to ignore any more?

This figure shows the result of plotting the data and the model (in the upper window)
and the contributions to $\chi^2$ (in the lower window). We see that above about 15 keV
the signal-to-noise becomes small. We also see, when comparing with the earlier figure, which bad
channels were ignored. Although visual inspection is not the most rigorous method for
deciding which channels to ignore (more on this subject later), it's good enough for
now, and will at least prevent us from getting grossly misleading results from the
fitting. To ignore energies above 15 keV:

```{code-cell} python
exo_me_spec.ignore("15.0-**")
```

Note that ignore (and notice) interpret integers as channel numbers and real
numbers as energies. The double star is a special indicator which just means the
extreme value in the spectrum.

### Preparing to fit the model

We are now ready to fit the data. Fitting is initiated by the command xs.Fit.perform().

As the fit proceeds, the screen displays the status of the fit for each iteration
until either the fit converges to the minimum $\chi^2$, or the maximum number of
iterations is exceeded.

The current maximum number of iterations can be found like this:

```{code-cell} python
print(xs.Fit.nIterations)
```

Similarly, we can set a new maximum number of iterations:

```{code-cell} python
xs.Fit.nIterations = 50
```

### Performing the model fit

```{code-cell} python
xs.Fit.perform()
```

There is a fair amount of information here, so we will unpack it a bit at a time. One
line is written out after each fit iteration. The columns labeled 'Chi-Squared' and
'Parameters' are obvious. The other two provide additional information on fit
convergence. At each step in the fit a numerical derivative of the statistic with
respect to the parameters is calculated. We call the vector of these derivatives 'beta'.

At the best-fit the norm of beta should be zero, so we write out |beta| divided by the
number of parameters as a check. The actual default convergence criterion is when the
fit statistic does not change significantly between iterations, so it is possible for
the fit to end while |beta| is still significantly different from zero. The |beta|/N
column helps us spot this case. The Lvl column also indicates how the fit is
converging and should generally decrease. Note that for the first iteration only the
powerlaw norm is varied. While not necessary this simple model, for more complicated
models only varying the norms on the first iteration helps the fit proper get started
in a reasonable region of parameter space.

At the end of the fit PyXspec writes out the Variances and Principal Axes and
Covariance Matrix sections. These are both based on the second derivatives of the
statistic with respect to the parameters. Generally, the larger these second
derivatives, the better determined the parameter (think of the case of a parabola
in 1-D). The Covariance Matrix is the inverse of the matrix of second derivatives. The
Variances and Principal Axes section is based on an eigenvector decomposition of the
matrix of second derivatives and indicates which parameters are correlated. We can see
in this case that the first eigenvector depends almost entirely on the powerlaw normalization,
while the other two are combinations of the nH and powerlaw PhoIndex. This tells us
that the normalization is independent, but the other two parameters are correlated.

The next section shows the best-fit parameters and error estimates. The latter are
just the square roots of the diagonal elements of the covariance matrix so implicitly
assume that the parameter space is multidimensional Gaussian with all parameters
independent. We already know in this case that the parameters are not independent so
these error estimates should only be considered guidelines to help us determine the
true errors later.

The final section shows the statistic values at the end of the fit. PyXspec defines
a fit statistic, used to determine the best-fit parameters and errors, and test
statistic, used to decide whether this model and parameters provide a good fit to the
data. By default, both statistics are $\chi^2$. When the test statistic is $\chi^2$ we
can also calculate the null hypothesis probability. This is the probability of getting
a value of $\chi^2$ as large or larger than observed if the model is correct. If this
probability is small then the model is not a good fit. The null hypothesis probability
can be calculated analytically for $\chi^2$ but not for some other test statistics so
PyXspec provides another way of determining the meaning of the statistic value. The
xs.Fit.goodness() method performs simulations of the data based on the current model
and parameters and compares the statistic values calculated with that for the real
data. If the observed statistic is larger than the values for the simulated data this
implies that the real data do not come from the model. To see how this works we will
use the command for this case (where it is not necessary):

### Checking the goodness of fit

***TALK ABOUT SIGNIFICANT PERFORMANCE INCREASE WITH PARALLELISATION ON A 12 CORE MAC - FROM 3.8s TO 22ms*** - ***<span style="color:red">SUSPICIOUS?</span>***

```{code-cell} python
xs.Xset.parallel.goodness = NUM_CORES
```

```{code-cell} python
cur_lt_stat_perc = xs.Fit.goodness(1000)

cur_test_stat = xs.Fit.testStatistic

# The 'previousGoodnessSims' attribute returns a list of strings - they
#  must be converted to floats before we make a histogram.
goodness_dist = np.array(xs.Fit.previousGoodnessSims).astype(float)
goodness_dist[:20]
```

```{warning}
We retrieve the goodness-of-fit distribution in the same cell as the call to the
`goodness()` method to ensure that no other fit or goodness calculation has been
run in between _this_ goodness call and the _retrieval_.

This might happen if, for instance, you are running the notebook out of order - as the
goodness-of-fit distribution is stored in the global fit manager, rather than a unique
model object, it could be overridden.
```

Approximately 60% of the simulations give a statistic value less than that
observed, consistent with this being a good fit. We can plot a histogram of the
$\chi^2$ values from the simulations with the observed value shown by the vertical
dotted line.

It is entirely possible to retrieve the bin centers and probability density values
from PyXspec and use them with matplotlib to reconstruct the histogram that XSPEC
would make - just as we've been doing for other visualizations.

Taking that route for a histogram is a little awkward, however, so why don't we
instead directly use the goodness-of-fit value distribution to construct and plot
a histogram.

We actually already retrieved the goodness-of-fit distribution, in the same cell we ran
the `goodness()` method, so creating a histogram is straightforward:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(6, 5.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.hist(
    goodness_dist,
    bins=20,
    density=True,
    ec="teal",
    histtype="step",
    hatch="/",
    linewidth=3,
    label=r"Simulation $\chi^{2}$",
)

plt.axvline(
    cur_test_stat,
    linestyle="dashed",
    color="goldenrod",
    linewidth=2,
    label=r"Test statistic",
)

plt.xlabel(r"$\chi^2$", fontsize=15)
plt.ylabel("Probability Density", fontsize=15)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
```

### Examining fit residuals

So the statistic implies the fit is good, but it is still always a good idea to look
at the data and residuals to check for any systematic differences that may not be
caught by the test.

```{code-cell} python
xs.Plot("data resid")

fit_pl_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "residual": np.array(xs.Plot.y(plotWindow=2)),
    "residual_err": np.array(xs.Plot.yErr(plotWindow=2)),
}

fit_pl_plot_data["energy_step"] = np.append(
    fit_pl_plot_data["energy"] - fit_pl_plot_data["energy_delta"],
    fit_pl_plot_data["energy"][-1] + fit_pl_plot_data["energy_delta"][-1],
)
```

```{code-cell} python
plot_fit_residual_spec(
    fit_pl_plot_data, inst_name="EXOSAT-ME", mod_expr=abs_pl_mod.expression
)
```

## 4. Error analysis

Now that we think we have the correct model we need to determine how well the
parameters are determined. The screen output at the end of the fit shows the
best-fitting parameter values, as well as approximations to their errors. These
errors should be regarded as indications of the uncertainties in the parameters and
should not be quoted in publications. The true errors, i.e. the confidence ranges, are
obtained using the xs.Fit.error() command. We want to run error on all three parameters
which is an intrinsically parallel operation so we can use PyXspec's support for
multiple cores and run the error estimations in parallel:

### XSPEC's `error()` method

```{code-cell} python
xs.Xset.parallel.error = NUM_CORES
xs.Fit.error("1-3")
```

Here, the numbers 1, 2, 3 refer to the parameter numbers in the Model par column of
the output at the end of the fit. For the first parameter, the column of absorbing
hydrogen atoms, the 90% confidence range is 0.110 to 1.033. This corresponds to an
excursion in $\chi^2$ of 2.706. The reason these better errors are not given
automatically as part of the fit output is that they entail further fitting. When
the model is simple, this does not require much CPU, but for complicated models the
extra time can be considerable. The error for each parameter is determined allowing
the other two parameters to vary freely. If the parameters are uncorrelated this is
all the information we need to know. However, we have an indication from the
covariance matrix at the end of the fit that the column and photon index are
correlated.

### Running `steppar` to explore parameter correlations

To investigate this further we can use the xs.Fit.steppar() to run a
grid over these two parameters:

```{code-cell} python
---
tags: [hide-output]
jupyter:
  output_hidden: true
---
xs.Xset.parallel.steppar = NUM_CORES
xs.Fit.steppar("1 0.0 1.5 25 2 1.5 3.0 25")

# See the warning below
sleep(5)
```

```{warning}
We are aware of an unexpected behaviour in PyXspec v2.1.5 where `steppar()` outputs can
spill over into the next cell's output, particularly when using Jupyter's 'run all'
option. Pausing Python's execution for a few seconds after the `steppar` call is
a crude workaround.
```

### Examining `steppar` output as a contour plot

The result of our `steppar()` run can be understood more clearly by plotting
confidence contours.

XSPEC's (and thus PyXspec's) default $\Delta\chi^2$ contour levels are:
1. 2.30 [$1\sigma$; 68.3%]
2. 4.61 [90%]
3. 9.21 [99%]

with the stated confidence levels valid for a **two degree of freedom** (i.e.
parameter) contour plot.

The contour command we're about to use does expect the input to be a set
of $\Delta\chi^2$ values, so if we, for instance, want to specify which contours
are calculated as confidence levels in fraction/percentage form, we need to perform
a quick calculation.

To infer a confidence level (or inversely a p-value) from a $\chi^2$-distribution, we
have to inverse the cumulative distribution function (CDF). To make this a
little simpler, we can just use SciPy's implementation of the $\chi^2$-distribution,
and call the `ppf(...)` method (standing for "percent point function").

As we want to plot these contours and label them with their confidence levels, we
store our chosen percentiles in a dictionary, with keys ready to be used in the
legend of the figure we're about to construct.

The `df=2` argument specifies that we want to calculate the $\Delta\chi^2$ values
for a **two degree of freedom** distribution - this is because we're calculating a
confidence region for both parameters we investigated with `steppar()`:

```{code-cell} python
cont_conf_perc = {r"$1\sigma$": 0.6826, r"$2\sigma$": 0.9554, r"$3\sigma$": 0.9973}

cont_chisq = chi2.ppf(list(cont_conf_perc.values()), df=2).round(2)
cont_chisq
```

```{note}
If you wish to calculate $\Delta\chi^{2}$ values from percentage confidence levels
to pass to an XSPEC `error()` call, **remember to set `df=1`**. When you're using
`error()` to calculate parameter uncertainties, you are not calculating a _joint_
confidence region, no matter how many parameters you told `error()` to explore.
```

Now we use PyXspec's `Plot` manager to prepare all the information required to
create a contour plot using matplotlib. By default XSPEC will include a
probability density image as a backdrop to its contour plots, but as we don't
require that for our purposes, we turn it off.

The command we pass to `Plot` specifies the number of contours, and their levels;
as we don't manually specify a minimum fit statistic (the first argument), the
command begins with ",,".

```{code-cell} python
xs.Plot.addCommand("image off")

xs.Plot(f"contour ,,{len(cont_chisq)},{','.join(cont_chisq.astype(str))}")
xs.Plot.delCommand(1)

steppar_plot_data = {
    "nh": xs.Plot.x(),
    "powerlaw_ind": xs.Plot.y(),
    "contour_height": xs.Plot.z(),
    "contour_level": xs.Plot.contourLevels(),
    "x_label": xs.Plot.labels()[0],
    "y_label": xs.Plot.labels()[1],
}

# Store the current fit statistic value
cur_fit_stat = xs.Fit.statistic
```

Now that PyXspec has done all the hard work for us, we can use the information we
just retrieved to make a nice contour plot:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
#
plt.figure(figsize=(5.5, 5.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", right=True, top=True)

cont_obj = plt.contour(
    steppar_plot_data["nh"],
    steppar_plot_data["powerlaw_ind"],
    steppar_plot_data["contour_height"],
    steppar_plot_data["contour_level"],
    cmap="rainbow",
)
plt.clabel(cont_obj)

plt.xlabel(steppar_plot_data["x_label"], fontsize=15)
plt.ylabel(steppar_plot_data["y_label"], fontsize=15)

fit_point_obj = plt.plot(
    abs_pl_mod.TBabs.nH.values[0],
    abs_pl_mod.powerlaw.PhoIndex.values[0],
    "+",
    color="black",
    markersize=15,
    markeredgewidth=0.8,
)

plt.axvline(
    abs_pl_mod.TBabs.nH.values[0], color="black", linestyle="solid", linewidth=1.2
)
plt.axhline(
    abs_pl_mod.powerlaw.PhoIndex.values[0],
    color="black",
    linestyle="solid",
    linewidth=1.2,
)

cont_handles = cont_obj.legend_elements()[0]

leg_labels = ["Fit result"] + list(cont_conf_perc.keys())
leg_handles = fit_point_obj + cont_handles

plt.legend(leg_handles, leg_labels, fontsize=14, loc=4)

plt.tight_layout()
plt.show()
```

## 5. Flux calculation

What else can we do with the fit? One thing is to derive the flux of the model. The
data by themselves only give the instrument-dependent count rate. The model, on the
other hand, is an estimate of the true spectrum emitted. In PyXspec, the model is
defined in physical units independent of the instrument.

### Calling the `calcFlux()` method

The `calcFlux()` method of the model manager `AllModels` (`AllModels` is the way of
operating on all `Model` objects in the same way as `AllData` on all `Spectrum`
objects) integrates the current model over an energy range specified by the
user (2.0–10.0 keV in this case):

```{code-cell} python
xs.AllModels.calcFlux("2.0 10.0")
```

From that calculation we can see that the energy flux is
${\sim}2.2 \times 10^{-11} \: \rm{erg}\:\rm{cm}^{-2}\:s^{-1}$.

When calculating the flux in this manner (i.e. not using the method discussed
in [the next subsection](#using-the-cflux-model-component-to-calculate-flux-value-and-uncertainty)),
we can retrieve the current value from the `flux` attribute of the spectrum object:

```{code-cell} python
exo_me_spec.flux[0]
```

Note that `calcFlux()` will integrate only within the energy range of the current
response matrix. If the model flux outside this energy range is desired - in effect, an
extrapolation beyond the data - then the `setEnergies()` method should be used.

This method defines a set of energies on which the models will be calculated. The
resulting models are then remapped onto the response energies for convolution with
the response matrix.

For example, if we want to know the flux of our model in the ROSAT PSPC band
of 0.2–2.0 keV, we enter:

```{code-cell} python
xs.AllModels.setEnergies("extend", "low,0.2,100")
xs.AllModels.calcFlux("0.2 2.0")
```

The energy flux, at ${\sim}8.8\times10^{-12} \: \rm{erg}\:\rm{cm}^{-2}\:s^{-1}$ is
lower in this band, but the photon flux is higher.

Model energies can be reset to the response energies using `xs.AllModels.setEnergies("reset")`.

### Using the _cflux_ model component to calculate flux value and uncertainty

Calculating the flux is not usually enough, we want its uncertainty as well. The best
way to do this is to make use of the _cflux_ model. Suppose further that what we
really want is the **unabsorbed** flux (i.e. what we think the object is emitting
prior to the wider Universe getting in the way) then we redefine the model by:

```{code-cell} python
abs_pl_par_vals = (
    abs_pl_mod(1).values[0],
    abs_pl_mod(2).values[0],
    abs_pl_mod(3).values[0],
)

abs_pl_cflux_mod = xs.Model(
    "tbabs*cflux(powerlaw)",
    setPars=(
        abs_pl_par_vals[0],
        0.2,
        2.0,
        -10.3,
        abs_pl_par_vals[1],
        abs_pl_par_vals[2],
    ),
)
```

The _Emin_ and _Emax_ parameters are set to the energy range over which we want the
flux to be calculated. We also have to fix the normalization of the powerlaw because the
normalization of the model will now be determined by the _lg10Flux_ parameter.

```{code-cell} python
abs_pl_cflux_mod.powerlaw.norm.frozen = True
```

Now we run the model fit and calculate the uncertainty on parameter
**four** (_lg10Flux_):

```{code-cell} python
xs.Fit.perform()
xs.Fit.error("4")
```

This process tells us that the 90% confidence range (the default when
`error("<par ID>")` is called without further arguments) of the 0.2–2.0 keV unabsorbed
flux is ${\sim}3.5 — 8.3 \: \times 10^{-11} \: \rm{erg}\:\rm{cm}^{-2}\:s^{-1}$.

Usefully, we can also programmatically retrieve the flux value and
just-calculated confidence interval. As _cflux_ is just another component model,
we can access its parameters in the same way we would any other:

```{code-cell} python
cur_flux = Quantity(10 ** abs_pl_cflux_mod.cflux.lg10Flux.values[0], "erg cm^-2 s^-1")
cur_flux
```

```{code-cell} python
cur_flux_conf_inter = 10 ** np.array(abs_pl_cflux_mod.cflux.lg10Flux.error[:2])
cur_flux_conf_inter = Quantity(cur_flux_conf_inter, "erg cm^-2 s^-1")
cur_flux_conf_inter
```

## 6. Testing alternative spectral models

The absorbed power law fit, as we've remarked, is good, and the parameters are
constrained. However, unless the purpose of our investigation is merely to measure
a photon index, it's a good idea to check whether alternative models can fit the data
just as well.

We also should derive upper limits on components such as iron emission lines and
additional continua, which, although not evident in the data nor required for a good
fit, are nevertheless important to constrain - though we'll get to that
in [Section 7](#7-deriving-upper-limits-on-model-parameters).

### Absorbed blackbody model

First, let's try an absorbed blackbody:

```{code-cell} python
abs_bb_mod = xs.Model("tbabs*bb")
xs.Fit.perform()
```

Note that the fit process has displayed a warning about the first parameter and its
estimated **error is -1**.

Unsurprisingly, this is a bad sign! It indicates that the fit is unable to constrain
the parameter, and it should be considered indeterminate. We can usually interpret this
as meaning that the model is not appropriate.

One thing to check in this case is that the model component has any contribution
within the energy range being calculated.

The black body fit is obviously not a good one. Not only is $\chi^2$ large, but the
best-fitting N$_{\rm H}$ is indeterminate.

When diagnosing a seemingly poor model fit, it is often useful to take a look at
the residuals (like we [already did for the absorbed power law model](#examining-fit-residuals)). To
that end, we once again ask PyXspec to provide the necessary plotting data:

```{code-cell} python
xs.Plot("data resid")

fit_bb_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "residual": np.array(xs.Plot.y(plotWindow=2)),
    "residual_err": np.array(xs.Plot.yErr(plotWindow=2)),
}

fit_bb_plot_data["energy_step"] = np.append(
    fit_bb_plot_data["energy"] - fit_bb_plot_data["energy_delta"],
    fit_bb_plot_data["energy"][-1] + fit_bb_plot_data["energy_delta"][-1],
)
```

Now we plot the data, and inspection of the residuals provides another confirmation
of our belief that the absorbed blackbody model is not a good choice. The
pronounced wave-like shape is **indicative of a bad choice of overall continuum**:

```{code-cell} python
plot_fit_residual_spec(
    fit_bb_plot_data,
    inst_name="EXOSAT-ME",
    mod_expr=abs_bb_mod.expression,
    sp_color="darkgreen",
    res_color="darkgreen",
    mod_color="darkgrey",
)
```

### Absorbed thermal bremsstrahlung model

Let's try thermal bremsstrahlung next, following the same procedure we did for the
absorbed blackbody in [the last subsection](#absorbed-blackbody-model).

First, we define a model instance and run a fit:

```{code-cell} python
abs_br_mod = xs.Model("tbabs*brems")
xs.Fit.perform()
```

Now we extract the data necessary to plot the fitted spectrum and residuals:

```{code-cell} python
xs.Plot("data resid")

fit_br_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "residual": np.array(xs.Plot.y(plotWindow=2)),
    "residual_err": np.array(xs.Plot.yErr(plotWindow=2)),
}

fit_br_plot_data["energy_step"] = np.append(
    fit_br_plot_data["energy"] - fit_br_plot_data["energy_delta"],
    fit_br_plot_data["energy"][-1] + fit_br_plot_data["energy_delta"][-1],
)
```

Finally, we make a visualization:

```{code-cell} python
plot_fit_residual_spec(
    fit_br_plot_data, inst_name="EXOSAT-ME", mod_expr=abs_br_mod.expression
)
```

It is clear that the Bremsstrahlung model is a better fit than the blackbody - and is
as good as the power law - although it shares the low absorption column.

### Absorbed power law model [frozen nH]

With two models that appear to be good fits to the spectrum (absorbed power law and
absorbed Bremsstrahlung), it's time to scrutinize their parameters in more detail.

From the EXOSAT database on HEASARC, we know that the target in question,
**1E1048.1-5937**, is almost on the plane of the Galaxy. In fact, the database also
provides values for the Galactic N$_{\rm H}$ based on 21-cm radio observations.

One estimate (though admittedly not the one you will get from the current version
of `nhtool`) puts it at $4\times10^{22}$ cm$^{-2}$, which is higher than the 90%
confidence upper limit from the power-law fit.

Perhaps, then, the power-law fit is not so good after all. What
we can do is fix (freeze in XSPEC terminology) the value of N$_{\rm H}$ at the
Galactic value and refit the power law. Although we won't get a good fit, the shape
of the residuals might give us a clue to what is missing.

We follow a familiar procedure, though here we make sure to freeze the value of
the Hydrogen column density at the estimate we're using:

```{code-cell} python
abs_pl_frz_nh_mod = xs.Model("tbabs*powerlaw")

abs_pl_frz_nh_mod.TBabs.nH = 4.0
abs_pl_frz_nh_mod.TBabs.nH.frozen = True

xs.Fit.perform()
```

Then fetching the information necessary to plot a fitted spectrum and residuals:

```{code-cell} python
xs.Plot("data resid")

fit_pl_frz_nh_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "residual": np.array(xs.Plot.y(plotWindow=2)),
    "residual_err": np.array(xs.Plot.yErr(plotWindow=2)),
}

fit_pl_frz_nh_plot_data["energy_step"] = np.append(
    fit_pl_frz_nh_plot_data["energy"] - fit_pl_frz_nh_plot_data["energy_delta"],
    fit_pl_frz_nh_plot_data["energy"][-1] + fit_pl_frz_nh_plot_data["energy_delta"][-1],
)
```

Finally, making a visualization:

```{code-cell} python
plot_fit_residual_spec(
    fit_pl_frz_nh_plot_data,
    inst_name="EXOSAT-ME",
    mod_expr=abs_pl_frz_nh_mod.expression,
)
```

In this version of the absorbed power-law fit, there appears to be an observational
surplus of softer photons, perhaps indicating a second continuum component needs to
be modeled.

### Absorbed power law + blackbody model [frozen nH]

To investigate this possibility, we can combine what we have with another additive
model component; a _bbody_.

Note that we freeze the temperature parameter of the black body to 2 keV (the canonical
temperature for nuclear burning on the surface of a neutron star in a low-mass
X-ray binary) using an XSPEC trick that setting the delta for a parameter to zero
switches its freeze/thaw status.

We also set the normalization of the component to a small number to start the fit
off in a sensible place since we are looking for a small change to the model.

```{code-cell} python
abs_pl_frz_nh_par_vals = (
    abs_pl_frz_nh_mod(1).values[0],
    abs_pl_frz_nh_mod(2).values[0],
    abs_pl_frz_nh_mod(3).values[0],
)

abs_pl_bb_frz_nh_mod = xs.Model(
    "tbabs(powerlaw+bb)",
    setPars=(
        abs_pl_frz_nh_par_vals[0],
        abs_pl_frz_nh_par_vals[1],
        abs_pl_frz_nh_par_vals[2],
        "2.0,0.0",
        1.0e-5,
    ),
)

abs_pl_bb_frz_nh_mod.TBabs.nH.frozen = True
```

We run the fit of this new two-continua-component model:

```{code-cell} python
xs.Fit.perform()
```

The fit is better than the one with just a power law and the fixed Galactic
column, but it is still not good. Thawing the black body temperature and fitting
does of course improve the fit, but the power law index becomes even steeper.

Now we have two separate additive models contributing to the continuum fit, we might
want to examine their individual contributions to this odd model.

To do that, we're going to make yet another version of a fitted spectrum
visualization. This time though we're going to drop the residual panel, and
retrieve/plot both the overall model, and the curves of the individual model components.

For this to work, we have to tell PyXspec to calculate the plotting information for
individual additive model components as well as the usual total model:

```{code-cell} python
xs.Plot.add = True
```

Now we fetch much the same data as we have previously, but this time also use
the `Plot.addComp(<additive model ID>)` method to retrieve the plotting information
for the individual model components:

```{code-cell} python
xs.Plot("data")

fit_pl_bb_plot_data = {
    "energy": np.array(xs.Plot.x()),
    "energy_delta": np.array(xs.Plot.xErr()),
    "rate": np.array(xs.Plot.y()),
    "rate_err": np.array(xs.Plot.yErr()),
    "total_model": np.array(xs.Plot.model()),
    "powerlaw_model": np.array(xs.Plot.addComp(1)),
    "bbody_model": np.array(xs.Plot.addComp(2)),
}

fit_pl_bb_plot_data["energy_step"] = np.append(
    fit_pl_bb_plot_data["energy"] - fit_pl_bb_plot_data["energy_delta"],
    fit_pl_bb_plot_data["energy"][-1] + fit_pl_bb_plot_data["energy_delta"][-1],
)
```

Now we can visualize the spectrum and the two-additive-model fit:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(7, 4.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

cur_sp_color = "navy"
cur_tot_mod_color = "firebrick"
cur_pl_mod_color = "darkorchid"
cur_bb_mod_color = "peru"

cur_tot_mod_ls = "solid"
cur_pl_mod_ls = "dashed"
cur_bb_mod_ls = (0, (3, 1, 1, 1))

plt.errorbar(
    fit_pl_bb_plot_data["energy"],
    fit_pl_bb_plot_data["rate"],
    xerr=fit_pl_bb_plot_data["energy_delta"],
    yerr=fit_pl_bb_plot_data["rate_err"],
    fmt="+",
    capsize=1.5,
    label="EXOSAT-ME data",
    color=cur_sp_color,
)

# Total model
plt.stairs(
    fit_pl_bb_plot_data["total_model"],
    fit_pl_bb_plot_data["energy_step"],
    baseline=None,
    fill=False,
    color=cur_tot_mod_color,
    alpha=0.8,
    label="Total model",
    linewidth=1.4,
    linestyle=cur_tot_mod_ls,
)

# Power law model component
plt.stairs(
    fit_pl_bb_plot_data["powerlaw_model"],
    fit_pl_bb_plot_data["energy_step"],
    baseline=None,
    fill=False,
    color=cur_pl_mod_color,
    alpha=0.8,
    label="Power law component",
    linewidth=1.4,
    linestyle=cur_pl_mod_ls,
)

# Bremsstrahlung model component
plt.stairs(
    fit_pl_bb_plot_data["bbody_model"],
    fit_pl_bb_plot_data["energy_step"],
    baseline=None,
    fill=False,
    color=cur_bb_mod_color,
    alpha=0.8,
    label="Blackbody component",
    linewidth=1.4,
    linestyle=cur_bb_mod_ls,
)

plt.xscale("log")
plt.yscale("log")

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))


plt.xlabel("Energy [keV]", fontsize=15)

plt.ylabel(
    r"Spectrum [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
)

plt.legend(fontsize=14)

plt.tight_layout()
plt.show()
```

We see that the black body and the power law have changed places, in that the power
law provides the soft photons required by the high absorption, while the black body
provides the harder photons. We could continue to search for a plausible, well-fitting
model, but the data, with their limited signal-to-noise and energy resolution, probably
don't warrant it (the original investigators published only the power law fit).

## 7. Deriving upper limits on model parameters

There is one final useful thing to do with the data - derive an upper limit
to the presence of a fluorescent iron emission line. We return to our original model
and add a gaussian emission line of fixed energy and width then fit to get:

```{code-cell} python
abs_pl_gauss_em_mod = xs.Model(
    "tbabs*(powerlaw + gaussian)", setPars=(1.0, 1.0, 1.0, "6.4,0.0", "0.1,0.0", 1.0e-4)
)
xs.Fit.perform()
```

The energy and width have to be frozen because, in the absence of an obvious line in
the data, the fit would be completely unable to converge on meaningful
values. Besides, our aim is to see how bright a line at 6.4 keV can be and still
not ruin the fit. To do this, we fit first and then use the error command to derive
the maximum allowable iron line normalization. We then set the normalization at this
maximum value with and, finally, derive the equivalent width. That is:

```{code-cell} python
xs.Fit.error("6")
```

Note that the true minimum value of the gaussian normalization is less than zero, but
the error search stopped when the minimum value hit zero, the "hard" lower limit of
the parameter.

```{code-cell} python
abs_pl_gauss_em_mod.gaussian.norm = abs_pl_gauss_em_mod.gaussian.norm.error[1]
```

The `eqwidth()` method takes the component number as its argument:

```{code-cell} python
xs.AllModels.eqwidth("3")
```

## About this notebook

Author: Keith Arnaud, XSPEC Lead, Associate Research Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-04-08

+++

### Additional Resources

Support: [XSPEC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xspec)

[Original PyXspec Jupyter Notebooks Repository](https://github.com/HEASARC/PyXspec-Jupyter-notebooks)

[Full XSPEC walkthrough dataset](https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/walkthrough.tar.gz)

[XSPEC plot devices](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node110.html)

[XSPEC plot types](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node113.html)

### Acknowledgements

### References

[Seward F. D., Charles P. A., Smale A. P. (1986)](https://ui.adsabs.harvard.edu/abs/1986ApJ...305..814S/abstract) - _A 6 Second Periodic X-Ray Source in Carina_
