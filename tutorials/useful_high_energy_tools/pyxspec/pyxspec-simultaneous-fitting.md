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
date: '2026-11-12'
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

This demonstration focuses...

This PyXspec tutorial follows on from the previous '{doc}`PyXspec basics <pyxspec-basics>`' demonstration, which we strongly encourage you to go through first.


XSPEC has the very useful facility of allowing models to be fitted simultaneously to
more than one data file. It is even possible to group files together and to fit
different models simultaneously. Reasons for fitting in this manner include:

- The same target is observed at several epochs but, although the source stays constant, the response matrix has changed. When this happens, the data files cannot be added together; they have to be fitted separately. Fitting the data files simultaneously yields tighter constraints.
- The same target is observed with different instruments. All the instruments on Suzaku, for example, observe in the same direction simultaneously. As far as XSPEC is concerned, this is just like the previous case: two data files with two responses fitted simultaneously with the same model.
- Different targets are observed, but the user wants to fit the same model to each data file with some parameters shared and some allowed to vary separately. For example, if we have a series of spectra from a variable AGN, we might want to fit them simultaneously with a model that has the same, common photon index but separately vary the normalization and absorption.

Other scenarios are possible – the important thing is to recognize the flexibility of
XSPEC in this regard.

As an example we will look at a case of fitting the same model to two different data
files but where not all the parameters are identical.

Just as in the {doc}`first PyXspec tutorial <pyxspec-basics>`' we will make use of an old dataset, as
it provides a simpler illustration than more modern data.

In particular, we will use two observations of Centaurus X-3 (alternatively known as Cen X-3), a very
well-studied high-mass X-ray binary (HMXRB) that was the first X-ray pulsar ever discovered.

The observations in question were taken by Ginga (originally known as ASTRO-C), a Japanese Space Agency (JAXA) X-ray
telescope that was launched in 1987, using the 'Large Area Proportional Counter' (LAC) instrument. Ginga's
observations of Cen X-3 were taken in 1989.


Its flux level before eclipse
was much lower than the level after eclipse. Here, we'll use XSPEC to see whether
spectra from these two phases can be fitted with the same model, which differs only in
the amount of absorption. The spectra and corresponding response matrices were obtained
from the HEASARC and are available from
[the full XSPEC walkthrough dataset](https://heasarc.gsfc.nasa.gov/FTP/software/xspec/walkthrough.tar.gz).


### Inputs

- Ginga LAC pre-eclipse (low-state) spectrum file for Centaurus X-3 – `losum.pha`
- Ginga LAC post-eclipse (high-state) spectrum file for Centaurus X-3 – `hisum.pha`
- Corresponding Ginga LAC response file – `ginga_lac.rsp`

### Outputs

- Diagnostic plots showing the simultaneously fitted spectra, models, and residuals for both data groups
- Best-fit model parameters for the linked and partially decoupled models

### Runtime

As of 12th November 2026, this notebook takes ~30-seconds to run to completion on Fornax using the 'small' server with 8GB RAM/2 cores.

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
    sp_color: str | Sequence[str] = ("navy", "darkgreen", "darkorange", "teal"),
    mod_color: str | Sequence[str] = (
        "firebrick",
        "darkorchid",
        "peru",
        "crimson",
    ),
    res_color: str | Sequence[str] | None = None,
    mod_ls: str | Sequence[str | tuple] = (
        "solid",
        "dashed",
        (0, (3, 1, 1, 1)),
        "dotted",
    ),
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
    :param str | Sequence[str] mod_color: Matplotlib color (or sequence of colors)
        to use for the fitted model line of each dataset.
    :param str | Sequence[str] | None res_color: Matplotlib color (or sequence
        of colors) to use for the residual data points. Defaults to matching
        `sp_color` if None.
    :param str | Sequence[str | tuple] mod_ls: Matplotlib linestyle (or
        sequence of linestyles) to use for the fitted model line of each dataset.
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
    mod_colors = _broadcast_arg(mod_color, num_spec)
    res_colors = sp_colors if res_color is None else _broadcast_arg(res_color, num_spec)
    mod_lstyles = _broadcast_arg(mod_ls, num_spec)
    inst_names = _broadcast_arg(inst_name, num_spec)
    mod_exprs = _broadcast_arg(mod_expr, num_spec)

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
            fmt="+",
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
                linewidth=1.4,
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
                linewidth=1.4,
                zorder=3,
            )

        res_ax.errorbar(
            p_data["energy"],
            p_data["residual"],
            xerr=p_data["energy_delta"],
            yerr=p_data["residual_err"],
            fmt="+",
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
```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# -------------- Set paths and create directories --------------
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = "../../../_data/PyXSPEC/Ginga"
else:
    ROOT_DATA_DIR = "PyXSPEC/Ginga"

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
indexing of the data files.

The files in each group share the same model but not
necessarily the same parameter values, which may be shared as common to all the groups
or varied separately from group to group.

Although each group may contain more than one
file, there is only one file in each of the two groups in this example.

```{important}
Unlike in the first '{doc}`PyXspec basics <pyxspec-basics>`' tutorial, where we defined
a PyXspec `Spectrum` object directly to load our spectrum, here we are using `xs.AllData()`
to load our spectra all at once.

This is a less 'Pythonic' interface, but mirrors the syntax used by command-line XSPEC.

Unlike loading using `Spectrum`, `xs.AllData()` **does not** return a `Spectrum`
object; instead it returns Python's null object, `None`.
```

Groups are
specified with the data command, with the group number preceding the file number, like
this:

The spectral files we need for this demonstration were downloaded in
the [Global Setup: Configuration](#configuration) section. As they were not downloaded to the
same directory as the notebook, so we will briefly change our working directory (using
the `chdir` context manager) as we load them.

Of course, we could pass full spectrum paths rather than changing
directories, but the 'RESPFILE' entry in each spectrum's header is a path relative
to the location of the spectrum file. As such, if we didn't change directories, PyXspec
would have been unable to find and automatically load the response file.

```{code-cell} python
with contextlib.chdir(ROOT_DATA_DIR):
    xs.AllData("1:1 losum 2:2 hisum")
```

```{tip}
The call to `AllData()` above should also display a summary of the spectra currently in
use (unless you have PyXspec's 'chatter' set to $<10$ using `xs.Xset.chatter=`).

You can also run `xs.AllData.show()` to see that summary at any time (again, if chatter
has been set high enough).
```

Here, the first group makes up the file `losum.pha`, which contains the spectrum of all
the low, pre-eclipse emission.

The second group makes up the second file, `hisum.pha`,
which contains all the high, post-eclipse emission.

Note that file number is "absolute"
in the sense that it is independent of group number.

Thus, if there were three files in
each of the two groups (`lo1.pha`, `lo2.pha`, `lo3.pha`, `hi1.pha`, `hi2.pha`, and
`hi3.pha`, say), rather than one, the six files would be specified as
`"1:1 lo1 1:2 lo2 1:3 lo3 2:4 hi1 2:5 hi2 2:6 hi3"`.

The ignore method works on file
number, and does not take group number into account. So, to ignore channels 1–3 and
37–48 of both files:

<span style="color:red">**Suggested context & potential code cells (Inspecting spectra before fitting)**:</span>
1. In `pyxspec-basics.md`, we strongly encouraged plotting the raw spectra *before* ignoring channels or fitting a model. Consider adding a code cell and figure here showing the two raw spectra (`losum` and `hisum`) across all 48 channels so the reader can visually see why channels 1–3 and 37–48 are ignored and how much lower the pre-eclipse count rate is compared to post-eclipse. (In the CLI XSPEC manual output, XSPEC prints `Net count rate (cts/s) for Spectrum:1  1.401e+02 +/- 3.549e-01` and `Net count rate (cts/s) for Spectrum:2  1.371e+03 +/- 3.123e+00`, with `Exposure Time: 1 sec` in both PHA headers.)
2. Explain *why* `xs.AllData.ignore("1-3 37-48")` applies to both spectra: when no spectrum index range (`<specRange>:`) is prefixed, `AllData.ignore` applies the channel range to every loaded spectrum. In the CLI XSPEC manual walkthrough (`node65.html`), this command is written with the explicit file number range as `ignore 1-2:1-3 37-48` to directly illustrate the preceding sentence (*"The ignore method works on file number, and does not take group number into account"*). It would also be helpful to remind the reader that integer values specify channel numbers whereas floating-point numbers specify energies (e.g. `1.0-3.0`).

### Examining the 'raw' spectra

<span style="color:red">**Explain how we're configuring plotting behaviour**</span>
```{code-cell} python
xs.Plot.xAxis = "keV"
xs.Plot("data")
```

```{code-cell} python
raw_spec_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(grp_id, 1)),
        "energy_delta": np.array(xs.Plot.xErr(grp_id, 1)),
        "rate": np.array(xs.Plot.y(grp_id, 1)),
        "rate_err": np.array(xs.Plot.yErr(grp_id, 1)),
        "x_label": xs.Plot.labels(1)[0],
        "y_label": xs.Plot.labels(1)[1],
        "title": xs.Plot.labels(1)[2],
    }
    raw_spec_plot_data.append(grp_data)
```

```{note}
We set XSPEC's plot device to "/null" in the [Global Setup: Configuration](#configuration) section, which
means that it won't output figures - instead we construct our own figures with matplotlib, extracting data
from PyXspec.
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# SUGGESTION: PLOT THE TWO RAW SPECTRA PRIOR TO IGNORING CHANNELS OR FITTING
plt.figure(figsize=(7.5, 5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

raw_labels = ("Ginga LAC (pre-eclipse)", "Ginga LAC (post-eclipse)")
raw_colors = ("navy", "darkgreen", "darkorange", "teal")

for idx, p_data in enumerate(raw_spec_plot_data):
    sp_label = raw_labels[idx] if idx < len(raw_labels) else f"Group {idx + 1} data"
    plt.errorbar(
        p_data["energy"],
        p_data["rate"],
        xerr=p_data["energy_delta"],
        yerr=p_data["rate_err"],
        fmt="+",
        capsize=1.5,
        label=sp_label,
        color=raw_colors[idx % len(raw_colors)],
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

### Excluding parts of the spectra from our fit

```{code-cell} python
xs.AllData.ignore("1-3 37-48")
```

```{code-cell} python
# SUGGESTION: REPLICATE THE EXPLICIT FILE-NUMBER RANGE SYNTAX FROM THE CLI XSPEC
#  MANUAL (node65.html)
xs.AllData.ignore("1-2:1-3 37-48")
```


<span style="color:purple">For the mature missions in the HEASARC archives, this information may have already been
encoded in the headers of the spectral files as a list of bad channels. To remove the
bad channels from the spectrum that we
[previously read in](#loading-multiple-spectra-into-separate-data-groups), we can use:</span>

```{code-cell} python
xs.AllData.ignore("bad")
```

### Choosing and setting up a model

The model we'll use at first to fit the two files is an absorbed power law with a
high-energy cut-off:

- **tbabs** -
- **highecut** - This multiplicative model has a constant value of **1** up to the 'cutoff energy'. Beyond that the model value begins to decrease with increasing energy, with the rate of decrease controlled by an 'e-folding energy' parameter.
- **pow** -

<span style="color:purple">We do not assign the return of the model definition to a variable - even though a single PyXspec `Model` instance is returned, PyXspec has _actually_ set up two separate instances of the model, one per data group. We will access them through PyXspec's `AllModels()` manager later on in this tutorial.</span>

```{code-cell} python
xs.Model("tbabs * highecut(pow)")
```

<br>

<span style="color:red">**Text from the CLI XSPEC manual (`node65.html`) omitted from the original PyXspec notebook**:</span>
<span style="color:red">After defining the model, we will be prompted for two sets of parameter values, one for the first group of data files (losum.pha), the other for the second group (hisum.pha). Here, we'll enter the absorption column of the first group as $10^{24}\ \mathrm{cm}^{-2}$ and enter the default values for all the other parameters in the first group. Now, when it comes to the second group of parameters, we enter a column of $10^{22}\ \mathrm{cm}^{-2}$ and then enter defaults for the other parameters. The rule being applied here is as follows: to tie parameters in the second group to their equivalents in the first group, take the default when entering the second-group parameters; to allow parameters in the second group to vary independently of their equivalents in the first group, enter different values explicitly.</span>

<span style="color:red">**Text from the CLI XSPEC manual (`node65.html`) following the model initialization**:</span>
<span style="color:red">Notice how the summary of the model, displayed immediately above, is different now that we have two groups, as opposed to one (as in all the previous examples). We can see that of the 10 model parameters, 6 are free (i.e., 4 of the second group parameters are tied to their equivalents in the first group). Fitting this model results in a huge $\chi^2$ (not shown here), because our assumption that only a change in absorption can account for the spectral variation before and after eclipse is clearly wrong.</span>

Notice how the summary of the model, displayed immediately above, is different now that
we have two groups, as opposed to one.

We can see that of the 10 model parameters, 5
for each data group and, by default, the parameters for the second data group are
linked to those for the first.

Let us suppose that we have a theory that the difference
between the two spectra is a change in the absorption.

We want to remove the link
between parameters 1 and 6 and the fit.

When there are two datagroups there are two
separate model objects created so to change the sixth parameter, which is the first
parameter of the model for the second datagroup we do:



```{tip}
Setting up the model above should also display an overview of its parameters currently in
use (unless you have PyXspec's 'chatter' set too low, see [the previous note on viewing a summary of loaded spectra](#loading-multiple-spectra-into-separate-data-groups)).

You can also run `xs.AllModels.show()` to see that summary at any time.
```


## 2. Fitting a shared model

```{code-cell} python
xs.Fit.nIterations = 50
```


<span style="color:red">**Suggested context (Retrieving data group models, setting initial $N_{\mathrm{H}}$, & untying parameters)**:</span>
1. Fix the slight typo in the original PyXspec text above ("5 for each data group" $\rightarrow$ "there are 5 for each data group", and "between parameters 1 and 6 and the fit" $\rightarrow$ "between parameters 1 and 6 and then fit"), or adopt the cleaner wording from the CLI XSPEC manual quoted above.
2. Consider using more descriptive Python variable names than `m1` and `m2` (e.g. `lo_grp_mod` and `hi_grp_mod`, or `grp1_mod` and `grp2_mod`).
3. Show that `m2(1).untie()` can also be written using component/parameter attribute names (`m2.phabs.nH.untie()`), which is much less error-prone for complex multi-component models than counting parameter indices across groups.
4. Notice that in the CLI XSPEC manual (`node65.html`), Group 1's `phabs.nH` was explicitly initialized to `100.0` ($10^{24}\ \mathrm{cm}^{-2}$) while Group 2's `phabs.nH` was set to `1.0` ($10^{22}\ \mathrm{cm}^{-2}$), whereas the PyXspec notebook left both at `1.0` and only called `m2(1).untie()`. Replicating the CLI initialization (`m1.phabs.nH = 100.0`) before fitting is shown in the suggested cell below.

```{code-cell} python
m1 = xs.AllModels(1)
m2 = xs.AllModels(2)
m2(1).untie()
xs.Fit.perform()
```

```{code-cell} python
# SUGGESTION: REPLICATE THE INITIAL PARAMETER VALUES FROM THE CLI XSPEC MANUAL
#  (node65.html) WHERE GROUP 1 nH IS SET TO 100 (10^24 cm^-2) AND GROUP 2 nH IS
#  SET TO 1 (10^22 cm^-2)
m1 = xs.AllModels(1)
m2 = xs.AllModels(2)
m1.TBabs.nH = 100.0
m2.TBabs.nH = 1.0
xs.Fit.perform()
```

<br>

<span style="color:red">**Suggested context (Fit output & extracting multi-group plot data)**:</span>
1. Add a short discussion of the fit output above: point out the very large $\chi^2$ (~16386 for 65 degrees of freedom) and the fact that XSPEC reports the individual $\chi^2$ contributions from each data group (`Spectrum 1` and `Spectrum 2`).
2. Explain how `xs.Plot.y(plotGroup, plotWindow)`, `xs.Plot.yErr(plotGroup, plotWindow)`, and `xs.Plot.model(plotGroup)` work when multiple data groups are loaded: the first integer argument selects the data/plot group (`1` for `losum`, `2` for `hisum`), while the second selects the panel (`1` for `"data"`, `2` for `"resid"`).



```{code-cell} python
xs.Plot.xAxis = "keV"
xs.Plot("data resid")

untied_nh_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(grp_id, 1)),
        "energy_delta": np.array(xs.Plot.xErr(grp_id, 1)),
        "rate": np.array(xs.Plot.y(grp_id, 1)),
        "rate_err": np.array(xs.Plot.yErr(grp_id, 1)),
        "model": np.array(xs.Plot.model(grp_id, 1)),
        "residual": np.array(xs.Plot.y(grp_id, 2)),
        "residual_err": np.array(xs.Plot.yErr(grp_id, 2)),
    }
    grp_data["energy_step"] = np.append(
        grp_data["energy"] - grp_data["energy_delta"],
        grp_data["energy"][-1] + grp_data["energy_delta"][-1],
    )
    untied_nh_plot_data.append(grp_data)
```

```{code-cell} python
plot_fit_residual_spec(
    untied_nh_plot_data,
    inst_name=["Ginga LAC (pre-eclipse)", "Ginga LAC (post-eclipse)"],
    mod_expr=["Group 1: untied nH", "Group 2: untied nH"],
    y_scale="linear",
    fig_size=(9.5, 7.5),
)
```

## 3. Decoupling the power-law normalization

Clearly our assumption that only a change in absorption can account for the spectral
variation before and after eclipse is clearly wrong. Perhaps scattering also plays a
role in reducing the flux before eclipse. This could be modeled (simply at first) by
allowing the normalization of the power law to be smaller before eclipse than after
eclipse. To decouple tied parameters, we can change the parameter value in the second
group to a value – any value – different from that in the first group (changing the
value in the first group has the effect of changing both without decoupling).

<span style="color:red">**Suggested context (Two ways to untie parameters)**:</span>
1. Note the duplicate word in the original text ("Clearly our assumption ... is clearly wrong").
2. Highlight that this demonstrates a *second* method for untying a parameter in PyXspec: assigning a new numerical value directly to the secondary group's parameter (`m2.powerlaw.norm = 1.0`) automatically breaks the link `= p5`, just like calling `m2.powerlaw.norm.untie()`. Consider adding a tip admonition comparing `.untie()` vs direct value assignment (and how to *re-tie* a parameter via `m2.powerlaw.norm.link = "5"` or `m2(10).link = "p5"`).

```{code-cell} python
m2.powerlaw.norm = 1.0
xs.Fit.perform()
```

<br>

```{code-cell} python
xs.Plot("data resid")

untied_nh_norm_plot_data = []
for grp_id in range(1, xs.AllData.nGroups + 1):
    grp_data = {
        "energy": np.array(xs.Plot.x(grp_id, 1)),
        "energy_delta": np.array(xs.Plot.xErr(grp_id, 1)),
        "rate": np.array(xs.Plot.y(grp_id, 1)),
        "rate_err": np.array(xs.Plot.yErr(grp_id, 1)),
        "model": np.array(xs.Plot.model(grp_id, 1)),
        "residual": np.array(xs.Plot.y(grp_id, 2)),
        "residual_err": np.array(xs.Plot.yErr(grp_id, 2)),
    }
    grp_data["energy_step"] = np.append(
        grp_data["energy"] - grp_data["energy_delta"],
        grp_data["energy"][-1] + grp_data["energy_delta"][-1],
    )
    untied_nh_norm_plot_data.append(grp_data)
```

```{code-cell} python
plot_fit_residual_spec(
    untied_nh_norm_plot_data,
    inst_name=["Ginga LAC (pre-eclipse)", "Ginga LAC (post-eclipse)"],
    mod_expr=["Group 1: untied nH & norm", "Group 2: untied nH & norm"],
)
```

This decoupling improves the fit, but it still clearly not good. Indeed, this simple
attempt to account for the spectral variability in terms of "blanket" cold absorption
and scattering does not work. More sophisticated models, involving additional
components and partial absorption, should be tried.

<span style="color:red">**Text from the CLI XSPEC manual (`node65.html`) concluding the second fit**:</span>
<span style="color:red">After fitting, this decoupling reduces $\chi^2$ by a factor of six to 15,308, but this is still too high. Indeed, this simple attempt to account for the spectral variability in terms of "blanket" cold absorption and scattering does not work. More sophisticated models, involving additional components and partial absorption, should be tried.</span>

<span style="color:red">**Suggested context (Quantitative discussion of the second fit from `node65.html`)**:</span>
1. Replace *"This decoupling improves the fit, but it still clearly not good"* (which is missing "is") with the more informative sentence from `node65.html` quoted above (*"After fitting, this decoupling reduces $\chi^2$ by a factor of six to 15,308, but this is still too high."*). Note that in `node65.html`, the first fit had a $\chi^2$ around ~90,000+ (when initialized with $N_{\mathrm{H},1}=100$), and the second fit reaches $\chi^2 = 15,308.95$ with 59 d.o.f. (`Spectrum 1`: $13,992.93$ using 33 bins; `Spectrum 2`: $1,316.02$ using 33 bins).
2. Point out the fitted parameter differences between Group 1 (`losum`, $N_{\mathrm{H}} \approx 20.6 \times 10^{22}\ \mathrm{cm}^{-2}$, $\mathrm{norm} \approx 0.0596$) and Group 2 (`hisum`, $N_{\mathrm{H}} \approx 1.31 \times 10^{22}\ \mathrm{cm}^{-2}$, $\mathrm{norm} \approx 0.314$), and note that Group 1 (the pre-eclipse spectrum) contributes the vast majority of the remaining $\chi^2$ (~13,993 vs ~1,316).

<span style="color:red">**Suggested extension (Completing the scientific story & plotting additive components)**:</span>
The original notebook ends abruptly here with a poor fit ($\chi^2 \approx 2483$ for 64 d.o.f.) and prominent residuals around ~6.4–7 keV (Fe K$\alpha$ emission line / iron edge) and at soft energies!
Consider adding a follow-up section that actually fits a more realistic model to Cen X-3 (e.g. adding a `gaussian` iron line around 6.4 keV and/or a partial-covering absorber `pcfabs`, or separate soft scattered vs direct absorbed power-law components), and then demonstrating multi-component plotting (`xs.Plot.add = True` and `xs.Plot.addComp(...)`, as in Section 6 of `pyxspec-basics.md`) across the two data groups so the reader finishes the tutorial with a good fit.


## 4. When might you want to simultaneously fit?


## About this notebook

Author: Keith Arnaud, XSPEC Lead, Associate Research Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-11-12

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

<span style="color:red">**Suggested addition (References)**: Add the Ginga LAC Cen X-3 reference (e.g. [Ebisawa et al. 1992](https://ui.adsabs.harvard.edu/abs/1992PASJ...44..225E/abstract) or [Nagase et al. 1992](https://ui.adsabs.harvard.edu/abs/1992ApJ...396..147N/abstract)) once confirmed.</span>
