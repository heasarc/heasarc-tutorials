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
date: '2026-04-06'
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

- Load X-ray spectral data and response files using PyXspec.
- Use PyXspec and matplotlib to visualize spectra and model fits.
- Define and fit spectral models to X-ray data.
- Evaluate goodness-of-fit using statistical methods.
- Calculate parameter errors and confidence contours.

## Introduction

Our first example uses very old data which is much simpler than more modern
observations and so can be used to better illustrate the basics of XSPEC analysis.

The 6 s period X-ray pulsar 1E1048.1-5937 was observed by EXOSAT in June 1985 for 20 ks.

In this example, we'll conduct a general investigation of the spectrum from the
Medium Energy (ME) instrument, i.e. follow the same sort of steps as the original
investigators ([Seward F. D., Charles P. A., Smale A. P. 1986](https://ui.adsabs.harvard.edu/abs/1986ApJ...305..814S/abstract)).

**<span style="color:red">THIS IS TRUE BUT PERHAPS HIGHLIGHT THAT THE DATA AREN'T ACQUIRED THAT WAY IN THIS IMPLEMENTATION</span>**
The ME spectrum and corresponding
response matrix were obtained from the HEASARC and are available
from https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/walkthrough.tar.gz

### Inputs

- EXOSAT-ME spectrum file for 1E1048.1-5937 - s54405.pha
- Corresponding EXOSAT-ME response file - s54405.rsp

### Outputs

- Various diagnostic plots showing data, models, and residuals
- Best-fit model parameters with uncertainties
- Flux measurements and confidence ranges
- Upper limit on iron emission line equivalent width

### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the '{name: size}' server with NGB RAM/ N cores.

## Imports

```{code-cell} python
import contextlib
import os
from urllib.request import urlretrieve

import numpy as np
import xspec as xs
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
if os.path.exists("../../_data"):
    ROOT_DATA_DIR = "../../_data/PyXSPEC/EXOSAT"
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

***SPIEL AND ALSO EXPLAIN WHERE THE FILES WERE DOWNLOADED***

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
    "energy": xs.Plot.x(),
    "energy_delta": xs.Plot.xErr(),
    "rate": xs.Plot.y(),
    "rate_err": xs.Plot.yErr(),
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
must be least one additive component in a model, but there is no restriction on the
number of modifying components.

Given the quality of our data, as shown by the plot, we'll choose an absorbed power
law. To set it up define a `Model` object called `model_one`.

### Setting up a model object

```{code-cell} python
model_one = xs.Model("phabs(powerlaw)")
```

### Renormalizing the model to our data

The current statistic is $\chi^2$ and is huge for the initial, default values - mostly
because the power law normalization is two orders of magnitude too large. This is
easily fixed using the renorm method.

```{code-cell} python
xs.Fit.renorm()
```

### Ignoring bad channels

We are not quite ready to fit the data (and obtain a better $\chi^2$), because not
all of the 125 PHA bins should be included in the fitting: some are below the lower
discriminator of the instrument and therefore do not contain valid data; some have
imperfect background subtraction at the margins of the pass band; and some may not
contain enough counts for $\chi^2$ to be strictly meaningful. To find out which
channels to discard (ignore in XSPEC terminology), consult mission-specific
documentation that will include information about discriminator settings, background
subtraction problems and other issues. For the mature missions in the HEASARC
archives, this information already has been encoded in the headers of the spectral
files as a list of bad channels. To remove the bad channels from the spectrum that
we read into s:

```{code-cell} python
xs.AllData.ignore("bad")
```

Note that PyXspec doesn't allow us to ignore "bad" channels for individual spectra
but does it for all loaded spectra. AllData is a special object which allows
operations on all current spectra. Now plot again:

```{code-cell} python
xs.Plot("ldata chi")
energies = xs.Plot.x()
edeltas = xs.Plot.xErr()
rates = xs.Plot.y(1, 1)
errors = xs.Plot.yErr(1, 1)
foldedmodel = xs.Plot.model()
model_data = xs.Plot.model()

dataLabels = xs.Plot.labels(1)
chiLabels = xs.Plot.labels(2)
# note that for matplotlib step plots we need an x-axis array which includes
#  the start and end value for each
# bin and the y-axis has to be the same size with an extra value added equal
#  to the value of the last bin
nE = len(energies)
stepenergies = list()
for i in range(nE):
    stepenergies.append(energies[i] - edeltas[i])
stepenergies.append(energies[-1] + edeltas[-1])
foldedmodel.append(foldedmodel[-1])
chi = xs.Plot.y(1, 2)

chi_plot_data = xs.Plot.y(1, 2)

chi.append(chi[-1])
```

```{code-cell} python
STEPPED_MODEL = True
```

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
    energies,
    rates,
    xerr=edeltas,
    yerr=errors,
    fmt="+",
    capsize=1.5,
    label="EXOSAT-ME data",
    color="navy",
)

if not STEPPED_MODEL:
    spec_ax.plot(
        energies, model_data, color="firebrick", label="Fitted model", alpha=0.8
    )
else:
    spec_ax.step(
        stepenergies,
        foldedmodel,
        where="post",
        color="firebrick",
        label="Fitted model",
        alpha=0.8,
    )

spec_ax.set_xscale("log")
spec_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
spec_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_yscale("log")
spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_ylabel(dataLabels[1], fontsize=15)

spec_ax.legend(fontsize=14)

chi_ax = ax_arr[1]
chi_ax.minorticks_on()
chi_ax.tick_params(which="both", direction="in", top=True, right=True)

if not STEPPED_MODEL:
    chi_ax.plot(energies, chi_plot_data, color="navy")
else:
    chi_ax.step(stepenergies, chi, where="post", color="navy")

chi_ax.axhline(0, color="goldenrod", linestyle="dashed")

chi_ax.set_xlabel(chiLabels[0], fontsize=15)
chi_ax.set_ylabel(
    r"$\frac{\rm{Residual}}{|\rm{Residual}|} \: \times \: \Delta\chi^2$", fontsize=15
)

plt.show()
```

We get a warning that the fit is not current because no fit has been performed yet.

Giving two options for the Plot command generates a plot with vertically stacked
windows. Up to six options can be given to the Plot command at a time. Forty channels
were rejected because they were flagged as bad - but do we need to ignore any more?

This figure shows the result of plotting the data and the model (in the upper window)
and the contributions to $\chi^2$ (in the lower window). We see that above about 15 keV
the S/N becomes small. We also see, comparing with the earlier figure, which bad
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

We are now ready to fit the data. Fitting is initiated by the command xs.Fit.perform().

As the fit proceeds, the screen displays the status of the fit for each iteration
until either the fit converges to the minimum $\chi^2$, or the maximum number of
iterations is exceeded. The maximum number of iterations is xs.Fit.nIterations.

```{code-cell} python
print(xs.Fit.nIterations)
```

### Performing the model fit

```{code-cell} python
xs.Fit.perform()
```

There is a fair amount of information here so we will unpack it a bit at a time. One
line is written out after each fit iteration. The columns labeled Chi-Squared and
Parameters are obvious. The other two provide additional information on fit
convergence. At each step in the fit a numerical derivative of the statistic with
respect to the parameters is calculated. We call the vector of these derivatives beta.

At the best-fit the norm of beta should be zero so we write out |beta| divided by the
number of parameters as a check. The actual default convergence criterion is when the
fit statistic does not change significantly between iterations so it is possible for
the fit to end while |beta| is still significantly different from zero. The |beta|/N
column helps us spot this case. The Lvl column also indicates how the fit is
converging and should generally decrease. Note that on the first iteration only the
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
in this case that the first eigenvector depends almost entirely on the powerlaw norm
while the other two are combinations of the nH and powerlaw PhoIndex. This tells us
that the norm is independent but the other two parameters are correlated.

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

***TALK ABOUT SIGNIFICANT PERFORMANCE INCREASE WITH PARALLELISATION ON A 12 CORE MAC - FROM 3.8s TO 22ms***

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

```{code-cell} python
len(goodness_dist)
```

Approximately 60% of the simulations give a statistic value less than that
observed, consistent with this being a good fit. We can plot a histogram of the
$\chi^2$ values from the simulations with the observed value shown by the vertical
dotted line.

It is entirely possible to retrieve the bin centers and probability density values
from PyXspec, and use them with matplotlib to reconstruct the histogram that XSPEC
would make - just as we've been doing for other visualizations.

Taking that route for a histogram is a little awkward, however, so why don't we
instead directly use the goodness-of-fit value distribution to construct and plot
a histogram.

We actually already retrieved the goodness-of-fit distribution, in the same cell we ran
the `goodness()` method, so creating a histogram is simple:

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
energies = xs.Plot.x()
edeltas = xs.Plot.xErr()
rates = xs.Plot.y(1, 1)
errors = xs.Plot.yErr(1, 1)
foldedmodel = xs.Plot.model()
model_data = xs.Plot.model()

dataLabels = xs.Plot.labels(1)
residLabels = xs.Plot.labels(2)
# note that for matplotlib step plots we need an x-axis array which includes the
#  start and end value for each
# bin and the y-axis has to be the same size with an extra value added equal to
#  the value of the last bin
nE = len(energies)
stepenergies = list()
for i in range(nE):
    stepenergies.append(energies[i] - edeltas[i])
stepenergies.append(energies[-1] + edeltas[-1])
foldedmodel.append(foldedmodel[-1])
resid = xs.Plot.y(1, 2)
residerr = xs.Plot.yErr(1, 2)
```

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
    energies,
    rates,
    xerr=edeltas,
    yerr=errors,
    fmt="+",
    capsize=1.5,
    label="EXOSAT-ME data",
    color="navy",
)

if not STEPPED_MODEL:
    spec_ax.plot(
        energies, model_data, color="firebrick", label="Fitted model", alpha=0.8
    )
else:
    spec_ax.step(
        stepenergies,
        foldedmodel,
        where="post",
        color="firebrick",
        label="Fitted model",
        alpha=0.8,
    )

spec_ax.set_yscale("log")
spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_ylabel(dataLabels[1], fontsize=15)

spec_ax.legend(fontsize=14)

res_ax = ax_arr[1]
res_ax.minorticks_on()
res_ax.tick_params(which="both", direction="in", top=True, right=True)

res_ax.errorbar(
    energies, resid, xerr=edeltas, yerr=residerr, fmt="+", capsize=1.5, color="navy"
)
res_ax.axhline(0, color="goldenrod", linestyle="dashed")

res_ax.set_xlabel(residLabels[0], fontsize=15)
res_ax.set_ylabel(residLabels[1], fontsize=15)

res_ax.set_xscale("log")
res_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
res_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.show()
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
correlated. To investigate this further we can use the xs.Fit.steppar() to run a
grid over these two parameters:

```{code-cell} python
xs.Fit.steppar("1 0.0 1.5 25 2 1.5 3.0 25")
```

```{code-cell} python
chi2.ppf([0.6826, 0.9554, 0.9973], df=2).round(2)
```

***DEFAULT CHI-SQ LEVELS (FOR TWO-PARAMETER CONTOURS) ARE:***
- 2.30 [1sig]
- 4.61 [90%]
- 9.21 [99%]

***PER THE PLOT CONTOUR DOC PAGE, AND THE SRC***

The results can be understood more clearly by plotting confidence contours:

```{code-cell} python
xs.Plot.addCommand("image off")
xs.Plot("contour")
xs.Plot.delCommand(1)
labels = xs.Plot.labels()
x = xs.Plot.x()
y = xs.Plot.y()
z = xs.Plot.z()
levelvals = xs.Plot.contourLevels()
statval = xs.Fit.statistic
```

***<span style="color: red">TALK ABOUT WHY INDEX CUTS OFF AT ZERO</span>***

***<span style="color: red">ALTERNATIVELY LET STEPPAR GO LOWER THAN ZERO FOR PHO IND, SHOW THAT IT GOES STRAIGHT, AND EXPLAIN WHY THERE IS NO CONSTRAINING POWER ON NH WITH A NEGATIVE PHO INDEX</span>***

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(5.5, 5.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", right=True, top=True)

plt.contour(x, y, z, levelvals, cmap="rainbow")

plt.ylabel(labels[0], fontsize=15)
plt.xlabel(labels[1], fontsize=15)
plt.errorbar(
    model_one.phabs.nH.values[0], model_one.powerlaw.PhoIndex.values[0], fmt="+"
)
legendstring = (
    f"min={statval:{10}.{4}}, levels={levelvals[0]:{10}.{4}},"
    f"{levelvals[1]:{10}.{4}},{levelvals[2]:{10}.{4}}"
)
plt.legend([legendstring], loc="best", fontsize=14)

plt.tight_layout()
plt.show()
```

***<span style="color: red">I THINK THIS SHOWS 1sigma, 90%, and 3sigma</span>***
~~The contours shown are for one, two, and three sigma. The dot marks the best-fit position.~~

## 5. Flux calculation

What else can we do with the fit? One thing is to derive the flux of the model. The
data by themselves only give the instrument-dependent count rate. The model, on the
other hand, is an estimate of the true spectrum emitted. In PyXspec, the model is
defined in physical units independent of the instrument. xs.AllModels.calcFlux()
integrates the current model over the range specified by the user:

```{code-cell} python
xs.AllModels.calcFlux("2.0 10.0")
```

AllModels is the way of operating on all Model objects in the same way as AllData
on all Spectrum objects.

Here we have chosen the range of 2-10 keV and find that the energy flux is
$2.2 \times 10^{-11} \: \rm{erg}\:\rm{cm}^{-2}\:s^{-1}$. Note that calcFlux will integrate only within
the energy range of the current response matrix. If the model flux outside this
range is desired - in effect, an extrapolation beyond the data - then the method
setEnergies should be used. This method defines a set of energies on which the models
will be calculated. The resulting models are then remapped onto the response energies
for convolution with the response matrix. For example, if we want to know the flux of
our model in the ROSAT PSPC band of 0.2-2 keV, we enter:

```{code-cell} python
xs.AllModels.setEnergies("extend", "low,0.2,100")
xs.AllModels.calcFlux("0.2 2.0")
```

The energy flux, at $8.8\times10^{-12} \: \rm{erg}\:\rm{cm}^{-2}\:s^{-1}$ is lower in this band but the
photon flux is higher. The model energies can be reset to the response energies
using xs.AllModels.setEnergies("reset"). Calculating the flux is not usually
enough, we want its uncertainty as well. The best way to do this is to use the
cflux model. Suppose further that what we really want is the flux without the
absorption then we redefine the model by

```{code-cell} python
parVals = model_one(1).values[0], model_one(2).values[0], model_one(3).values[0]
model_one = xs.Model(
    "pha*cflux(pow)", setPars=(parVals[0], 0.2, 2.0, -10.3, parVals[1], parVals[2])
)
```

The Emin and Emax parameters are set to the energy range over which we want the flux
to be calculated. We also have to fix the norm of the powerlaw because the
normalization of the model will now be determined by the lg10Flux parameter.

```{code-cell} python
model_one.powerlaw.norm.frozen = True
```

```{code-cell} python
xs.Fit.perform()
xs.Fit.error("4")
```

for a 90% confidence range on the 0.2-2 keV unabsorbed flux of
$3.49\times10^{-11}$ - $8.33\times10^{-11} \: \rm{erg}\:\rm{cm}^{-2}\:s^{-1}$.

## 6. Testing alternative spectral models

The fit, as we've remarked, is good, and the parameters are constrained. But unless
the purpose of our investigation is merely to measure a photon index, it's a good idea
to check whether alternative models can fit the data just as well. We also should
derive upper limits to components such as iron emission lines and additional continua,
which, although not evident in the data nor required for a good fit, are nevertheless
important to constrain. First, let's try an absorbed black body:

```{code-cell} python
model_one = xs.Model("phabs*bb")
xs.Fit.perform()
```

Note that the fit has written out a warning about the first parameter and its
estimated error is written as -1. This indicates that the fit is unable to constrain
the parameter and it should be considered indeterminate. This usually indicates that
the model is not appropriate. One thing to check in this case is that the model
component has any contribution within the energy range being calculated.

The black body fit is obviously not a good one. Not only is $\chi^2$ large, but the
best-fitting N$_{\rm H}$ is indeterminate. Inspection of the residuals confirms
this: the pronounced wave-like shape is indicative of a bad choice of overall continuum.

```{code-cell} python
xs.Plot("data resid")
energies = xs.Plot.x()
edeltas = xs.Plot.xErr()
rates = xs.Plot.y(1, 1)
errors = xs.Plot.yErr(1, 1)
foldedmodel = xs.Plot.model()
dataLabels = xs.Plot.labels(1)
residLabels = xs.Plot.labels(2)
# note that for matplotlib step plots we need an x-axis array
#  which includes the start and end value for each
# bin and the y-axis has to be the same size with an extra value
#  added equal to the value of the last bin
nE = len(energies)
stepenergies = list()
for i in range(nE):
    stepenergies.append(energies[i] - edeltas[i])
stepenergies.append(energies[-1] + edeltas[-1])
foldedmodel.append(foldedmodel[-1])
resid = xs.Plot.y(1, 2)
residerr = xs.Plot.yErr(1, 2)
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.subplot(211)
plt.xscale("log")
plt.yscale("log")
plt.ylabel(dataLabels[1])
plt.title(dataLabels[2])
plt.errorbar(energies, rates, xerr=edeltas, yerr=errors, fmt=".")
plt.step(stepenergies, foldedmodel, where="post")
plt.subplot(212)
plt.xscale("log")
plt.xlabel(residLabels[0])
plt.ylabel(residLabels[1])
plt.errorbar(energies, resid, xerr=edeltas, yerr=residerr, fmt=".")
plt.hlines(0.0, stepenergies[0], stepenergies[-1], linestyles="dashed")
```

```{code-cell} python
fig, ax_arr = plt.subplots(nrows=2, figsize=(7, 6), height_ratios=(3, 1.5), sharex=True)
# Shrink the vertical gap between the panels to zero
fig.subplots_adjust(hspace=0)

spec_ax = ax_arr[0]
spec_ax.minorticks_on()
spec_ax.tick_params(which="both", direction="in", top=True, right=True)

spec_ax.errorbar(
    energies,
    rates,
    xerr=edeltas,
    yerr=errors,
    fmt="+",
    capsize=1.5,
    label="EXOSAT-ME data",
    color="navy",
)

if not STEPPED_MODEL:
    spec_ax.plot(
        energies, model_data, color="firebrick", label="Fitted model", alpha=0.8
    )
else:
    spec_ax.step(
        stepenergies,
        foldedmodel,
        where="post",
        color="firebrick",
        label="Fitted model",
        alpha=0.8,
    )

spec_ax.set_yscale("log")
spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_ylabel(labels[1], fontsize=15)

spec_ax.legend(fontsize=14)

res_ax = ax_arr[1]
res_ax.minorticks_on()
res_ax.tick_params(which="both", direction="in", top=True, right=True)

res_ax.errorbar(
    energies, resid, xerr=edeltas, yerr=residerr, fmt="+", capsize=1.5, color="navy"
)
res_ax.axhline(0, color="goldenrod", linestyle="dashed")

res_ax.set_xlabel(residLabels[0], fontsize=15)
res_ax.set_ylabel(residLabels[1], fontsize=15)

res_ax.set_xscale("log")
res_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
res_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.show()
```

Note the wave-like shape of the residuals which indicates how poor the fit is, i.e.
that the continuum is obviously not a black body.

Let's try thermal bremsstrahlung next:

```{code-cell} python
model_one = xs.Model("phabs*br")
xs.Fit.perform()
```

Bremsstrahlung is a better fit than the black body - and is as good as the power
law - although it shares the low absorption column. With two good fits, the power
law and the bremsstrahlung, it's time to scrutinize their parameters in more detail.

From the EXOSAT database on HEASARC, we know that the target in question,
1E1048.1-5937, is almost on the plane of the Galaxy. In fact, the database also
provides the value of the Galactic N$_{\rm H}$ based on 21-cm radio observations. At
$4\times10^{22}$ cm$^{-2}$, it is higher than the 90 percent-confidence upper limit
from the power-law fit. Perhaps, then, the power-law fit is not so good after all. What
we can do is fix (freeze in XSPEC terminology) the value of N$_{\rm H}$ at the
Galactic value and refit the power law. Although we won't get a good fit, the shape
of the residuals might give us a clue to what is missing.

```{code-cell} python
model_one = xs.Model("phabs*po")
model_one.phabs.nH = 4.0
model_one.phabs.nH.frozen = True
xs.Fit.perform()
```

```{code-cell} python
xs.Plot()
energies = xs.Plot.x()
edeltas = xs.Plot.xErr()
rates = xs.Plot.y(1, 1)
errors = xs.Plot.yErr(1, 1)
foldedmodel = xs.Plot.model()
dataLabels = xs.Plot.labels(1)
residLabels = xs.Plot.labels(2)
# note that for matplotlib step plots we need an x-axis array which
#  includes the start and end value for each
# bin and the y-axis has to be the same size with an extra value
#  added equal to the value of the last bin
nE = len(energies)
stepenergies = list()
for i in range(nE):
    stepenergies.append(energies[i] - edeltas[i])
stepenergies.append(energies[-1] + edeltas[-1])
foldedmodel.append(foldedmodel[-1])
resid = xs.Plot.y(1, 2)
residerr = xs.Plot.yErr(1, 2)
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.subplot(211)
plt.xscale("log")
plt.yscale("log")
plt.ylabel(dataLabels[1])
plt.title(dataLabels[2])
plt.errorbar(energies, rates, xerr=edeltas, yerr=errors, fmt=".")
plt.step(stepenergies, foldedmodel, where="post")
plt.subplot(212)
plt.xscale("log")
plt.xlabel(residLabels[0])
plt.ylabel(residLabels[1])
plt.errorbar(energies, resid, xerr=edeltas, yerr=residerr, fmt=".")
plt.hlines(0.0, stepenergies[0], stepenergies[-1], linestyles="dashed")
```

```{code-cell} python
fig, ax_arr = plt.subplots(nrows=2, figsize=(7, 6), height_ratios=(3, 1.5), sharex=True)
# Shrink the vertical gap between the panels to zero
fig.subplots_adjust(hspace=0)

spec_ax = ax_arr[0]
spec_ax.minorticks_on()
spec_ax.tick_params(which="both", direction="in", top=True, right=True)

spec_ax.errorbar(
    energies,
    rates,
    xerr=edeltas,
    yerr=errors,
    fmt="+",
    capsize=1.5,
    label="EXOSAT-ME data",
    color="navy",
)

if not STEPPED_MODEL:
    spec_ax.plot(
        energies, model_data, color="firebrick", label="Fitted model", alpha=0.8
    )
else:
    spec_ax.step(
        stepenergies,
        foldedmodel,
        where="post",
        color="firebrick",
        label="Fitted model",
        alpha=0.8,
    )

spec_ax.set_yscale("log")
spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

spec_ax.set_ylabel(labels[1], fontsize=15)

spec_ax.legend(fontsize=14)

res_ax = ax_arr[1]
res_ax.minorticks_on()
res_ax.tick_params(which="both", direction="in", top=True, right=True)

res_ax.errorbar(
    energies, resid, xerr=edeltas, yerr=residerr, fmt="+", capsize=1.5, color="navy"
)
res_ax.axhline(0, color="goldenrod", linestyle="dashed")

res_ax.set_xlabel(residLabels[0], fontsize=15)
res_ax.set_ylabel(residLabels[1], fontsize=15)

res_ax.set_xscale("log")
res_ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
res_ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.show()
```

There appears to be a surplus of softer photons, perhaps indicating a second continuum
component. To investigate this possibility, we can add a component to our model. Here,
we'll add a black body component. Note that we freeze the temperature parameter of
the black body to 2 keV (the canonical temperature for nuclear burning on the surface
of a neutron star in a low-mass X-ray binary) using an XSPEC trick that setting the
delta for a parameter to zero switches its freeze/thaw status. We also set the
normalization of the component to a small number to start the fit off in a sensible
place since we are looking for a small change to the model.

```{code-cell} python
parVals = model_one(1).values[0], model_one(2).values[0], model_one(3).values[0]
model_one = xs.Model(
    "phabs(pow+bb)", setPars=(parVals[0], parVals[1], parVals[2], "2.0,0.0", 1.0e-5)
)
model_one.phabs.nH.frozen = True
```

```{code-cell} python
xs.Fit.perform()
```

The fit is better than the one with just a power law and the fixed Galactic
column, but it is still not good. Thawing the black body temperature and fitting
does of course improve the fit, but the power law index becomes even steeper. Looking
at this odd model with the command

```{code-cell} python
xs.Plot("model")
energies = xs.Plot.x()
edeltas = xs.Plot.xErr()

modelvals = xs.Plot.model()
modelcomp1 = xs.Plot.addComp(1)
modelcomp2 = xs.Plot.addComp(2)

summ_mod_values = xs.Plot.model()
comp1_mod_values = xs.Plot.addComp(1)
comp2_mod_values = xs.Plot.addComp(2)

labels = xs.Plot.labels()
nE = len(energies)
stepenergies = list()
for i in range(nE):
    stepenergies.append(energies[i] - edeltas[i])
stepenergies.append(energies[-1] + edeltas[-1])
modelvals.append(modelvals[-1])
modelcomp1.append(modelcomp1[-1])
modelcomp2.append(modelcomp2[-1])
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(7, 4.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

tot_leg_lab = "Total model"
comp_one_leg_lab = "Power law component"
comp_two_leg_lab = "Blackbody component"

tot_color = "darkorchid"
comp_one_color = "tab:blue"
comp_two_color = "peru"

tot_ls = "solid"
comp_one_ls = "dashed"
comp_two_ls = (0, (3, 1, 1, 1))

if not STEPPED_MODEL:
    plt.plot(
        energies,
        summ_mod_values,
        color=tot_color,
        label=tot_leg_lab,
        alpha=0.8,
        linestyle=tot_ls,
    )
    plt.plot(
        energies,
        comp1_mod_values,
        color=comp_one_color,
        label=comp_one_leg_lab,
        alpha=0.8,
        linestyle=comp_one_ls,
    )
    plt.plot(
        energies,
        comp2_mod_values,
        color=comp_two_color,
        label=comp_two_leg_lab,
        alpha=0.8,
        linestyle=comp_two_ls,
    )
else:
    plt.step(
        stepenergies,
        modelvals,
        color=tot_color,
        label=tot_leg_lab,
        where="post",
        linestyle=tot_ls,
    )
    plt.step(
        stepenergies,
        modelcomp1,
        color=comp_one_color,
        where="post",
        label=comp_one_leg_lab,
        linestyle=comp_one_ls,
    )
    plt.step(
        stepenergies,
        modelcomp2,
        color=comp_two_color,
        where="post",
        label=comp_two_leg_lab,
        linestyle=comp_two_ls,
    )

plt.ylim(3.0e-6)
plt.xlim(1.25, 15)

plt.xscale("log")
plt.yscale("log")

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.xlabel(labels[0], fontsize=15)
plt.ylabel(labels[1], fontsize=15)

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

There is, however, one final, useful thing to do with the data: derive an upper limit
to the presence of a fluorescent iron emission line. We return to our original model
and add a gaussian emission line of fixed energy and width then fit to get:

```{code-cell} python
model_one = xs.Model(
    "phabs*(po + ga)", setPars=(1.0, 1.0, 1.0, "6.4,0.0", "0.1,0.0", 1.0e-4)
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
model_one.gaussian.norm = model_one.gaussian.norm.error[1]
```

```{code-cell} python
xs.AllModels.eqwidth("3")
```

The eqwidth method takes the component number as its argument.

## About this notebook

Author: Keith Arnaud, XSPEC Lead, Associate Research Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-04-06

+++

### Additional Resources

Support: [XSPEC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xspec)

[XSPEC plot devices](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node110.html)

### Acknowledgements

### References

[Seward F. D., Charles P. A., Smale A. P. (1986)](https://ui.adsabs.harvard.edu/abs/1986ApJ...305..814S/abstract) - _A 6 Second Periodic X-Ray Source in Carina_
