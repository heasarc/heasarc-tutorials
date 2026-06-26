---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: NICER Team
date: '2026-06-26'
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
title: Combining NICER observations and generating spectra
---

# Combining NICER observations and generating spectra

Learning Goals

By the end of this tutorial, you will be able to:
- Understand the function of the `nicerl3-spect` pipeline.
- Generate a spectrum, background, and associated responses in the NICER-recommended way.
- Choose between different background models for NICER data.
- Troubleshoot common pipeline errors.
- Generalize the analysis across multiple targets using multiprocessing.
- Load and visualize the resulting spectral products using PyXspec and Matplotlib.

Introduction

Spectroscopy is a fundamental pillar of most X-ray data analysis projects. An X-ray spectrum provides information about the X-ray target that usually cannot be found by any other means. Thus, extracting a spectrum is a fundamental task that almost all NICER users will need to do.

This thread describes how to extract all NICER spectral products using a single pipeline task called `nicerl3-spect`. This task generates all products needed for spectral analysis: spectrum, background estimate, responses (ARF and RMF), and some other script files to get you started with analysis.

The `nicerl3-spect` task became available with HEASoft 6.31, which was released in November, 2022. The task provides a straightforward and streamlined way to extract spectral products with NICER-recommended methods. While `nicerl3-spect` is targeted at the "new user" of NICER data, it is equally usable by advanced users. Many of the complicated and/or poorly documented steps are now done automatically.

Inputs

- NICER Observation Data (Dynamically downloaded via Astroquery)
- Geomagnetic data required by `nicerl2`
Both are downloaded directly from the HEASARC archive within this notebook.

Outputs

- Source spectrum file (`*sr.pha`)
- Background spectrum file (`*bg.pha`)
- ARF response file (`*arf`)
- RMF response file (`*rmf`)
- Background response file (`*bkgrmf`)
- XSPEC "load" file (`*_load.xcm`)

Runtime

As of 25th June 2026, this notebook takes ~<span style="color:red">***??***</span>-minutes to run to completion on Fornax using the 'medium' server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import contextlib
import multiprocessing as mp
import os
import re
from random import randint
from shutil import rmtree
from typing import List, Optional, Tuple

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.heasarc import Heasarc
from IPython.display import display
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

def process_nicer_xti(obs_dir: str, cur_obs_id: str, out_dir: str):
    """
    A wrapper function for nicerl2, designed to be passed to a
    multiprocessing pool to process multiple observations in parallel.
    """

    # Ensure that the observation directory passed by the user is absolute before
    #  we start changing directories.
    # Once we use the chdir context to switch directories during processing, we'll
    #  retrieve a relative path to limit the number of characters in the string
    #  passed to xapipeline (long paths can sometimes cause problems)
    obs_dir = os.path.abspath(obs_dir)

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, f"nicerl2_{randint(0, int(1e8))}")
    os.makedirs(temp_work_dir)

    # Set up the output file names
    evt_out = os.path.basename(EVT_PATH_TEMP).format(oi=cur_obs_id, m="mpuall")

    uf_evt_out = os.path.basename(UFEVT_PATH_TEMP).format(oi=cur_obs_id, m="mpuall")

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        # The processing/preparation stage of any X-ray telescope's data is the most
        #  likely to go wrong, and we use a Python try-except as an automated way to
        #  collect ObsIDs that had an issue during processing.
        try:
            out = hsp.nicerl2(
                indir=os.path.relpath(obs_dir),
                ufafile=uf_evt_out,
                clfile=evt_out,
                geomag_path=GEOMAG_PATH,
                clobber=True,
                noprompt=True,
                chatter=TASK_CHATTER,
            )

            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    prod_lookup = {}

    # Moves files from the temporary output directory into the
    #  final output directory
    if (
        os.path.exists(temp_work_dir)
        and len(os.listdir(temp_work_dir)) != 0
        and task_success
    ):

        final_evt = os.path.join(out_dir, evt_out)
        os.rename(os.path.join(temp_work_dir, evt_out), final_evt)

        final_uf_evt = os.path.join(out_dir, uf_evt_out)
        os.rename(os.path.join(temp_work_dir, uf_evt_out), final_uf_evt)

        # Make sure to remove the temporary directory
        rmtree(temp_work_dir)

        # This task should have updated an auxiliary file as well, we need to copy it
        final_make_filt = os.path.join(out_dir, f"ni{cur_obs_id}.mkf")
        os.rename(
            os.path.join(obs_dir, "auxil", f"ni{cur_obs_id}.mkf"), final_make_filt
        )

        # Populate a dictionary with the paths to generated products
        prod_lookup["events"] = final_evt
        prod_lookup["uf_events"] = final_uf_evt
        prod_lookup["make_filter"] = final_make_filt

    return cur_obs_id, out, task_success, prod_lookup


def combine_nicer_xti(
    src_name,
    event_files: List[str],
    uf_event_files: List[str],
    makefilt_files: List[str],
    orbit_files: List[str],
    out_dir: str,
):
    """ """

    event_files = list(map(lambda x: os.path.abspath(x), event_files))
    uf_event_files = list(map(lambda x: os.path.abspath(x), uf_event_files))
    makefilt_files = list(map(lambda x: os.path.abspath(x), makefilt_files))
    orbit_files = list(map(lambda x: os.path.abspath(x), orbit_files))

    out_dir = os.path.abspath(out_dir)

    cur_obs_ids = []
    for cur_evt in event_files:
        # We can extract the ObsID directly from the header of the event list - it is
        #  safer than having them be passed to this function separately.
        with fits.open(cur_evt) as read_evto:
            cur_obs_ids.append(read_evto["EVENTS"].header["OBS_ID"])

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, f"niobsmerge_{randint(0, int(1e8))}")
    os.makedirs(temp_work_dir)

    # Set up list files of parameter inputs
    with open(os.path.join(temp_work_dir, "events.lis"), "w") as evento:
        evento.writelines([f + "\n" for f in event_files])

    with open(os.path.join(temp_work_dir, "uf_events.lis"), "w") as uf_evento:
        uf_evento.writelines([f + "\n" for f in uf_event_files])

    with open(os.path.join(temp_work_dir, "makefilts.lis"), "w") as makefilto:
        makefilto.writelines([f + "\n" for f in makefilt_files])

    with open(os.path.join(temp_work_dir, "orbs.lis"), "w") as orbo:
        orbo.writelines([f + "\n" for f in orbit_files])

    # We write the ObsIDs to a file, so there is a quick way to tell which
    #  were combined for a given output directory.
    with open(os.path.join(temp_work_dir, "OBSIDS.md"), "w") as obsido:
        obsido.writelines([f + "\n" for f in cur_obs_ids])

    # Set up the output file names
    evt_out = os.path.basename(EVT_PATH_TEMP).format(oi="COMBINED", m="mpuall")

    uf_evt_out = os.path.basename(UFEVT_PATH_TEMP).format(oi="COMBINED", m="mpuall")

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        try:
            out = hsp.niobsmerge(
                indirs="NONE",
                outdir=".",
                doorbfiles=True,
                inclfiles="@events.lis",
                inufafiles="@uf_events.lis",
                inmkfiles="@makefilts.lis",
                inorbfiles="@orbs.lis",
                clfile=evt_out,
                ufafile=uf_evt_out,
                clobber=True,
                noprompt=True,
                chatter=TASK_CHATTER,
            )

            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    prod_lookup = {}

    # Moves files from the temporary output directory into the
    #  final output directory
    if (
        os.path.exists(temp_work_dir)
        and len(os.listdir(temp_work_dir)) != 0
        and task_success
    ):

        final_evt = os.path.join(out_dir, evt_out)
        os.rename(os.path.join(temp_work_dir, evt_out), final_evt)

        final_uf_evt = os.path.join(out_dir, uf_evt_out)
        os.rename(os.path.join(temp_work_dir, uf_evt_out), final_uf_evt)

        final_make_filt = os.path.join(out_dir, "nimerged.mkf")
        os.rename(os.path.join(temp_work_dir, "nimerged.mkf"), final_make_filt)

        final_orbit = os.path.join(out_dir, "nimerged.orb")
        os.rename(os.path.join(temp_work_dir, "nimerged.orb"), final_orbit)

        # Make sure to remove the temporary directory
        rmtree(temp_work_dir)

        # Populate a dictionary with the paths to generated products
        prod_lookup["events"] = final_evt
        prod_lookup["uf_events"] = final_uf_evt
        prod_lookup["make_filter"] = final_make_filt
        prod_lookup["orbit"] = final_orbit

    return src_name, cur_obs_ids, out, task_success, prod_lookup


def gen_nicer_xti_spectrum(
    src_name,
    event_file,
    uf_event_file,
    makefilt_file,
    out_dir,
    bkg_model: str = "scorpeon",
):
    """
    A wrapper function for nicerl3-spect, processing spectral extraction
    tasks in parallel across independent working directories.
    """
    event_file = os.path.abspath(event_file)
    uf_event_file = os.path.abspath(uf_event_file)
    makefilt_file = os.path.abspath(makefilt_file)
    out_dir = os.path.abspath(out_dir)

    # We can extract the ObsID directly from the header of the event list - it is
    #  safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]

    # Set up the output file names for the source spectrum we're about to generate.
    sp_out = os.path.basename(SP_PATH_TEMP).format(
        oi=cur_obs_id,
        src_name=src_name,
    )

    rmf_out = os.path.basename(RMF_PATH_TEMP).format(
        oi=cur_obs_id,
        src_name=src_name,
    )

    arf_out = os.path.basename(ARF_PATH_TEMP).format(
        oi=cur_obs_id,
        src_name=src_name,
    )

    # This tool will output a Python file with the background model
    #  definition - we define the name of the Python file here so
    #  it is easy for us to use later.
    bg_script_out = sp_out.replace(".fits", "_bg_script.py")

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, f"nicerl3_spect_{randint(0, int(1e8))}")
    os.makedirs(temp_work_dir)

    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        try:
            # indir is a null value because we specify the file names individually
            #  to account for our different storage structure
            out = hsp.nicerl3_spect(
                indir="",
                cldir=".",
                clfile=os.path.relpath(event_file),
                ufafile=os.path.relpath(uf_event_file),
                mkfile=os.path.relpath(makefilt_file),
                phafile=sp_out,
                arffile=arf_out,
                rmffile=rmf_out,
                bkgmodeltype=bkg_model,
                clobber=True,
                noprompt=True,
                chatter=TASK_CHATTER,
                cleanup=True,
                outlang="python",
                bkgscript=bg_script_out,
            )

            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    prod_lookup = {}

    # Moves files from the temporary output directory into the
    #  final output directory
    if (
        os.path.exists(temp_work_dir)
        and len(os.listdir(temp_work_dir)) != 0
        and task_success
    ):

        final_sp = os.path.join(out_dir, sp_out)
        os.rename(os.path.join(temp_work_dir, sp_out), final_sp)

        final_rmf = os.path.join(out_dir, rmf_out)
        os.rename(os.path.join(temp_work_dir, rmf_out), final_rmf)

        final_arf = os.path.join(out_dir, arf_out)
        os.rename(os.path.join(temp_work_dir, arf_out), final_arf)

        final_bg_script = os.path.join(out_dir, bg_script_out)
        os.rename(os.path.join(temp_work_dir, bg_script_out), final_bg_script)

        # Make sure to remove the temporary directory
        rmtree(temp_work_dir)

        # Populate a dictionary with the paths to generated products
        prod_lookup["spec"] = final_sp
        prod_lookup["rmf"] = final_rmf
        prod_lookup["arf"] = final_arf
        prod_lookup["bg_script"] = final_bg_script

    return src_name, cur_obs_id, out, task_success, prod_lookup


def correct_scorpeon_script(script_path: str) -> str:
    with open(script_path, "r") as scripto:
        script_cont = scripto.read()

    # For some reason these scripts are sometimes written with paths that
    #  erroneously point to a level above the current directory.
    script_cont = script_cont.replace("../", "")

    # Not sure the import of sys is necessary
    # script_cont = script_cont.replace("import sys", "# import sys")

    # Use re.escape in case the substring contains special regex characters
    patterns = {
        "AllData": rf"(?=({re.escape('AllData')}))",
        "AllModels": rf"(?=({re.escape('AllModels')}))",
    }

    # re.findall returns a list of matched strings, so we extract the span indices
    all_to_rep = [
        f"{script_cont[match.start()-1]}{patt_name}"
        for patt_name, cur_patt in patterns.items()
        for match in re.finditer(cur_patt, script_cont)
        if script_cont[match.start() - 3 : match.start()] != "xs."
    ]

    for to_rep in set(all_to_rep):
        script_cont = script_cont.replace(to_rep, f"{to_rep[0]}xs.{to_rep[1:]}")

    return script_cont


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

    # First axis (the large, top-most one) is where we will plot the spectrum
    #  data points, and fitted model lines.
    spec_ax = ax_arr[0]
    # Turn minor axis ticks on, and configure the direction they point, and that
    #  they also appear on the top and right sides of the plot.
    spec_ax.minorticks_on()
    spec_ax.tick_params(which="both", direction="in", top=True, right=True)

    # First we plot the spectrum data points, including the count rate uncertainty,
    #  and the size of each energy bin as error bars.
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

    # If the user has requested that the model fit be shown as a 'stepped' line (i.e.
    #  what the standard XSPEC plots look like), we have to use the 'stairs' method
    #  rather than the standard 'plot' method.
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
            zorder=3,
        )
    # Otherwise, the model will be plotted as a smooth line.
    else:
        spec_ax.plot(
            plot_data["energy"],
            plot_data["model"],
            color=mod_color,
            label=mod_label,
            alpha=0.8,
            linewidth=1.4,
            zorder=3,
        )

    # We allow the user to set specific x and y axis limits when they call this
    #  function - if they have passed limits, we enforce them here (the residual
    #  axis will inherit the limits as well, because we set sharex=True when
    #  we defined the figure.
    if x_lims is not None:
        spec_ax.set_xlim(x_lims)
    if y_lims is not None:
        spec_ax.set_ylim(y_lims)

    # We just assume the user wants a logged y-scale, which I don't think is too
    #  restrictive.
    spec_ax.set_yscale("log")
    # Alter the formatting of the labels so that they are 0.1, 0.01, 0.001 etc.
    spec_ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
    # And make sure to set the y-axis label
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
:tags: [hide-input]

# Generalizing the pipeline: A list of targets to process.
TARGET_LIST = ["RXJ2143.0+0654", "RXJ1856.5-3754"]
MAX_OBS_PER_TARGET = 5

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# ------------- Configure global package settings --------------
# Raise Python exceptions if a heasoftpy task fails
# TODO Remove once this becomes a default in heasoftpy
hsp.Config.allow_failure = False

# Set up the method for spawning processes.
mp.set_start_method("fork", force=True)
# --------------------------------------------------------------


# ----------- Set HEASoft to use the S3-bucket CALDB -----------
os.environ["CALDB"] = "https://nasa-heasarc.s3.amazonaws.com/caldb"
# --------------------------------------------------------------


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
    ROOT_DATA_DIR = "../../../_data/NICER/"
else:
    ROOT_DATA_DIR = "NICER/"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("NICER_output")
os.makedirs(OUT_PATH, exist_ok=True)
# --------------------------------------------------------------


# -------- Get geomagnetic data ---------
# This ensures that geomagnetic data required for NICER analyses are downloaded
GEOMAG_PATH = os.path.join(ROOT_DATA_DIR, "geomag")
os.makedirs(GEOMAG_PATH, exist_ok=True)
out = hsp.nigeodown(outdir=GEOMAG_PATH)
# ---------------------------------------


# ------------- Set up output file path templates --------------
EVT_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{src_name}",
    "{oi}",
    "nicer-xti-{m}-obsid{oi}-enALL-events.fits",
)

UFEVT_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{src_name}",
    "{oi}",
    "nicer-xti-{m}-obsid{oi}-enALL-unfilteredevents.fits",
)

SP_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{src_name}",
    "{oi}",
    "nicer-xti-obsid{oi}-enALL-spectrum.fits",
)

RMF_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{src_name}",
    "{oi}",
    "nicer-xti-obsid{oi}-enALL-spectrum.rmf",
)

ARF_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{src_name}",
    "{oi}",
    "nicer-xti-obsid{oi}-enALL-spectrum.arf",
)
# --------------------------------------------------------------
```

## 1. Data Identification and Acquisition

*To make this analysis generalized, we first dynamically locate and download the data for our chosen targets. We use the `astroquery` python module to resolve the target coordinates and cross-reference them with the NICER master catalog.*

+++

### Identifying the NICER master catalog

```{code-cell} python
Heasarc.list_catalogs(keywords="nicer")
```

```{code-cell} python
catalog_name = "nicermastr"
```

### Searching the NICER Master Catalog

*We query the catalog for observations within the default search radius of our target coordinates, sorting the results by time. We store the relevant Observation IDs in a dictionary (`TARGET_LIST`), limiting the number of observations per target to keep runtime manageable.*

```{code-cell} python
search_results = {}
rel_obsids = {}

for cur_name in TARGET_LIST:
    cur_coord = SkyCoord.from_name(cur_name)
    cur_res = Heasarc.query_region(cur_coord, catalog=catalog_name, columns="*")
    cur_res.sort("exposure", reverse=True)

    display(cur_res)
    print("\n")

    if MAX_OBS_PER_TARGET is not None:
        cur_res = cur_res[:MAX_OBS_PER_TARGET]
        rel_obsids[cur_name] = np.array(cur_res["obsid"].value)

    search_results[cur_name] = cur_res
```

### Downloading the Selected Observations

*Now that we have identified the data links for each observation, we iterate through our dictionary and download the raw data into subdirectories named after the respective target.*

```{code-cell} python
# for cur_name, cur_res in search_results.items():
#     cur_datalinks = Heasarc.locate_data(cur_res)

#     Heasarc.download_data(cur_datalinks, host="aws", location=ROOT_DATA_DIR)
```

## 2. Data Preparation

Before generating spectral products, we must download the raw observation data and run standard level-2 processing. The following cells fetch a sample observation and clean the events using `nicerl2`.

### Running Level-2 Processing in Parallel

*Because `nicerl2` takes time to run and processes independent datasets, it is highly suitable for multiprocessing. We construct a list of arguments for each observation and pass them to our parallel pool using the wrapper function defined in the Global Setup.*

```{code-cell} python
arg_combs_l2 = []
for cur_name, cur_ois in rel_obsids.items():
    for cur_oi in cur_ois:
        cur_obs_dir = os.path.join(ROOT_DATA_DIR, cur_oi)
        cur_out_dir = os.path.join(OUT_PATH, cur_name, cur_oi)

        arg_combs_l2.append([cur_obs_dir, cur_oi, cur_out_dir])


with mp.Pool(NUM_CORES) as p:
    with tqdm(total=len(arg_combs_l2), desc="Processing NICER XTI") as onwards:
        jobs = [
            p.apply_async(
                process_nicer_xti, args=item, callback=lambda _: onwards.update(1)
            )
            for item in arg_combs_l2
        ]
        l2_results = [job.get() for job in jobs]

l2_out_prods = {cur_res[0]: cur_res[3] for cur_res in l2_results if cur_res[2]}
```

## 3. Combining NICER observations

```{code-cell} python
arg_combs_nicomb = []
for cur_name, cur_ois in rel_obsids.items():
    cur_args = [cur_name]

    cur_args.append([l2_out_prods[oi]["events"] for oi in cur_ois])
    cur_args.append([l2_out_prods[oi]["uf_events"] for oi in cur_ois])
    cur_args.append([l2_out_prods[oi]["make_filter"] for oi in cur_ois])
    cur_args.append(
        [os.path.join(ROOT_DATA_DIR, oi, "auxil", f"ni{oi}.orb.gz") for oi in cur_ois]
    )

    cur_args.append(os.path.join(OUT_PATH, cur_name, "combined"))
    arg_combs_nicomb.append(cur_args)


with mp.Pool(NUM_CORES) as p:
    with tqdm(
        total=len(arg_combs_nicomb), desc="Combining NICER observations"
    ) as onwards:
        jobs = [
            p.apply_async(
                combine_nicer_xti, args=item, callback=lambda _: onwards.update(1)
            )
            for item in arg_combs_nicomb
        ]
        nicomb_results = [job.get() for job in jobs]

nicomb_out_prods = {cur_res[0]: cur_res[4] for cur_res in nicomb_results if cur_res[3]}
```

## 4. Extracting NICER spectra

Once you have your cleaned events, you are ready to make a spectrum.

<span style="color:red">Run the nicerl3-spect spectral product pipeline task with the following command.</span>

+++

`nicerl3-spect` is designed by default to work with NICER observation data within its standard observation directory.

However, in some cases the user will have done their "own" analysis and thus generated their own cleaned event files. It is still possible to use `nicerl3-spect` to generate spectral products. You can specify the exact name of the cleaned event file, UFA file, and MPU good time files using the `clfile`, `ufafile`, and `mkfile` parameters.

+++

### What can go wrong?

+++

#### No Good Time

+++

One of the most common failures of `nicerl3-spect` occurs when there is no good time. Because NICER observes targets in low Earth orbit, it is subject to many extreme variations in background and visibility. Usually `nicerl2` screens out extreme variations automatically, leaving "Good Time" (also known as a Good Time Interval or GTI).

However, occasionally, the environmental conditions are so bad that the entire observation is screened out, leaving 0 seconds of good time. When there is no good time, there are no events to extract.

`nicerl3-spect` will fail with <span style="color:red">Status 218</span> and you will see the following on the console output:

<span style="color:red">ERROR: spectrum has zero good exposure time ... ERROR: niextspect failed nicerl3-spect: ERROR: Error extracting target spectrum</span>

This simply means there is no good time. Unfortunately there is no easy fix for this, other than perhaps to relax the screening criteria you used in `nicerl2` to allow slightly more marginal data.

+++

### Generating combined spectra

+++

*Just as we did with `nicerl2`, we map our spectral extraction wrapper across all targets and observations in parallel.*

```{code-cell} python
nicer_back_mod = "scorpeon"
```

```{code-cell} python
arg_combs_comb_sp = []
for cur_name, cur_prods in nicomb_out_prods.items():

    arg_combs_comb_sp.append(
        [
            cur_name,
            cur_prods["events"],
            cur_prods["uf_events"],
            cur_prods["make_filter"],
            os.path.join(OUT_PATH, cur_name, "combined"),
            nicer_back_mod,
        ]
    )

with mp.Pool(NUM_CORES) as p:
    with tqdm(
        total=len(arg_combs_comb_sp), desc="Generating combined NICER spectra"
    ) as onwards:
        jobs = [
            p.apply_async(
                gen_nicer_xti_spectrum, args=item, callback=lambda _: onwards.update(1)
            )
            for item in arg_combs_comb_sp
        ]
        comb_spec_results = [job.get() for job in jobs]

comb_spec_prods = {
    cur_res[0]: cur_res[4] for cur_res in comb_spec_results if cur_res[3]
}
```

### Generating individual spectra

```{code-cell} python
arg_combs_indiv_sp = []

for cur_name, cur_ois in rel_obsids.items():
    for cur_oi in cur_ois:
        cur_prods = l2_out_prods[cur_oi]
        arg_combs_indiv_sp.append(
            [
                cur_name,
                cur_prods["events"],
                cur_prods["uf_events"],
                cur_prods["make_filter"],
                os.path.join(OUT_PATH, cur_name, cur_oi),
                nicer_back_mod,
            ]
        )

with mp.Pool(NUM_CORES) as p:
    with tqdm(
        total=len(arg_combs_indiv_sp), desc="Generating individual NICER spectra"
    ) as onwards:
        jobs = [
            p.apply_async(
                gen_nicer_xti_spectrum, args=item, callback=lambda _: onwards.update(1)
            )
            for item in arg_combs_indiv_sp
        ]
        indiv_spec_results = [job.get() for job in jobs]

indiv_spec_prods = {
    cur_res[1]: cur_res[4] for cur_res in indiv_spec_results if cur_res[3]
}
```

As the task runs, you will see a lot of output on the screen, detailing each step in the process. Depending on the size of the observation and the processing speed of your computer, `nicerl3-spect` may take anywhere from 10 seconds to several minutes to run.

Upon completion, `nicerl3-spect` generates multiple outputs, placing them in the same directory as the cleaned events (or the working directory, depending on configurations).

The "load" file will help get the user started quickly by giving an example script that will load all the products into XSPEC.

`nicerl3-spect` chains together multiple standard NICER tasks in the team-recommended fashion. Although not recommended, users can also run the tasks manually themselves as well. The tasks are manually extract spectrum with `niextspect`, manually apply systematic error to spectrum, manually apply QUALITY to a spectrum, manually applying GROUPING to a spectrum, manually generating responses, and manually generating backgrounds.

Although individual manual tasks are provided, the NICER team recommends to use `nicerl3-spect` to generate all spectral products in a consistent manner.

## 5. Background Models

The NICER background has variations. To account for this, the NICER team has devised several empirical background models.

By default, `nicerl3-spect` uses the **SCORPEON** model. The SCORPEON model uses a generic "template" of the background to which physical models are attached in order to model changes. The `nicerl3-spect` generates the appropriate responses, scripts, and model settings so that it can be loaded directly into XSPEC. The SCORPEON model is currently the most heavily developed by the NICER team.

There are also alternative models that the NICER team supports. Namely, the "3C50" model and the "Space Weather" model. These models are typically called "library" models. They use a large library of true "blank sky" observations and cross-match conditions with the target observation to produce a single background spectrum.

You can select a different model with the `bkgmodeltype` parameter.

For example, to run with the 3C50 model, <span style="color:red">use the following command, nicerl3-spect 1234567890 bkgmodeltype=3c50 clobber=YES</span>. The `bkgformat` parameter is optional and by default it will choose the highest fidelity format available for that model.

*Because we automated the generation of the `scorpeon` outputs above, we omit re-running the tool here to save computational time, though users can easily change the string argument passed in the multiprocessing setup loop above to swap to 3C50.*

Each of the background models has additional parameters or settings that allow more in-depth control of the model generation. `nicerl3-spect` does support these. For the SCORPEON model, the `bkgcomponents`, `bkgvariant`, `bkgsoftlanding` and `bkgver` parameters are passed to the SCORPEON modeling tasks.

+++

## 5. Fitting NICER spectra with PyXspec

+++

What happens next is really about the science. At this point, the scientist needs to understand and apply the correct spectral model to their data.

<span style="color:red">A note on how XSPEC deals with subdirectory names. XSPEC does not handle subdirectory names well. It will not understand if you run nicerl3-spect from one directory and then change to another directory. Consider the following example: nicerl3-spect 1234567890 clobber=YES cd 1234567890/xti/event_cl xspec @ni1234567890_load.xcm. On the face of it, it appears the user ran nicerl3-spect in the directory containing the observation and then changed into the xti/event_cl subdirectory to begin spectral analysis. Unfortunately, the path names for 1234567890/xti/event_cl are hard-coded in the *_load.xcm and *_bkg.xcm files that nicerl3-spect produces, and XSPEC will get confused by this.</span>

Instead of using the `.xcm` script in the CLI, we can natively load the generated spectra into PyXspec and visualize the data using Matplotlib.

*Now that we have processed multiple observations for multiple targets, we iterate through our data dictionary to plot them on a dynamically generated subplot grid.*

+++

### Setting up PyXspec

```{code-cell} python
# The strange comment on the end of this line is for the benefit of our
#  automated code-checking processes. You shouldn't import modules anywhere but
#  the top of your file, but this is unfortunately necessary at the moment
import xspec as xs  # noqa: E402

# Limits the amount of output from XSPEC that PyXspec will display
# xs.Xset.chatter = 0


# Stops any XSPEC figure output on the usual plot devices
xs.Plot.device = "/null"


# Other xspec settings
xs.Plot.xAxis = "keV"
xs.Plot.background = True
xs.Fit.statMethod = "cstat"
xs.Fit.query = "no"
xs.Fit.nIterations = 500
```

### Combined spectra

```{code-cell} python
xs.AllData.clear()
xs.AllModels.clear()

cur_sps = {}
cur_iter = 1
# for cur_name, cur_comb_sp_prods in comb_spec_prods.items():
for cur_name, cur_comb_sp_prods in [
    ["RXJ2143.0+0654", comb_spec_prods["RXJ2143.0+0654"]]
]:

    with contextlib.chdir(os.path.dirname(cur_comb_sp_prods["spec"])):
        xs.AllData(f"{cur_iter}:{cur_iter} {cur_comb_sp_prods['spec']}")

        cur_spec = xs.AllData(cur_iter)
        cur_spec.response = cur_comb_sp_prods["rmf"]
        cur_spec.response.arf = cur_comb_sp_prods["arf"]

        # This function prepares the auto-generated script for use in this notebook
        cur_bg_script_cont = correct_scorpeon_script(cur_comb_sp_prods["bg_script"])

        #
        nicer_bkgspect = cur_iter

        # This executes the string 'cur_bg_script_cont' (which contains the contents
        #  of the 'corrected' version of the background-loading script produced
        #  by nicerl3-spect) as Python code.
        # WE GENERALLY CAUTION AGAINST USING EXEC AT ALL, IT ISN'T CONSIDERED
        #  PARTICULARLY SAFE, but it is the most robust approach to loading
        #  the background models in this notebook.
        exec(cur_bg_script_cont)

        cur_iter += 1

xs.Model("tbabs*bbody", setPars={1: 0.0522, 2: 1.1})
# for data_group_ind in  range(2, len(comb_spec_prods)+1):
#     xs.AllModels(data_group_ind).untie()

xs.AllData.ignore("bad")
```

```{code-cell} python
xs.AllData.show()
```

```{code-cell} python
xs.AllModels.show()
```

```{code-cell} python
xs.Fit.renorm()
xs.Fit.perform()
```

```{code-cell} python
xs.Plot("data resid")

cur_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "residual": np.array(xs.Plot.y(plotWindow=2)),
    "residual_err": np.array(xs.Plot.yErr(plotWindow=2)),
}

cur_plot_data["energy_step"] = np.append(
    cur_plot_data["energy"] - cur_plot_data["energy_delta"],
    cur_plot_data["energy"][-1] + cur_plot_data["energy_delta"][-1],
)
```

```{code-cell} python
plot_fit_residual_spec(cur_plot_data, inst_name="EXOSAT-ME", mod_expr="tbabsxbbody")
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Author: NICER Team

Updated On: 2026-06-26

+++

## Additional Resources

NICER Support: [NICER Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=nicer)

HEASARC Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[NICER Data Analysis Threads](https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/)

## Acknowledgements

This demonstration was based off the ['Complete Spectral Product Pipeline'](https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/nicerl3-spect/) and ['Combining Multiple NICER Observations'](https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/combine-obs/) NICER data analysis threads.

## References
