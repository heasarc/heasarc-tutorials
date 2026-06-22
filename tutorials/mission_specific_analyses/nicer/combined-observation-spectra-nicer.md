---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: NICER Team
date: '2026-06-22'
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
execution:
  cal-files:
    xmm-ccf: False
    chandra: False
    xspec-models: True
title: Getting started with ROSAT All Sky Survey data
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

As of 2026-06-18, this notebook takes ~180s to run to completion (heavily dependent on archive server download speeds and local processing power, as well as the number of parallel targets chosen).

## Imports

```{code-cell} python
# %pip install --pre astroquery --upgrade
```

```{code-cell} python
# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_nicer_spectral_pipeline.txt

import contextlib
import multiprocessing as mp
import os
from random import randint
from shutil import rmtree

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.heasarc import Heasarc
from IPython.display import display
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

def process_nicer_xti(obs_dir: str, cur_obs_id: str):
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
                geomag_path=GEOMAG_PATH,
                clobber=True,
                noprompt=True,
                chatter=TASK_CHATTER,
            )

            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    # Moves files from the temporary output directory into the
    #  final output directory
    if os.path.exists(temp_work_dir) and len(os.listdir(temp_work_dir)) != 0:
        for f in os.listdir(temp_work_dir):
            os.rename(os.path.join(temp_work_dir, f), os.path.join(out_dir, f))

        # Make sure to remove the temporary directory
        rmtree(temp_work_dir)

        # This task should have updated an auxiliary file as well, we need to copy it
        os.rename(
            os.path.join(obs_dir, "auxil", f"ni{cur_obs_id}.mkf"),
            os.path.join(out_dir, f"ni{cur_obs_id}.mkf"),
        )

    return cur_obs_id, out, task_success


def gen_nicer_xti_spectrum(
    event_file,
    src_name,
    out_dir,
    bkg_model: str = "scorpeon",
    ufa_file: str = None,
    make_filt_file: str = None,
):
    """
    A wrapper function for nicerl3-spect, processing spectral extraction
    tasks in parallel across independent working directories.
    """
    event_file = os.path.abspath(event_file)

    # We can extract the ObsID directly from the header of the event list - it is
    #  safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]

    if ufa_file is None:
        ufa_file = os.path.join(
            os.path.dirname(event_file), f"ni{cur_obs_id}_0mpu7_ufa.evt"
        )

    if make_filt_file is None:
        make_filt_file = os.path.join(
            os.path.dirname(event_file), f"ni{cur_obs_id}.mkf"
        )

    out_dir = os.path.abspath(out_dir)

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
                ufafile=os.path.relpath(ufa_file),
                mkfile=os.path.relpath(make_filt_file),
                phafile=sp_out,
                arffile=arf_out,
                rmffile=rmf_out,
                bkgmodeltype=bkg_model,
                outlang="python",
                clobber=True,
                noprompt=True,
                chatter=TASK_CHATTER,
            )

            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    # Moves files from the temporary output directory into the
    #  final output directory
    if os.path.exists(temp_work_dir) and len(os.listdir(temp_work_dir)) != 0:
        for f in os.listdir(temp_work_dir):
            os.rename(os.path.join(temp_work_dir, f), os.path.join(out_dir, f))

        # Make sure to remove the temporary directory
        rmtree(temp_work_dir)

    return cur_obs_id, out, task_success
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
for cur_name, cur_res in search_results.items():
    cur_datalinks = Heasarc.locate_data(cur_res)

    Heasarc.download_data(cur_datalinks, host="aws", location=ROOT_DATA_DIR)
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
    l2_results = p.starmap(process_nicer_xti, arg_combs_l2)
```

```{code-cell} python
process_nicer_xti(*arg_combs_l2[0])
```

## 3. Extract Spectral Products

Once you have your cleaned events, you are ready to make a spectrum.

<span style="color:red">Run the nicerl3-spect spectral product pipeline task with the following command.</span>

### Running nicerl3-spect in Parallel

*Just as we did with `nicerl2`, we map our spectral extraction wrapper across all targets and observations in parallel.*

+++

def gen_nicer_xti_spectrum(event_file, src_name, out_dir, bkg_model: str = "scorpeon", ufa_file: str = None, make_filt_file: str = None):

```{code-cell} python
# arg_combs_l3 = []
# for src, obs_list in TARGET_LIST.items():
#     for obs in obs_list:
#         obs_dir = os.path.join(ROOT_DATA_DIR, src, obs)
#         arg_combs_l3.append([obs_dir, "scorpeon"])

# with mp.Pool(NUM_CORES) as p:
#     l3_results = p.starmap(gen_nicer_xti_spectrum, arg_combs_l3)
```

```{code-cell} python
test_evt = os.path.join(
    OUT_PATH, "RXJ2143.0+0654", "5617010105", "ni5617010105_0mpu7_cl.evt"
)
out_dir = os.path.join(OUT_PATH, "RXJ2143.0+0654", "5617010105")
gen_nicer_xti_spectrum(test_evt, "RXJ2143.0+0654", out_dir)
```

As the task runs, you will see a lot of output on the screen, detailing each step in the process. Depending on the size of the observation and the processing speed of your computer, `nicerl3-spect` may take anywhere from 10 seconds to several minutes to run.

Upon completion, `nicerl3-spect` generates multiple outputs, placing them in the same directory as the cleaned events (or the working directory, depending on configurations).

The "load" file will help get the user started quickly by giving an example script that will load all the products into XSPEC.

`nicerl3-spect` chains together multiple standard NICER tasks in the team-recommended fashion. Although not recommended, users can also run the tasks manually themselves as well. The tasks are manually extract spectrum with `niextspect`, manually apply systematic error to spectrum, manually apply QUALITY to a spectrum, manually applying GROUPING to a spectrum, manually generating responses, and manually generating backgrounds.

Although individual manual tasks are provided, the NICER team recommends to use `nicerl3-spect` to generate all spectral products in a consistent manner.

## 4. Background Models

The NICER background has variations. To account for this, the NICER team has devised several empirical background models.

By default, `nicerl3-spect` uses the **SCORPEON** model. The SCORPEON model uses a generic "template" of the background to which physical models are attached in order to model changes. The `nicerl3-spect` generates the appropriate responses, scripts, and model settings so that it can be loaded directly into XSPEC. The SCORPEON model is currently the most heavily developed by the NICER team.

There are also alternative models that the NICER team supports. Namely, the "3C50" model and the "Space Weather" model. These models are typically called "library" models. They use a large library of true "blank sky" observations and cross-match conditions with the target observation to produce a single background spectrum.

You can select a different model with the `bkgmodeltype` parameter.

For example, to run with the 3C50 model, <span style="color:red">use the following command, nicerl3-spect 1234567890 bkgmodeltype=3c50 clobber=YES</span>. The `bkgformat` parameter is optional and by default it will choose the highest fidelity format available for that model.

*Because we automated the generation of the `scorpeon` outputs above, we omit re-running the tool here to save computational time, though users can easily change the string argument passed in the multiprocessing setup loop above to swap to 3C50.*

Each of the background models has additional parameters or settings that allow more in-depth control of the model generation. `nicerl3-spect` does support these. For the SCORPEON model, the `bkgcomponents`, `bkgvariant`, `bkgsoftlanding` and `bkgver` parameters are passed to the SCORPEON modeling tasks.

## 5. Variation: Choosing Different Input File Names

`nicerl3-spect` is designed by default to work with NICER observation data within its standard observation directory.

However, in some cases the user will have done their "own" analysis and thus generated their own cleaned event files. It is still possible to use `nicerl3-spect` to generate spectral products. You can specify the exact name of the cleaned event file, UFA file, and MPU good time files using the `clfile`, `ufafile`, and `mkfile` parameters.

## 6. What Can Go Wrong

### No Good Time

One of the most common failures of `nicerl3-spect` occurs when there is no good time. Because NICER observes targets in low Earth orbit, it is subject to many extreme variations in background and visibility. Usually `nicerl2` screens out extreme variations automatically, leaving "Good Time" (also known as a Good Time Interval or GTI).

However, occasionally, the environmental conditions are so bad that the entire observation is screened out, leaving 0 seconds of good time. When there is no good time, there are no events to extract.

`nicerl3-spect` will fail with <span style="color:red">Status 218</span> and you will see the following on the console output:

<span style="color:red">ERROR: spectrum has zero good exposure time ... ERROR: niextspect failed nicerl3-spect: ERROR: Error extracting target spectrum</span>

This simply means there is no good time. Unfortunately there is no easy fix for this, other than perhaps to relax the screening criteria you used in `nicerl2` to allow slightly more marginal data.

## 7. Next Steps: Spectral Analysis

What happens next is really about the science. At this point, the scientist needs to understand and apply the correct spectral model to their data.

<span style="color:red">A note on how XSPEC deals with subdirectory names. XSPEC does not handle subdirectory names well. It will not understand if you run nicerl3-spect from one directory and then change to another directory. Consider the following example: nicerl3-spect 1234567890 clobber=YES cd 1234567890/xti/event_cl xspec @ni1234567890_load.xcm. On the face of it, it appears the user ran nicerl3-spect in the directory containing the observation and then changed into the xti/event_cl subdirectory to begin spectral analysis. Unfortunately, the path names for 1234567890/xti/event_cl are hard-coded in the *_load.xcm and *_bkg.xcm files that nicerl3-spect produces, and XSPEC will get confused by this.</span>

Instead of using the `.xcm` script in the CLI, we can natively load the generated spectra into PyXspec and visualize the data using Matplotlib.

*Now that we have processed multiple observations for multiple targets, we iterate through our data dictionary to plot them on a dynamically generated subplot grid.*

+++

xspec.Xset.restore('mymodel.xcm')

```{code-cell} python
# Configure PyXspec plotting device and energy axis
xs.Plot.device = "/null"
xs.Plot.xAxis = "keV"

# Prepare a multi-axes figure
fig, ax_arr = plt.subplots(
    len(TARGET_LIST), 1, figsize=(8, 4.5 * len(TARGET_LIST)), sharex=True
)

# Ensure ax_arr is iterable if there is only 1 target
if len(TARGET_LIST) == 1:
    ax_arr = [ax_arr]

for i, (src, obs_list) in enumerate(TARGET_LIST.items()):
    ax = ax_arr[i]
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

    for obs in obs_list:
        obs_dir = os.path.join(ROOT_DATA_DIR, src, obs)
        spec_file = os.path.join(obs_dir, f"ni{obs}mpu7_sr.pha")

        # Guard against observations that failed extraction (No Good Time)
        if not os.path.exists(spec_file):
            print(f"Skipping ObsID {obs} for {src}: No valid spectrum found.")
            continue

        xs.AllData.clear()
        s = xs.Spectrum(spec_file)

        xs.Plot("ldata")
        x_vals = np.array(xs.Plot.x())
        x_errs = np.array(xs.Plot.xErr())
        y_vals = np.array(xs.Plot.y())
        y_errs = np.array(xs.Plot.yErr())

        ax.errorbar(
            x_vals,
            y_vals,
            xerr=x_errs,
            yerr=y_errs,
            fmt="+",
            label=f"ObsID {obs}",
            capsize=2,
            alpha=0.8,
        )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Counts s$$^{-1}$$ keV$$^{-1}$$", fontsize=15)
    ax.set_title(f"Target: {src}", fontsize=15)
    ax.legend(fontsize=12)

ax_arr[-1].set_xlabel("Energy [keV]", fontsize=15)
ax_arr[-1].set_xlim(0.2, 10.0)

plt.tight_layout()
plt.show()
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-06-22

+++

## Additional Resources

- NICER Data Analysis Threads: https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/

## Acknowledgements


## References

This work made use of:
- HEASoft and HEASoftPy
- PyXspec

+++

About this notebook

Authors: HEASARC Team, and the Fornax team.

Contact: For help with this notebook, please open a topic in the Fornax Community Forum "Support" category.
