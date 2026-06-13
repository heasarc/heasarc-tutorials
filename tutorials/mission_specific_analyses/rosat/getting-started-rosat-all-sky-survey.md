---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: Mike Corcoran
  affiliations: [The Catholic University of America, 'HEASARC, NASA Goddard']
  orcid: 0000-0002-7762-3172
  website: https://science.gsfc.nasa.gov/sci/bio/michael.f.corcoran
date: '2026-03-12'
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
title: Getting started with ROSAT All Sky Survey data
---

# Getting started with ROSAT All Sky Survey data

## Learning Goals

By the end of this tutorial, you will be able to:
- Fetch a catalog from Vizier and cross-match it with a catalog hosted by HEASARC.
- Identify and fetch ROSAT All-Sky Survey (RASS) data relevant to a sample of sources.
- Examine pregenerated RASS images.
- Generate new RASS images with custom spatial binning and energy bands.
- Acquire the ROSAT-PSPCC Redistribution Matrix File (RMF), extract RASS spectra, and generate Ancillary Response Files (ARFs) for a sample of sources; then fit models using PyXSPEC and extract results.

## Introduction

The ROSAT All Sky Survey (RASS) was, unsurprisingly, a survey that observed the entire
sky using the ROSAT (standing for 'Röntgensatellit') X-ray mission. ROSAT launched
in 1990 and was active until the beginning of 1999, when it was shut down after
significant deterioration of the satellite's navigational systems.

Three X-ray instruments could be moved into the focal plane of the single X-ray
telescope mounted on the spacecraft (though they could not be used simultaneously):
- **High Resolution Imager (HRI)** - A micro-channel plate (MCP) imager very similar to the one flown on the Einstein Observatory in 1978. High spatial resolution (~$2^{\prime\prime}$), but effectively no spectral resolution.
- **Position Sensitive Proportional Counter B (PSPCB)** - One of a pair of proportional counters that could measure the position and energy of an incident photon using the charge produced when it was absorbed by the detector gas. Had moderate spatial resolution (~$25^{\prime\prime}$), low spectral resolution (~41% at 1 keV), and was sensitive in the 0.07–2.4 keV range.
- **Position Sensitive Proportional Counter C (PSPCC)** - The second of a pair of proportional counters, PSPC**C** was the primary instrument, and was used to perform the ROSAT All-Sky Survey at the beginning of the mission. It was destroyed in 1991 after an error caused ROSAT to slew across the Sun.

ROSAT also had an extreme ultraviolet (XUV) imager called the **Wide Field
Camera (WFC)**, with a 5$^{\circ}$ diameter field of view (FoV), a spatial resolution
of ~$2.3^{\prime}$, and was sensitive between 62–206 **eV** (~60–100 Å).

The ROSAT All-Sky Survey was taken using the ROSAT-PSPC**C** instrument, though it was
left incomplete following the destruction of the instrument in 1991. Follow-up
observations to fill in the gaps were performed using the PSPC**B** instrument much
later in the mission's life, but were taken in 'pointed' rather than 'scanning' mode, and
as such are not included in the RASS archive. Instead, they are archived with all other
pointed ROSAT observations and will not be used in this demonstration.

The effective angular resolution of RASS was worse than that of the PSPC
instruments, at **~45$^{\prime\prime}$**, as the spacecraft was constantly slewing
while taking the observations.

RASS' data are organized into 'skyfields', each with their own sequence ID. Each
skyfield represents a **$6.4^{\circ}\times6.4^{\circ}$** area of the sky, and is
built from multiple slewing observations.


This tutorial will give you the skills required to start using RASS observations to
measure X-ray properties of a set of sources. To demonstrate, we will be using a sample
of over 700 **M dwarf** stars from the 'CARMENES input catalogue of M
dwarfs' ([Alonso-Floriano F. J. et al. 2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...577A.128A/abstract)).
We won't be analyzing the entire dataset. However, there will still be a substantial number
of sources to work with, which will give you an idea of how to use RASS data for large
samples (one of the best use cases).

We also hope to make clear the limitations of what you can do with RASS data; ROSAT is
one of the older X-ray missions and utilized less sophisticated instrumentation and
optics than more modern observatories. That does impose restrictions on what we can
reasonably expect to achieve, in terms of energy range coverage, sensitivity, and
spectral/spatial resolution.

On the other hand, the ROSAT All-Sky Survey is still (as of early 2026), the only
publicly available all-sky X-ray imaging dataset, with over 1.35e+5 sources in the 'Second ROSAT all-sky survey' source
catalog (2RXS; [Boller T. et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract)). The
scientific potential of the RASS archive is still very great, and being able to directly
analyze the data, rather than rely solely on catalogs, may help you with your own research interests.

### Inputs

- The CARMENES input catalogue of M dwarfs ([Alonso-Floriano F. J. et al. 2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...577A.128A/abstract)).

### Outputs

- Visualizations of pre-processed RASS images.
- Newly generated RASS images.
- Source/background region files and spectra.
- Result table from fitting spectral models using PyXSPEC, and accompanying visualizations of spectra.

### Runtime

As of 12th March 2026, this notebook takes ~13 m to run to completion on Fornax using the 'medium' server with 16GB RAM/ 4 cores.

## Imports

```{code-cell} python
import contextlib
import multiprocessing as mp
import os
from random import randint
from shutil import copyfile, rmtree
from typing import Tuple
from warnings import warn

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo as vo
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS
from astroquery.heasarc import Heasarc
from astroquery.vizier import Vizier
from matplotlib.ticker import FuncFormatter
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, Regions, SkyRegion
from tqdm import tqdm
from xga.imagetools.misc import pix_deg_scale
from xga.products import EventList, ExpMap, Image, RateMap
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def gen_rass_image(
    event_file: str,
    out_dir: str,
    cur_seq_id: str,
    lo_en: Quantity,
    hi_en: Quantity,
    im_bin: int = 90,
):
    """
    This function wraps the HEASoft 'extractor' tool and is used to spatially bin
    ROSAT-PSPC event lists into images. The HEASoftPy interface to 'extractor' is used.

    Both the energy band and the image binning factor, which controls how
    many 'pixels' in the native SKY X-Y coordinate of the event list are binned into
    a single image pixel, can be specified.

    Default `im_bin` will produce a 512x512 image for RASS data.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate an image from.
    :param str out_dir: The directory where output files should be written.
    :param str cur_seq_id: RASS sequence ID (as found in HEASARC RASS table).
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the image.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the image.
    :param int im_bin: Number of ROSAT-PSPC SKY X-Y pixels to bin into a single image
        pixel.
    """
    # Make sure the lower and upper energy limits make sense
    if lo_en > hi_en:
        raise ValueError(
            "The lower energy limit must be less than or equal to the upper "
            "energy limit."
        )
    else:
        lo_en_val = lo_en.to("keV").value
        hi_en_val = hi_en.to("keV").value

    # Convert the energy limits to channel limits, rounding down and up to the nearest
    #  integer channel for the lower and upper bounds respectively.
    lo_ch = np.floor((lo_en / PSPC_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / PSPC_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the image we're about to generate.
    im_out = os.path.basename(IM_PATH_TEMP).format(
        oi=cur_seq_id, ibf=im_bin, lo=lo_en_val, hi=hi_en_val
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "im_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.extractor(
            filename=evt_file_chan_sel,
            imgfile=im_out,
            noprompt=True,
            clobber=True,
            binf=im_bin,
            xcolf="X",
            ycolf="Y",
            gti="STDGTI",
            events="STDEVT",
            chatter=TASK_CHATTER,
            regionfile="NONE",
        )

    # Move the output image file to the proper output directory from
    #  the temporary working directory
    os.rename(os.path.join(temp_work_dir, im_out), os.path.join(out_dir, im_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_rass_spectrum(
    event_file: str,
    out_dir: str,
    cur_seq_id: str,
    source_name: str,
    rel_src_reg: SkyRegion,
    src_reg_file: str,
    back_reg_file: str,
    wmap_im_bin: int = 8,
) -> Tuple[hsp.core.HSPResult, hsp.core.HSPResult, str, str]:
    """
    Function that wraps the HEASoftPy interface to the HEASoft extractor tool, set
    up to generate spectra from ROSAT-PSPC observations. The function will
    generate a spectrum for the source region and a background spectrum for
    the background region.

    This function will only extract events from PI channels 0-256; everything
    above 256 is dicarded. ROSAT PSPC RMF files only cover channels 0-256, so
    there is no point including anything else.

    Input region files MUST be in the Sky X-Y coordinate system. The 'rel_src_reg'
    input must be a SkyRegion object (i.e., in the RA-Dec coordinate system) defining
    the source region - we will extract RA, Dec, and radius value from it.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate a ROSAT-PSPC spectrum from.
    :param str out_dir: The directory where output files should be written.
    :param str cur_seq_id: RASS sequence ID (as found in HEASARC RASS table).
    :param str source_name: The name of the source for which we are
        generating a spectrum.
    :param SkyRegion rel_src_reg: The SkyRegion object (i.e., in the RA-Dec coordinate
        system) defining the region from which we wish to generate a source spectrum.
        RA, Dec, and radius values will be extracted from this object.
    :param str src_reg_file: Path to the region file (IN THE SKY X-Y COORDINATE SYSTEM)
        defining the source region for which we wish to generate a spectrum.
    :param str back_reg_file: Path to the region file (IN THE SKY X-Y COORDINATE SYSTEM)
        defining the background region for which we wish to generate a spectrum.
    :param int wmap_im_bin: Number of ROSAT-PSPC SKY X-Y pixels to bin into a
        single image pixel for the 'weighted map' included in ROSAT spectra.
        Default is 8. BEWARE - very low values may cause you to run
        out of memory when generating spectra from all-sky data tiles.
    """

    # Get RA, Dec, and radius values in the right format
    ra_val = rel_src_reg.center.ra.to("deg").value.round(6)
    dec_val = rel_src_reg.center.dec.to("deg").value.round(6)
    rad_val = rel_src_reg.radius.to("deg").value.round(4)

    # Set up the output file names for the source and background spectra we're
    #  about to generate.
    sp_out = os.path.basename(SP_PATH_TEMP).format(
        oi=cur_seq_id, ra=ra_val, dec=dec_val, rad=rad_val, sn=source_name
    )
    sp_back_out = os.path.basename(BACK_SP_PATH_TEMP).format(
        oi=cur_seq_id, ra=ra_val, dec=dec_val, sn=source_name
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "spec_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        # We append a PI channel limit to match the number of
        #  channels in the PSPC RMF file
        src_out = hsp.extractor(
            filename=os.path.relpath(event_file) + "[PI=0:256]",
            phafile=sp_out,
            regionfile=os.path.relpath(src_reg_file),
            xcolf="X",
            ycolf="Y",
            xcolh="X",
            ycolh="Y",
            binh=wmap_im_bin,
            ecol="PI",
            gti="STDGTI",
            events="STDEVT",
            fullimage=False,
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

        # Now for the background spectrum
        back_out = hsp.extractor(
            filename=os.path.relpath(event_file) + "[PI=0:256]",
            phafile=sp_back_out,
            regionfile=os.path.relpath(back_reg_file),
            xcolf="X",
            ycolf="Y",
            xcolh="X",
            ycolh="Y",
            binh=wmap_im_bin,
            ecol="PI",
            gti="STDGTI",
            events="STDEVT",
            fullimage=False,
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

    # Move the spectra up from the temporary directory
    fin_sp_out = os.path.join(out_dir, sp_out)
    os.rename(os.path.join(temp_work_dir, sp_out), fin_sp_out)

    fin_bsp_out = os.path.join(out_dir, sp_back_out)
    os.rename(os.path.join(temp_work_dir, sp_back_out), fin_bsp_out)

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return src_out, back_out, fin_sp_out, fin_bsp_out


def gen_rosat_pspc_arf(
    out_dir: str,
    spec_file: str,
    rmf_file: str = "CALDB",
) -> Tuple[hsp.core.HSPResult, str]:
    """
    A wrapper function for the HEASoft `pcarf` task, which we use to generate
    ARFs for ROSAT-PSPC spectra.

    :param str out_dir: The directory where output files should be written.
    :param str spec_file: The path to the spectrum file for which to generate an ARF.
    :param str rmf_file: The path to the RMF file necessary to generate an ARF.
    """

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "pcarf_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # We can use the spectrum file name to set up the output ARF file name
    arf_out = os.path.basename(spec_file).replace("-spectrum.fits", ".arf")

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():

        out = hsp.pcarf(
            phafil=os.path.relpath(spec_file),
            outfil=os.path.relpath(arf_out),
            rmffile=rmf_file,
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

    # Move the ARF file up from the temporary directory
    fin_arf_out = os.path.join(out_dir, arf_out)
    os.rename(os.path.join(temp_work_dir, arf_out), fin_arf_out)

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out, fin_arf_out
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 0

# The approximate energy per channel for ROSAT-PSPC
PSPC_EV_PER_CHAN = Quantity(9.9, "eV/chan")
```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# ------------- Configure global package settings --------------
# Raise Python exceptions if a heasoftpy task fails
# TODO Remove once this becomes a default in heasoftpy
hsp.Config.allow_failure = False

# Set up the method for spawning processes.
mp.set_start_method("fork", force=True)
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
    ROOT_DATA_DIR = "../../../_data/RASS/"
else:
    ROOT_DATA_DIR = "RASS/"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directories into which we save output files from this example.
ROOT_OUT_PATH = os.path.abspath("RASS_output")

# We're dealing with a sample of sources - some data products we generate will
#  be valid for any source in a particular RASS sequence, but others will
#  be specific to a source.
# As such, we have two skews of output paths: one for the source-specific products,
#  and one for the global products.
SEQ_OUT_PATH = os.path.join(ROOT_OUT_PATH, "global_products")
os.makedirs(SEQ_OUT_PATH, exist_ok=True)

SRC_OUT_PATH = os.path.join(ROOT_OUT_PATH, "source_products")
os.makedirs(SRC_OUT_PATH, exist_ok=True)

# --------------------------------------------------------------


# ------------- Set up output file path templates --------------
# --------- IMAGES ---------
IM_PATH_TEMP = os.path.join(
    SEQ_OUT_PATH,
    "{oi}",
    "rosat-pspc-seqid{oi}-imbinfactor{ibf}-en{lo}_{hi}keV-image.fits",
)
# --------------------------

# -------- REGIONS ---------
SRC_REG_PATH_TEMP = os.path.join(
    SRC_OUT_PATH,
    "{sn}",
    "rosat-pspc-seqid{oi}-name{sn}-source.reg",
)

BCK_REG_PATH_TEMP = os.path.join(
    SRC_OUT_PATH,
    "{sn}",
    "rosat-pspc-seqid{oi}-name{sn}-background.reg",
)
# --------------------------

# -------- SPECTRA ---------
SP_PATH_TEMP = os.path.join(
    SRC_OUT_PATH,
    "{sn}",
    "rosat-pspc-seqid{oi}-name{sn}-ra{ra}-dec{dec}-radius{rad}deg-"
    "enALL-spectrum.fits",
)

BACK_SP_PATH_TEMP = os.path.join(
    SRC_OUT_PATH,
    "{sn}",
    "rosat-pspc-seqid{oi}-name{sn}-ra{ra}-dec{dec}-enALL-back-spectrum.fits",
)
# --------------------------

# ---- GROUPED SPECTRA -----
GRP_SP_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum", "-{gt}grp{gs}-spectrum")
# --------------------------

# ---------- RMF -----------
RMF_PATH_TEMP = os.path.join(SRC_OUT_PATH, "{sn}", "rosat-pspc-seqid{oi}.rmf")
# --------------------------

# ---------- ARF -----------
ARF_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum.fits", ".arf")
# --------------------------
# --------------------------------------------------------------


# ---------- Set up preprocessed file path templates -----------
# --------- EVENTS ---------
PREPROC_EVT_PATH_TEMP = os.path.join(
    ROOT_DATA_DIR,
    "{loi}",
    "{loi}_bas.fits.Z",
)
# --------------------------

# --------- IMAGES ---------
# Specifically 'band 1' images between 0.07-2.4 keV
PREGEN_IMAGE_PATH_TEMP = os.path.join(
    ROOT_DATA_DIR,
    "{loi}",
    "{loi}_im1.fits.Z",
)
# --------------------------

# -------- EXPMAPS ---------
PREGEN_EXPMAP_PATH_TEMP = os.path.join(
    ROOT_DATA_DIR,
    "{loi}",
    "{loi}_mex.fits",
)
# --------------------------
# --------------------------------------------------------------
```

***

## 1. Fetching the CARMENES M dwarf catalog and matching to a RASS catalog

We stated in the [introduction](#introduction) that we would use the CARMENES 'input
catalog of M dwarfs' as the starting point for this demonstration. That way, we can
show you how to approach RASS data analysis for a _sample_ of sources.

To use the catalog, we're going to need to acquire it from somewhere. In this case,
that somewhere is the VizieR service ([DOI:10.26093/cds/vizier](https://doi.org/10.26093/cds/vizier)), which
we will access using the Astroquery Python module ([Ginsburg et al. 2019](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract)).

### Getting the CARMENES catalog from VizieR

We have already [imported](#imports) the `Vizier` class from Astroquery, so we can now
set up an instance of it (with some non-default arguments) that can be used to fetch
our catalog of interest.

The `row_limit=-1` argument tells Astroquery to return all rows from the catalog, and
the `columns=["**", "_RAJ2000", "_DEJ2000"]` tells it to also return every column (as
well as the VizieR-standard decimal degree RA and Dec values):

```{code-cell} python
viz = Vizier(row_limit=-1, columns=["**", "_RAJ2000", "_DEJ2000"])
viz
```

We already know the 'bibcode' of the CARMENES catalog (**J/A+A/577/A128**), but if you
didn't, you could search VizieR using the `viz` object we created.

By passing a list of keywords (every keyword must be associated with a catalog for
that catalog to be returned) to the `find_catalogs()` method, we find a few possible
matches. To narrow them down further, we can display the short description of each
returned catalog:

```{code-cell} python
cat_search = viz.find_catalogs(["CARMENES", "input"])

# Return is an ordered dictionary, with bibcode keys and catalog object values
for cur_bibcode, cur_cat in viz.find_catalogs(["CARMENES", "input"]).items():
    print(cur_bibcode, "-", cur_cat.description)
```

With the short descriptions shown above, you should be able to find the bibcode of
the catalog you're interested in.

Passing the bibcode of your chosen catalog to the `get_catalogs()` method presents
us with a `TableList` object that contains one entry per table included in the
catalog.

The CARMENES catalog we're looking at contains **two** tables:
- The first is the catalog of M dwarfs we're going to use.
- The second contains the literature references from which the catalog was compiled.

```{code-cell} python
carm_samp = viz.get_catalogs("J/A+A/577/A128")
carm_samp
```

We pull out the main catalog table, which is an Astropy `Table` object:

```{code-cell} python
carm_cat = carm_samp[0]
carm_cat
```

### Setting up a connection to the HEASARC TAP service

So, we have the catalog of M dwarfs that we want to examine using the RASS data
archive. At this point we _could_ just feed the whole set of stars into the
RASS analyses we perform later in this tutorial.

However, to simplify this demonstration, we would rather deal only with sources that
have been _detected_ in the ROSAT All-Sky Survey. To that end, we will perform a simple
spatial cross-match between the CARMENES catalog and the
2RXS ([Boller T. et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract)) catalog
of RASS sources.

We will use the HEASARC Table Access Protocol (TAP) service to perform the
cross-match, uploading the CARMENES table we just retrieved.

In order for us to be able to do that, we need to set up a connection to the HEASARC
TAP service. Here we use the [PyVO](https://github.com/astropy/pyvo) Python module to
search for the right service:

```{code-cell} python
tap_services = vo.regsearch(servicetype="tap", keywords=["heasarc"])
tap_services
```

We can extract the first entry from that search return, and we have our connection!

```{code-cell} python
heasarc_vo = tap_services[0]
```

### Writing a query to match CARMENES to 2RXS

Now we have a connection to the HEASARC TAP service, we will be able to upload
our CARMENES table and perform a simple cross-match.

All that's left is to write and submit an Astronomical Data Query
Language (ADQL) query (almost a tautology) that tells the HEASARC TAP service
to try and identify a 2RXS entry within a search radius of each CARMENES M dwarf.

We already know the HEASARC name for the 2RXS catalog (which we store in a variable
below). However, if you want to match to a different catalog that you don't already
know the HEASARC name for you might want to look at the
'{doc}`Find specific HEASARC catalogs using Python <../../heasarc_service_skills/heasarc_catalogs/finding_relevant_heasarc_catalog>`
demonstration.

```{code-cell} python
heasarc_cat_name = "rass2rxs"
```

We select a search radius of 8$^{\prime\prime}$, though you should consider your own
choice carefully as it will depend on your science goals and the type of objects you
want to look at:

```{code-cell} python
MATCH_RADIUS = Quantity(8, "arcsec")
```

Finally, we write the query itself. As ADQL queries go, it's fairly simple; the only
matching (and filtering) criteria we apply is that a 2RXS source must be within
the search radius of a CARMENES source to be considered a match.

It's worth noting that we will be able to run this query on all the CARMENES sources
at once, rather than having to run it separately for every entry.

Breaking down the query:
- `SELECT *` will return all columns from both tables.
- `FROM {hcn} as rasscat` will 'load' the HEASARC catalog with the alias 'rasscat' ({hcn} will be replaced by 'rass2rxs' in this case).
- `FROM ... tap_upload.carmenes as carm` will 'load' the table we upload (see the [query submission subsection](#submitting-the-query-to-the-heasarc-tap-service)) with the alias 'carm'.
- `WHERE contains(point('ICRS',cat.ra,cat.dec), circle('ICRS',carm.{cra},carm.{cdec},{md}))=1` will require that a 2RXS coordinate (`cat.ra` and `cat.dec`) be within the search radius of a CARMENES coordinate (`carm.{cra}` and `carm.{cdec}`) to be considered a match.

```{code-cell} python
query = (
    "SELECT * "
    "FROM {hcn} as rasscat, tap_upload.carmenes as carm "
    "WHERE "
    "contains(point('ICRS',rasscat.ra,rasscat.dec), "
    "circle('ICRS',carm.{cra},carm.{cdec},{md}))=1".format(
        md=MATCH_RADIUS.to("deg").value.round(4),
        cra="_RAJ2000",
        cdec="_DEJ2000",
        hcn=heasarc_cat_name,
    )
)

query
```

### Preparing the CARMENES catalog for upload

Actually, [writing the query](#writing-a-query-to-match-carmenes-to-2rxs) wasn't
_really_ the last thing we needed to do. Before we [upload the CARMENES table and submit
the matching query](#submitting-the-query-to-the-heasarc-tap-service) we have to make
some adjustments to the CARMENES catalog.

These adjustments are necessary to avoid errors when using the HEASARC TAP service to
run the matching query. Firstly, the HEASARC TAP service will change all column
names to their **lowercase** equivalents. So, if there are any columns that are
identically named, apart from the case of some letters, we have to rename them:

```{code-cell} python
carm_cat.rename_column("e_pEWa", "pEWa_errmi")
carm_cat.rename_column("E_pEWa", "pEWa_errpl")

carm_cat.rename_column("SpTC", "SpTColor")
```

Additionally, if you include RA and Dec columns that are in sexagesimal format (as
opposed to decimal degrees), you may encounter an error since the distance-calculation
function does not work on string data types. As such, and because
the author of this tutorial is biased against sexagesimal coordinates, we will
just remove those columns:

```{code-cell} python
carm_cat.remove_columns(["RAJ2000", "DEJ2000"])
```

Finally, we add a new column with a clean identifying name for each CARMENES source,
based on the 'No' column containing the CARMENES unique entry number. When we start
generating data products, it's good to know you have IDs to include in file and
directory names that don't include special characters or spaces.

We note that the 'Karmn' column included in the CARMENES catalog would be another
good candidate for this purpose.

```{code-cell} python
carm_cat.add_column(
    ["CARMENES-" + str(carm_id) for carm_id in carm_cat["No"]], name="id_name"
)
```

### Submitting the query to the HEASARC TAP service

All the pieces have come together, and we can run the CARMENES-2RXS cross-match query
by passing it to the `service.run_sync(...)` method of the HEASARC TAP service
connection we retrieved earlier.

The CARMENES catalog can be passed straight into the `uploads` argument as it is an
Astropy `Table` object. Note that the key of the dictionary passed to the `uploads`
argument must match the name of the table in the query
[defined previously](#writing-a-query-to-match-carmenes-to-2rxs).

```{code-cell} python
carm_2rxs_match = heasarc_vo.service.run_sync(query, uploads={"carmenes": carm_cat})
```

We can then convert the return to an Astropy `Table` and visualize it:

```{code-cell} python
carm_2rxs_match = carm_2rxs_match.to_table()
carm_2rxs_match
```

It is easy to determine the number (and percentage) of CARMENES sources that
had a match in the 2RXS catalog - we have shrunk the original catalog significantly, but
we still have a lot of sources to work with:

```{code-cell} python
num_match = len(carm_2rxs_match)
perc_match = round((num_match / len(carm_cat)) * 100, 1)

print("Number of CARMENES sources matched:", num_match)
print("Percentage of CARMENES sources matched:", f"{perc_match}%")
```

Finally, it might also be helpful to see a list of all the column names. Note that the
HEASARC TAP service has prepended the column names with the name of the table (or at
least the alias we defined in the query) they originated from:

```{code-cell} python
carm_2rxs_match.colnames
```

### Extracting CARMENES coordinates for the matched sources

In preparation for the rest of this notebook, we extract the CARMENES M dwarf
RA-Dec coordinates for the matched sources and place them in an Astropy `SkyCoord`
object:

```{code-cell} python
matched_carm_coords = SkyCoord(
    carm_2rxs_match["carm__raj2000"].value,
    carm_2rxs_match["carm__dej2000"].value,
    unit="deg",
)

matched_carm_coords[:6]
```

### Map CARMENES ID names to accepted names of the M dwarfs

Once again in preparation for the rest of this demonstration, we define a dictionary
to make it easy to map between the 'CARMENES-{ID}' names we created earlier, and the
recognized names of the CARMENES stars:

```{code-cell} python
id_name_to_actual = {en["carm_id_name"]: en["carm_name"] for en in carm_2rxs_match}
id_name_to_actual
```

## 2. Downloading relevant ROSAT All-Sky Survey data

At this point we've defined a subset of the original CARMENES M dwarf catalog whose
entries all have a match in the 2RXS catalog. We now need to download the RASS
data that is relevant to those sources.

### Getting relevant RASS sequence IDs

An added bonus we get from matching the CARMENES M dwarfs to the 2RXS catalog is that
the resulting match table contains the RASS 'skyfield number' which uniquely identifies
the ROSAT All-Sky Survey region that contains the source.

We need to retrieve the RASS data for each skyfield represented in the match table.

Extracting the skyfield numbers from the match table allows us to build a list
of RASS 'sequence IDs' which can be used to fetch the correct data from the HEASARC:

```{code-cell} python
uniq_seq_ids = np.unique(carm_2rxs_match["rasscat_skyfield_number"].value.data).astype(
    str
)
uniq_seq_ids = "RS" + uniq_seq_ids + "N00"
uniq_seq_ids
```

For convenience, we also define a dictionary that maps the 'CARMENES-{ID}' name we
gave each CARMENES source to the RASS sequence ID relevant to that source:

```{code-cell} python
src_seq_ids = {
    en["carm_id_name"]: "RS" + str(en["rasscat_skyfield_number"]) + "N00"
    for en in carm_2rxs_match
}
```

### Identifying the ROSAT All-Sky Survey 'master' table

We're going to use Astroquery's `Heasarc` object to fetch the name of
the 'master', or observation summary, table for the ROSAT All-Sky Survey. This
table has one entry per RASS sequence ID, and in the next subsection we'll indirectly
use it to retrieve links to the data files we need.

To find the right table, we pass `master=True`, to indicate we are only interested in
retrieving mission master tables, and a string of space-separated keywords (both of which
must be matched for a table to be returned):

```{code-cell} python
rass_obs_tab_name = Heasarc.list_catalogs(keywords="RASS ROSAT", master=True)[0]["name"]
rass_obs_tab_name
```

```{note}
While most missions archived by HEASARC have only one 'master' table associated with
them, ROSAT has two; 'rassmaster', which we're using in this demonstration, and 'rosmaster', which
contains information on the observations taken during the **pointed** phase of ROSAT's mission.
```

### Identifying data links for each RASS sequence ID

With the name of the RASS observation summary table in hand, we want to extract the
rows corresponding to the RASS sequence IDs relevant to our M dwarfs. We're going to
do that with another ADQL query (this time submitted through the Astroquery module, as
it is easier to use to retrieve data links than PyVO).

To prepare for the query, we construct an ADQL-compatible list of the RASS sequence IDs
we're interested in:

```{code-cell} python
seq_id_str = "('" + "','".join(uniq_seq_ids) + "')"
seq_id_str
```

Using that list, we construct and pass an ADQL query that requires that a RASS
master table row contain one of the listed RASS sequence IDs to be returned. The
return is converted to an Astropy `Table` object:

```{code-cell} python
rass_seqs = Heasarc.query_tap(
    f"SELECT * from {rass_obs_tab_name} where seq_id IN {seq_id_str}"
).to_table()

rass_seqs
```

The resulting table can then be passed to the `Heasarc.locate_data()` method, which will
return a table containing the links to the actual locations of the relevant data files:

```{code-cell} python
rass_data_links = Heasarc.locate_data(rass_seqs, rass_obs_tab_name)
rass_data_links
```

### Downloading the relevant RASS observation data

To download the data files, we can simply pass the data links table to
the `Heasarc.download_data(...)` method, with additional arguments specifying that
we want to download the data from the HEASARC AWS S3 bucket (rather than the HEASARC
FTP server), and that we want to store the downloaded data in the `ROOT_DATA_DIR`
directory specified in the [Global Setup: Constants](#constants) section:

```{code-cell} python
---
tags: [hide-output]
jupyter:
  output_hidden: true
---
Heasarc.download_data(rass_data_links, "aws", ROOT_DATA_DIR)
```

### What is included in the downloaded data?

Examining the contents of one of the directories we just downloaded, we find that
there really aren't that many files in there. The most interesting ones are:

- *{RASS SEQUENCE ID}_bas.fits.Z* - The event list for this RASS skyfield, containing tables of accepted and rejected events, as well as tables of Good Time Intervals (GTIs).
- *{RASS SEQUENCE ID}_im{BAND}.fits.Z* - Whole skyfield images generated in different energy bands;
  - **1** - 0.07-2.4 keV [full energy range of ROSAT-PSPC]
  - **2** - 0.4-2.4 keV [ROSAT-PSPC 'hard band', though a soft band by modern standards]
  - **3** - 0.07-0.4 keV [ROSAT-PSPC 'soft band']
- *{RASS SEQUENCE ID}_bk{BAND}.fits.Z* - Maps of skyfield background in different energy bands.
- *{RASS SEQUENCE ID}_mex.fits.Z* - The exposure map for the skyfield.
- *{RASS SEQUENCE ID}_anc.fits.Z* - Ancillary information about orbit and pointing of the spacecraft.

```{code-cell} python
os.listdir(os.path.join(ROOT_DATA_DIR, uniq_seq_ids[0].lower()))
```

### Examining pregenerated RASS images

We can immediately take advantage of the pregenerated images and exposure maps
included in each skyfield's data directory by loading them into `XGA` `Image` and
`ExpMap` classes, setting up count-rate maps, and visualizing the regions surrounding
our 2RXS-matched subset of CARMENES M dwarfs.

This sets up the count-rate map objects and stores them in a dictionary for later use:

```{code-cell} python
# Dictionary to store instantiated pregenerated ratemaps
pregen_ratemaps = {}

for cur_src_name, cur_seq_id in src_seq_ids.items():
    cur_im = Image(
        PREGEN_IMAGE_PATH_TEMP.format(loi=cur_seq_id.lower()),
        cur_seq_id,
        "",
        "",
        "",
        "",
        Quantity(0.07, "keV"),
        Quantity(2.4, "keV"),
    )

    cur_ex_path = PREGEN_EXPMAP_PATH_TEMP.format(loi=cur_seq_id.lower())
    # The archive inconsistenly provides compressed exposure maps
    if not os.path.exists(cur_ex_path):
        cur_ex_path += ".Z"

    cur_ex = ExpMap(
        cur_ex_path,
        cur_seq_id,
        "",
        "",
        "",
        "",
        Quantity(0.07, "keV"),
        Quantity(2.4, "keV"),
    )

    cur_rt = RateMap(cur_im, cur_ex)
    cur_rt.src_name = cur_src_name

    pregen_ratemaps[cur_src_name] = cur_rt
```

```{note}
The RASS exposure maps ({RASS SEQUENCE ID}_mex.fits(.Z)) archived by HEASARC are not
consistently compressed. Some are compressed using Zlib (with a .Z extension), while
others are not compressed at all.
```

Now we can create a fairly large figure that visualizes the RASS data for every source
in our matched subset of CARMENES M dwarfs. Each panel is centered on the CARMENES
coordinate of the M dwarf and has a half-side length configured by `ZOOM_HALF_SIDE_ANG`.

```{code-cell} python
# Half-side length for zoomed-in images centered on our sources
ZOOM_HALF_SIDE_ANG = Quantity(3, "arcmin")
```

The displayed maps are in counts-per-second, but they are not consistently scaled,
and we have not added a colorbar to indicate pixel values, so this figure is not
meant for scientific interpretation, merely visual inspection:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
num_cols = 4
fig_side_size = 3

num_ims = len(pregen_ratemaps)

num_rows = int(np.ceil(num_ims / num_cols))

fig, ax_arr = plt.subplots(
    ncols=num_cols,
    nrows=num_rows,
    figsize=(fig_side_size * num_cols, fig_side_size * num_rows),
)
plt.subplots_adjust(wspace=0.02, hspace=0.02)

ax_ind = 0
for ax_arr_ind, ax in np.ndenumerate(ax_arr):
    if ax_ind >= num_ims:
        ax.set_visible(False)
        continue

    ax.set_axis_off()

    cur_src_name, cur_rt = list(pregen_ratemaps.items())[ax_ind]

    # Fetch the actual source name from the CARMENES catalog
    cur_actual_name = carm_2rxs_match["carm_name"][ax_ind]

    # Fetch the CARMENES coordinate of the current source
    cur_coord = matched_carm_coords[ax_ind]
    # Turn the coord into an Astropy quantity, which the current version of
    #  XGA requires instead of a SkyCoord object.
    cur_coord_quan = Quantity([cur_coord.ra, cur_coord.dec], "deg")

    pd_scale = pix_deg_scale(cur_coord_quan, cur_rt.radec_wcs)
    pix_half_size = (ZOOM_HALF_SIDE_ANG / pd_scale).to("pix").astype(int)

    pix_coord = cur_rt.coord_conv(cur_coord_quan, "pix")
    x_lims = [
        (pix_coord[0] - pix_half_size).value,
        (pix_coord[0] + pix_half_size).value,
    ]
    y_lims = [
        (pix_coord[1] - pix_half_size).value,
        (pix_coord[1] + pix_half_size).value,
    ]

    cur_rt.get_view(
        ax,
        zoom_in=True,
        manual_zoom_xlims=x_lims,
        manual_zoom_ylims=y_lims,
        custom_title=cur_actual_name,
    )

    ax_ind += 1

plt.tight_layout()
plt.show()
```

An important part of working with large datasets, be they of one object or
multiple (as in this case) is **memory management**.

It might be tempting to load every image/exposure map into memory (and keep them
there) as even laptops tend to have 8–16 GB of RAM at this point. Also, reading data
from disk is slower than accessing it from memory.

_However_, memory saturation can creep up on you (with unpredictable consequences).

It's important to get into good habits, even with an older mission like ROSAT whose data
files are generally fairly small, and even with the relatively diminutive sample of
M dwarfs we're dealing with here.

As we've finished using the data associated with the pregenerated count-rate maps we
can free up some RAM by deleting the data arrays.

We do make use of the exposure maps
[to correct spectrum exposure times](#correcting-rass-exposure-times-in-spectral-files)
later on in the demonstration, but because we're using `XGA` product classes, the
exposure map data will be automatically re-loaded from disk when needed:

```{code-cell} python
for cur_rt in pregen_ratemaps.values():
    # An upcoming XGA release includes better memory management, which
    #  will remove the necessity of much of this
    del cur_rt.image.data
    del cur_rt.expmap.data

    del cur_rt._data
    cur_rt._data = None
```

## 3. Generating new RASS images

We've already made use of the pregenerated images included in the ROSAT All-Sky
Survey archive, but what if we wanted to generate new versions? This section will
take you through that process.

There are some practical limitations to what you can expect from RASS data:
- Both the spatial and energy resolutions of ROSAT All-Sky Survey data are quite
coarse; so if you want more finely binned images to tease out some spatial
features, for instance, then be cautious.

- Similarly, if you're defining custom energy ranges, you will have to carefully consider
the energy bounds you're using so as to include enough spectral channels for there
to be a usable number of photons in each pixel.

### Making event lists easily accessible

In preparation for the generation of our new RASS images (and the
[extraction of new spectra](#4-generating-rass-spectra-for-our-sample)
later on in this demonstration), we will load our skyfield event lists into `XGA`
`EventList` objects.

These objects won't read the event list **tables** into memory, at least not
automatically (we won't be interacting with them through Python in this demonstration,
so that data won't be required).

Instead, they will provide a convenient interface to the event list headers:

```{code-cell} python
preproc_event_lists = {}

for cur_src_name, cur_seq_id in src_seq_ids.items():
    cur_evt_path = PREPROC_EVT_PATH_TEMP.format(loi=cur_seq_id.lower())
    cur_evts = EventList(cur_evt_path, obs_id=cur_seq_id)
    cur_evts.src_name = cur_src_name

    preproc_event_lists[cur_src_name] = cur_evts

preproc_event_lists
```

### Defining energy bands for new images

To make images with custom energy bounds, we need to know the mapping between the
ROSAT-PSPC PI channels and energies, as the image generation tool
[we're about to use](#running-image-generation) wants us to specify **channel bounds**, rather than energy
bounds.

The energy-channel scaling for ROSAT-PSPC is well known, and we have defined a
`PSPC_EV_PER_CHAN` Astropy `Quantity` constant in the [Global Setup: Constants](#constants)
section. You could also derive this value from the ROSAT-PSPC**C** Redistribution
Matrix File (RMF), which describes the relationship between
channels and energy - we fetch the RMF in a [later section](#redistribution-matrix-file-rmf).

Now we get to define the energy bounds for the images we want to generate.

As we've previously mentioned, RASS' energy range is quite limited by modern
standards, only 0.07–2.4 keV. For this demonstration we make new images in two
energy bands; 0.5–2.0 keV and 1.0–2.0 keV.

The 0.5–2.0 keV band is often referred to as the 'soft band', at least in X-ray
galaxy cluster studies (every field seems to have its own definition of what 'soft'
means), and might be useful for comparisons to images from other missions.

```{code-cell} python
rass_im_en_bounds = Quantity([[0.5, 2.0], [1.0, 2.0]], "keV")
```

```{note}
If you run this demonstration with a modified `rass_im_en_bounds` variable, note that
even a single energy band should be defined as though it were part of a list
(e.g., `Quantity([[0.5, 2.0]], "keV")`), to make it compatible with the image
generation function we use later in the notebook.
```

Converting those energy bounds to channel bounds is straightforward, we simply
divide the energy values by our assumed mapping between energy and channel.

The resulting lower and upper bound channel values are rounded down and up to
the nearest integer channel respectively.

```{code-cell} python
rass_im_ch_bounds = (rass_im_en_bounds / PSPC_EV_PER_CHAN).to("chan")
rass_im_ch_bounds[:, 0] = np.floor(rass_im_ch_bounds[:, 0])
rass_im_ch_bounds[:, 1] = np.ceil(rass_im_ch_bounds[:, 1])
rass_im_ch_bounds = rass_im_ch_bounds.astype(int)
rass_im_ch_bounds
```

```{note}
Though we demonstrate how to convert energy bounds to channel bounds above, the
wrapper function for image generation will repeat this exercise, as it will
write energy bounds into output file names.
```

### Image binning factor

The final choice we have to make before generating new images is the
'binning factor' (or factors). These control the spatial resolution of the output
images, and are essentially the number of RASS' Sky X-Y 'pixels' that get binned
into a single output image pixel.

Archived RASS images were created with a **binning factor of 90**, resulting in a
**512x512** grid, and a pixel scale of **45$^{\prime\prime}$**.

Calculating the binning factor required for a particular image pixel scale is
quite straightforward. We can pull the intrinsic Sky X-Y pixel scale from the
header of an events list, then divide our desired pixel scale by that number.

As we're extracting the Sky X-Y pixel scale from _only_ the **TCDLT1** entry (there is
another equivalent value for the y-direction stored under **TCDLT2**) there is an
implicit assumption here that the Sky X-Y pixels are square, but that is reasonable.

Here we demonstrate calculating the binning factor for a pixel scale of
$1^{\prime}$; the chain of method calls (`.to('').round(0).astype(int).value`)
applied to the calculation:
1. Ensures the Astropy quantity result is dimensionless, rather than in units of $\frac{\prime}{\circ}$.
2. Rounds to the nearest integer.
3. Converts the data type to integer and then reads out the integer value from the Astropy quantity.

```{code-cell} python
cur_evts = list(preproc_event_lists.values())[0]
cur_skyxy_ps = abs(Quantity(cur_evts.event_header["TCDLT1"], "deg/pix"))

calc_ibf = (Quantity(1, "arcmin/pix") / cur_skyxy_ps).to("").round(0).astype(int).value
calc_ibf
```

We have somewhat arbitrarily chosen two coarser binning factors for this
demonstration, corresponding to pixel scales of $90^{\prime\prime}$ and
$135^{\prime\prime}$ respectively:

```{code-cell} python
# List of binning factors for the new images
bin_factors = [180, 270]
```

```{danger}
Choosing very small values for the binning factor, for instance **1**, will mean
that generation of new images will consume a great deal of memory, and output files
will be very large.

Incidentally, there would be very little point to generating images at the Sky X-Y
pixel scale for RASS, as it would **dramatically** oversample the practical angular
resolution of the survey.
```

### Running image generation

There is no HEASoft tool specifically intended to generate RASS images, but there is a
generalized HEASoft image (and other data product) generation task that we can use.

If you have previously generated images, light curves, or spectra from HEASARC-hosted
X-ray data on the command line, you may well have come across `XSELECT`; a HEASoft
tool for interactively generating data products from event lists.

When creating data products, `XSELECT` calls the HEASoft `extractor` task, which we
will now use to demonstrate the creation of RASS images.

As with all uses of HEASoft tasks in this notebook, our call to `extractor` will be
through the HEASoftPy Python interface - specifically the `hsp.extractor` function.

We have implemented a wrapper to this function in the
[Global Setup: Functions](#functions) section of this notebook, primarily so that we
can easily run the generation of new images in parallel:

```{code-cell} python
arg_combs = [
    [
        cur_evts.path,
        os.path.join(SEQ_OUT_PATH, cur_evts.obs_id),
        cur_evts.obs_id,
        *cur_bnds,
        cur_bf,
    ]
    for cur_evts in preproc_event_lists.values()
    for cur_bnds in rass_im_en_bounds
    for cur_bf in bin_factors
]

with mp.Pool(NUM_CORES) as p:
    im_result = p.starmap(gen_rass_image, arg_combs)
```

### Example visualization of new images

To show off the various images we just created for each RASS skytile relevant
to our M dwarf sample, we create a figure that displays them in a grid; each column
corresponds to a different energy band, and each row to a different binning factor:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
demo_evts = list(preproc_event_lists.values())[0]
demo_seq_id = demo_evts.obs_id

im_side_size = 5.5

num_cols = len(rass_im_en_bounds)
num_rows = len(bin_factors)

fig, ax_arr = plt.subplots(
    ncols=num_cols,
    nrows=num_rows,
    figsize=(im_side_size * num_cols, im_side_size * num_rows),
)
plt.subplots_adjust(wspace=0.0, hspace=0.06)

for cur_bnd_ind, cur_bnd in enumerate(rass_im_en_bounds):
    cur_lo = cur_bnd[0].to("keV").value
    cur_hi = cur_bnd[1].to("keV").value

    for cur_bf_ind, cur_bf in enumerate(bin_factors):
        cur_im_path = IM_PATH_TEMP.format(
            oi=demo_seq_id,
            ibf=cur_bf,
            lo=cur_bnd[0].to("keV").value,
            hi=cur_bnd[1].to("keV").value,
        )
        cur_im = Image(cur_im_path, demo_seq_id, "", "", "", "", *cur_bnd)

        cur_ax = ax_arr[cur_bf_ind, cur_bnd_ind]
        cur_ax.set_axis_off()

        cur_im.get_view(
            cur_ax,
            zoom_in=True,
            custom_title=f"{demo_seq_id} - {cur_lo}-{cur_hi} keV - "
            f"binning factor {cur_bf}",
        )

plt.show()
```

## 4. Generating RASS spectra for our sample

The image data products we generated in the
[previous section](#3-generating-new-rass-images) were general, valid for any source
that happens to be within those skytiles. It might not even be necessary to make new
skytile images for your science case, the archived images could well be sufficient.

Now, though, we generate **spectra** - these data products are specific to the
sources we want to study and **are not** contained in the RASS archive.

The energy resolution/range of ROSAT All-Sky Survey data is quite limited, as was the
sensitivity of the PSPC instrument and the average exposure time across the survey.

However, as we've previously stated, this is still the **only** publicly available
imaging X-ray dataset across the entire sky - it is quite possible that RASS spectra
can still lend insight to your research.

### Defining the size of source and background regions

To make a useful spectrum, we have to define a spatial region that we
have decided will contain photons emitted by our source. That will have to be done
for each source we want to study.

For this particular demonstration, we will also define _another_ region per M dwarf, which
will define where the background spectrum is extracted from. Though there are other methods
for handling the ROSAT-PSPC background
(the [`pcparpha`](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/pcparpha.html)
tool will generate particle background spectra for ROSAT-PSPC, for instance), they are
outside the scope of this demonstration.

In a more complex case - for instance, if we were studying an extended source like a
galaxy cluster or a supernova remnant - we might also have to worry about excluding
regions unrelated, contaminating, sources. That is outside the scope of this getting
started guide, however, and we do not exclude any regions in this demonstration.


This tutorial creates identically sized source and background regions for each source
in our sample. Source regions are circles, with their angular radii set to the value of
`SRC_ANG_RAD`, centered on the CARMENES RA-Dec coordinate; background regions are also
centered on the CARMENES coordinate, but are circular annuli, with inner radii set to
`INN_BACK_FACTOR`$\times$`SRC_ANG_RAD` and outer radii set
to `OUT_BACK_FACTOR`$\times$`SRC_ANG_RAD`:

```{code-cell} python
SRC_ANG_RAD = Quantity(2, "arcmin")

INN_BACK_FACTOR = 1.05
OUT_BACK_FACTOR = 1.5
```

You could alter the values above to adjust the size of the regions for your own
use. It would also be straightforward to modify the
[setting up Astropy regions](#setting-up-astropy-regions-representing-source-and-background-regions)
section to be able to specify different region sizes for different sources.


### Constructing RA-Dec ↔ RASS sky pixel WCSes

In the previous section, we talked about defining the size and location of
source/background regions in terms of angular radius and RA-Dec central coordinates.

Unfortunately, the `extractor` tool we're going to use to make our spectra wants us to
pass regions that are in the RASS Sky X-Y coordinate system (there can be issues
with the generation of ['weighting maps'](#defining-image-binning-for-the-weighting-maps) for RASS data if not) - so we need to make
sure that we're able to convert from RA-Dec to Sky X-Y.

Luckily for us, the RASS event lists we'll be creating new spectra from contain
the necessary information to construct Astropy World Coordinate System (WCS) objects
that can do just that.

Here we set up one WCS per skytile (which is the same thing as saying one per source
in this case) and store them in a dictionary for later access and use. The following
information is being pulled from the event list's **STDEVT** table header (through
an `XGA` `EventList` object):
- **Pixel scale** - One entry per direction (i.e., X and Y), and assigned to the `cdelt` property, these describe how many degrees a single RASS Sky X-Y pixel step corresponds to.
- **Critical pixel** - An X-Y pixel coordinate that acts as a reference point for how Sky X-Y coordinates map onto RA-Dec coordinates, works in concert with the next entry.
- **RA-Dec at critical pixel** - The RA-Dec coordinate corresponding to the Sky X-Y coordinate defined by the previous entry.
- **Sky X-Y ↔ RA-Dec projection** - Which projection is used to map the cartesian Sky X-Y coordinate system onto the spherical RA-Dec coordinate system.

```{code-cell} python
radec_skyxy_wcses = {}

for cur_src_name, cur_evts in preproc_event_lists.items():
    cur_wcs = WCS(naxis=2)
    cur_wcs.wcs.cdelt = [
        cur_evts.event_header["TCDLT1"],
        cur_evts.event_header["TCDLT2"],
    ]

    cur_wcs.wcs.crpix = [
        cur_evts.event_header["TCRPX1"],
        cur_evts.event_header["TCRPX2"],
    ]

    cur_wcs.wcs.crval = [
        cur_evts.event_header["TCRVL1"],
        cur_evts.event_header["TCRVL2"],
    ]

    cur_wcs.wcs.ctype = [
        cur_evts.event_header["TCTYP1"],
        cur_evts.event_header["TCTYP2"],
    ]

    radec_skyxy_wcses[cur_src_name] = cur_wcs
```

### Setting up Astropy regions representing source and background regions

We are now ready to set up our source and background regions; first in RA-Dec
coordinates (with angular radii), and then also in Sky X-Y coordinates (and sky
pixel radii).

The latter regions will be saved to disk as region files and ultimately passed to the
HEASoft `extractor` tool when we create our new spectra.

In order to simplify the definition of these regions, we will use the
Astropy-affiliated [`regions` module](https://github.com/astropy/regions). It's also
worth considering that this approach scales well to more complicated cases where
we need to exclude a set of contaminating regions before we extract a spectrum.

Iterating through each source in our sample, we:
1. Fetch the appropriate WCS for the current source (set up in the [previous section](#constructing-ra-dec--rass-sky-pixel-wcses)).
2. Fetch the CARMENES RA-Dec coordinate for the current source.
3. Create an Astropy `CircleSkyRegion` instance, centered on the CARMENES RA-Dec, with a radius taken from `SRC_ANG_RAD`, and store it in a dictionary for later.
4. Repeat the last step but instantiate a `CircleAnnulusSkyRegion` for the background region, with an inner radius of `INN_BACK_FACTOR`$\times$`SRC_ANG_RAD` and an outer radius of `OUT_BACK_FACTOR`$\times$`SRC_ANG_RAD`.
5. Convert both source and background regions to the Sky X-Y coordinate system and write them to disk as DS9-formatted region files.

```{code-cell} python
src_bck_radec_regs = {n: {"src": None, "bck": None} for n in src_seq_ids.keys()}
src_bck_skyxy_reg_files = {n: {"src": None, "bck": None} for n in src_seq_ids.keys()}

for cur_src_ind, cur_name_wcs in enumerate(radec_skyxy_wcses.items()):
    cur_src_name, cur_wcs = cur_name_wcs

    cur_src_coord = matched_carm_coords[cur_src_ind]

    cur_src_radec_reg = CircleSkyRegion(cur_src_coord, SRC_ANG_RAD)
    src_bck_radec_regs[cur_src_name]["src"] = cur_src_radec_reg

    cur_bck_radec_reg = CircleAnnulusSkyRegion(
        cur_src_coord, SRC_ANG_RAD * INN_BACK_FACTOR, SRC_ANG_RAD * OUT_BACK_FACTOR
    )
    src_bck_radec_regs[cur_src_name]["bck"] = cur_bck_radec_reg

    # Convert to Sky X-Y coordinates
    cur_src_skyxy_reg = cur_src_radec_reg.to_pixel(cur_wcs)
    cur_bck_skyxy_reg = cur_bck_radec_reg.to_pixel(cur_wcs)

    # Write Sky X-Y regions to files
    os.makedirs(os.path.join(SRC_OUT_PATH, cur_src_name), exist_ok=True)

    cur_src_skyxy_reg_path = SRC_REG_PATH_TEMP.format(
        sn=cur_src_name, oi=src_seq_ids[cur_src_name]
    )
    with open(cur_src_skyxy_reg_path, "w") as srco:
        srco.write(
            Regions([cur_src_skyxy_reg])
            .serialize(format="ds9")
            .replace("image", "physical")
        )
    src_bck_skyxy_reg_files[cur_src_name]["src"] = cur_src_skyxy_reg_path

    cur_bck_skyxy_reg_path = BCK_REG_PATH_TEMP.format(
        sn=cur_src_name, oi=src_seq_ids[cur_src_name]
    )
    with open(cur_bck_skyxy_reg_path, "w") as bcko:
        bcko.write(
            Regions([cur_bck_skyxy_reg])
            .serialize(format="ds9")
            .replace("image", "physical")
        )
    src_bck_skyxy_reg_files[cur_src_name]["bck"] = cur_bck_skyxy_reg_path
```

```{note}
During the preparation of the RASS Sky X-Y coordinate system region files using the
Astropy-affiliated `regions` module, we generate a serialization (the string contents
of the final file) of each region, rather than simply writing directly to disk using
`Regions([...]).write(_region file path_, _format_).

This is because we need to replace the coordinate system name that is automatically
used for all non-RA-Dec files written by the `regions` module (**image**), with
**physical**, which is what the `extractor` tool will be expecting.
```

### Defining image binning for the 'weighting maps'

As it turns out, when we create a new spectrum with HEASoft's `extractor` task, we're
also generating an image and storing it in a FITS image extension of the
spectrum file.

The new image stored in each spectrum is a 'weighted map' (or 'WMAP', and no not the
CMB observatory), and will be used during the [generation of Ancillary Response
Files (ARFs)](#ancillary-response-files-arf).

ARFs describe the 'effective area' (i.e., sensitivity) as a function of
incident-photon-energy. The ARFs used during normal analyses are a combination of the
X-ray optics' (called the X-ray Mirror Assembly, or XMA, for ROSAT) and the
energy-dependent efficiency of the detector.

A WMAP is essentially the same as a 'normal' X-ray image and allows ARF calculation
to find the average of ROSAT-PSPC response across the source region, weighted by the
number of photons arriving at each point.

Weighted ARF calculation is particularly important for scanning-mode observations such
as those that comprise the ROSAT All-Sky Survey, as the X-ray sky is
drifting through the PSPC**C** FoV, see
[Belloni T., et al. (1994)](https://ui.adsabs.harvard.edu/abs/1994A%26A...283.1037B/abstract)
for a discussion. Using the WMAP in ARF generation is meant to help account for this
movement across the instrument, but its efficacy has not been well explored.

Also worth noting is that weighting ARFs is also very important for the analysis
of extended sources (true of all spectro-imaging X-ray missions), though we are
treating all our M dwarfs as point sources for this tutorial.

All that said, we need to choose a binning factor (the same idea as
[when we generated new images in the last section](#image-binning-factor)) for the
WMAPs that will be generated with our new spectra. We select the same binning factor as
was used to make archived RASS images; you may wish to experiment with different values
to see how they affect the resulting ARFs.

```{code-cell} python
wmap_bin_factor = 90
```

### Running spectrum generation

Here we run the actual generation of spectra for each M dwarf in our sample; just
like with our [image generation](#running-image-generation), individual products
will be generated in parallel, maximizing the use of our computing resources and
saving us some time.

The creation of new spectra from ROSAT All-Sky Survey data is achieved through the
use of HEASoft's `extractor` task (again, just like our image generation) - this
demonstration uses the Python interface provided by HEASoftPy; `hsp.extractor(...)`.

We have set up a wrapper function, `gen_rass_spectrum(...)`, to generate RASS spectra
in the ['Global Setup: Functions'](#functions) section of this notebook, mostly to make
it easier to parallelize.

Now we generate one source, and one background, spectrum per M dwarf, passing paths to
the [region files we set up](#setting-up-astropy-regions-representing-source-and-background-regions),
the RA-Dec source region object (so that coordinates and radius can be extracted and
included in the file name), and the binning factor we [defined above](#defining-image-binning-for-the-weighting-maps):

```{code-cell} python
arg_combs = [
    [
        cur_evts.path,
        os.path.join(SRC_OUT_PATH, cur_name),
        cur_evts.obs_id,
        cur_name,
        src_bck_radec_regs[cur_name]["src"],
        src_bck_skyxy_reg_files[cur_name]["src"],
        src_bck_skyxy_reg_files[cur_name]["bck"],
        wmap_bin_factor,
    ]
    for cur_name, cur_evts in preproc_event_lists.items()
]

with mp.Pool(NUM_CORES) as p:
    sp_result = p.starmap(gen_rass_spectrum, arg_combs)
```

```{important}
Technically the ROSAT-PSPC PI channel range goes up to **500**, but only the
first **256** are actually usable for analysis. Still, `extractor(...)` would
create 500-channel spectra if you let it, and those files would be incompatible
with 256-channel RMF we will fetch in the [next section](#redistribution-matrix-file-rmf).

As such, the file path passed to `extractor` in the `gen_rass_spectrum()` function
(see the ['Global Setup: Functions'](#functions) section of this notebook), has a
channel filter command appended to it - "[PI=0:256]". This ensures that only the
valid channels are considered, and makes the spectrum compatible with the RMF.
```

### Generating supporting files

A spectrum tells us how many photons were observed by ROSAT-PSPC in each detector
channel, but to turn that into information about what was **emitted** by a source, we
need a couple more data products.

#### Redistribution Matrix File (RMF)

RMFs describe how a detector's channels correspond to the _energies_ of incident
photons - knowing that takes the spectrum from photons observed in a particular
channel, to photons observed at a particular energy.

That is clearly useful, as different astrophysical processes can be responsible for
different photon energies, and we need an energy spectrum in our observer's frame to
be able to explore the source's rest frame spectrum.

For RASS, we need to fetch the ROSAT-PSPC**C** RMF from the
[HEASARC Calibration Database (CALDB)](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_intro.html). HEASoft's `quzcif`
tool (we're using the HEASoftpy interface) allows us to query the HEASARC CALDB for
specific files - it can both return the names of matching files and download the file
itself.

Many arguments can be passed to this tool to narrow down the files that are returned,
including the name of the mission, instrument, and the type of CALDB file we're looking
for.

In this case we set `mission="rosat"` (of course), the instrument as 'pspcc' (as
ROSAT-PSPC**C** was used for the all-sky survey, and PSPC**B** for pointed
observations), and the CALDB `codename="MATRIX"` (RMFs are stored under this codename).

The only other argument we pass to filter the search results is `expr="pich.eq.256"`; this
translates as "return matching files where a PI channel boundary is equal to 256". In this
case 256 is the upper boundary of the valid PI range of the RMF we wish to retrieve.

ROSAT's CALDB entry includes a **PROS 34-channel variant** of the RMF, which would be
invalid for our purposes. It is important to know, though, that the 256-channel RMF
greatly oversamples the energy response of the PSPC. The 34-channel PROS response
is a better match to the intrinsic energy resolution.

As we set `retrieve=True` the RMF will be downloaded to our current directory, so we
use a context manager that temporarily changes the working directory to
`ROOT_DATA_DIR`, and set up a variable containing the path to the freshly acquired RMF:

```{code-cell} python
# This will find and download (retrieve=True) the ROSAT-PSPCC RMF file for
#  the 256 standard channel data
with contextlib.chdir(ROOT_DATA_DIR):
    caldb_rmf_ret = hsp.quzcif(
        mission="rosat",
        instrument="pspcc",
        codename="MATRIX",
        filter="-",
        date="-",
        time="-",
        expr="pich.eq.256",
        noprompt=True,
        retrieve=True,
        clobber=True,
    )

    # Store the path to the downloaded RMF in a variable, we'll use this later
    single_rmf_path = os.path.join(ROOT_DATA_DIR, caldb_rmf_ret.output[0].split(" ")[0])
```

```{important}
RMFs fundamentally describe the calibration of an instrument's channels and event
energies. Both the understanding of that calibration, and the behaviors of the
instrument itself, often evolve with time. As such, if you are retrieving other RMFs
using `quzcif` you should set the time and date filters to ensure you retrieve a file
that matches your observation.

We do not set times and dates here because only one RMF was released for
ROSAT-PSPC**C**, as the instrument was destroyed fairly early in the mission's lifetime.
```

We just downloaded the appropriate RMF for RASS, so now we'll make a copy for each
spectrum we've generated (a little wasteful, but it is a small file):

```{code-cell} python
rmf_paths = []

for cur_src_name, cur_seq_id in src_seq_ids.items():
    out_rmf_path = RMF_PATH_TEMP.format(oi=cur_seq_id, sn=cur_src_name)
    copyfile(single_rmf_path, out_rmf_path)

    rmf_paths.append(out_rmf_path)
```

#### Ancillary Response Files (ARF)

HEASoft's `pcarf` task is specifically intended to make ARFs for ROSAT-PSPC data, so
we will make good use of it!

We discussed what ARFs are
[in an earlier section](#defining-image-binning-for-the-weighting-maps) - the
benefit of understanding the instrument's sensitivity as a function of energy is
that we can model what the original 'real' spectrum emitted by the source would have
to be, to match the spectrum said instrument has observed.

That modeling is basically what X-ray spectral fitting is all about - various tools
exist to perform the task, but in [Section 5](#5-fitting-spectral-models-using-pyxspec) we'll
use the PyXSPEC module.

We wish to parallelize the generation of ARFs for different spectra, and so create a
wrapper function (`gen_rosat_pspc_arf(...)`, see the ['Global Setup: Functions'](#functions) section)
around the `pcarf` task to make that easier.

The inputs are very limited, only the directory to write the ARF to, the path to the
source spectrum file (extracted from the return of the spectrum generation
step; `sp_result[cur_ind][2]`), and a path to the RMF file are required (the RMF path
can also be set to 'CALDB' to automatically acquire it for this process):

```{code-cell} python
arg_combs = [
    [
        os.path.join(SRC_OUT_PATH, cur_name),
        sp_result[cur_ind][2],
        single_rmf_path,
    ]
    for cur_ind, cur_name in enumerate(preproc_event_lists)
]

with mp.Pool(NUM_CORES) as p:
    arf_result = p.starmap(gen_rosat_pspc_arf, arg_combs)
```

### Correcting RASS exposure times in spectral files

An important quirk of ROSAT All-Sky Survey data is that the 'LIVETIME' information (the
amount of time the detector was 'on source' and collecting data) contained in the
event list is unhelpful/incorrect in two ways:
1. It reports the total 'live time' for the entire skytile, which was observed in many passes; this means a much larger live time than was actually spent on a source.
2. The live time entry is deliberately negative (e.g., a whole-skytile livetime of 17961 s is reported as -17961 s) to ensure it isn't accidentally used in analyses.

Our newly generated spectra will have inherited that incorrect information, and so we need
to fix it.

Thankfully, that's pretty straightforward; we can use the exposure maps included in the
archived RASS data directories (the contents of which we discussed in
[a previous section](#what-is-included-in-the-downloaded-data)) to look up the
_actual_ exposure time at the coordinates of each spectrum's source.

Exposure times at source coordinates are fetched from the `ExpMap` instances we
[set up earlier](#examining-pregenerated-rass-images) by passing the relevant
source coordinate to the `get_exp(...)` method.

The FITS headers of the source and background spectra are then updated so that the
"EXPOSURE" entry is set to the newly extracted exposure time:

```{code-cell} python
for cur_ind, cur_src_name in enumerate(preproc_event_lists):

    cur_sp_path = sp_result[cur_ind][2]
    cur_bsp_path = sp_result[cur_ind][3]

    cur_coord = matched_carm_coords[cur_ind]
    cur_coord_quan = Quantity([cur_coord.ra, cur_coord.dec], "deg")

    cur_ex = pregen_ratemaps[cur_src_name].expmap
    del cur_ex.data
    cur_exp_time = cur_ex.get_exp(cur_coord_quan)

    with fits.open(cur_sp_path, mode="update") as speco:
        for en in speco:
            if "EXPOSURE" in en.header:
                del en.header["EXPOSURE"]
            en.header["EXPOSURE"] = cur_exp_time.to("s").value.round(5)

    with fits.open(cur_bsp_path, mode="update") as bspeco:
        for en in bspeco:
            if "EXPOSURE" in en.header:
                del en.header["EXPOSURE"]
            en.header["EXPOSURE"] = cur_exp_time.to("s").value.round(5)
```

### Grouping spectral channels

RASS spectra are likely to be low signal-to-noise due to the small
effective area of ROSAT-PSPC (relative to many modern missions), and the short
mean exposure time of the survey (~400 s). That said, due to the scanning pattern
of the ROSAT All-Sky Survey, the ecliptic pole regions have considerably longer
total exposure times. As such, sources in these regions (the Magellanic Clouds, for
instance) can have quite high signal-to-noise compared to the rest of the sky.

As such, it is normally going to be a good idea to 'group' the channels of a RASS
spectrum; combining sequential channels into a single bin until a particular quality
metric is reached (e.g., a minimum number of counts, or a minimum signal-to-noise).

Some missions have created their own tools to perform this task, but HEASoft includes
a generalized task called `ftgrouppha` that can be applied to any spectrum.

Several grouping metrics are implemented in `ftgrouppha`; we'll take the simplest
option and require a minimum number of counts per channel. The following will
be passed to the task and will group channels until there are at least three counts:

```{code-cell} python
spec_group_type = "min"
spec_group_scale = 3
```

Now we will apply `ftgrouppha` to each source spectrum, and save the output as a new
grouped spectrum file. That grouped spectrum file is what will be used for model
fitting in [Section 5](#5-fitting-spectral-models-using-pyxspec).

Grouping a spectrum in this manner is not particularly computationally expensive, so
we have not bothered to write a wrapper function for `ftgrouppha` and parallelize the
process as we did for the product generation tasks. Note, however, that if you are
working on a much larger sample, you may want to take the time to parallelize this step.

The paths to the initial spectra are retrieved from the output of the spectrum
generation step (`sp_result[cur_ind][2]`), and the paths to the output files are
stored in a dictionary (`grouped_sp_paths`) for later use:

```{code-cell} python
grp_spec_paths = {}

for cur_ind, cur_src_name in enumerate(preproc_event_lists):

    cur_sp_path = sp_result[cur_ind][2]

    cur_grp_spec = cur_sp_path.replace(
        "-spectrum", f"-{spec_group_type}grp{spec_group_scale}-spectrum"
    )

    hsp.ftgrouppha(
        infile=cur_sp_path,
        outfile=cur_grp_spec,
        grouptype=spec_group_type,
        groupscale=spec_group_scale,
        clobber=True,
        chatter=TASK_CHATTER,
        noprompt=True,
    )

    grp_spec_paths[cur_src_name] = cur_grp_spec
```

### Adding supporting file paths to spectrum headers

The very last thing we want to do before we can fit models to our spectra is to
alter their FITS headers so they point to the ARF, RMF, and background files.

We don't absolutely *need* to do this, as the paths to supporting files can be
manually passed to PyXSPEC (and command-line XSPEC) as the data are loaded in.

However, including the paths in the headers means the ARF, RMF, and background spectra
can be loaded in automatically, so we might as well:

```{code-cell} python
for cur_ind, cur_src_name in enumerate(preproc_event_lists):

    cur_gsp_path = grp_spec_paths[cur_src_name]
    cur_bsp_path = sp_result[cur_ind][3]

    cur_arf_path = arf_result[cur_ind][1]
    cur_rmf_path = rmf_paths[cur_ind]

    with fits.open(cur_gsp_path, mode="update") as speco:

        del speco["SPECTRUM"].header["RESPFILE"]
        speco["SPECTRUM"].header["RESPFILE"] = os.path.basename(cur_rmf_path)

        del speco["SPECTRUM"].header["ANCRFILE"]
        speco["SPECTRUM"].header["ANCRFILE"] = os.path.basename(cur_arf_path)

        del speco["SPECTRUM"].header["BACKFILE"]
        speco["SPECTRUM"].header["BACKFILE"] = os.path.basename(cur_bsp_path)
```

```{caution}
We add the RESPFILE, ANCRFILE, and BACKFILE keywords _after_ grouping because some
FITS file modifications (such as using `ftgrouppha`) can add '&' characters to the end
of long strings in FITS headers. That then causes PyXSPEC to fail to read in supporting
files for the spectrum.

As HEASARC-tutorials demonstrations lean toward using easy-to-read, informative, file
names that are by necessity quite long, we add the correct paths using the Astropy
fits module.
```

## 5. Fitting spectral models using PyXSPEC

Having gone to the trouble of generating ROSAT All-Sky Survey spectra for our sample
of M dwarfs, we'll now fit some models and try to extract some properties!

Using the Python interface (PyXSPEC) to the ubiquitous XSPEC model fitting
software, we will:
1. Fit both power-law and blackbody models to each spectrum.
2. Calculate and store the model fluxes.
3. Extract and store the model parameters and uncertainties.
4. Prepare to create visualizations of the fitted spectra.

Note that this demonstration was not written by an expert in the X-ray
emission of M dwarfs (or indeed any kind of star), so please don't necessarily
take these models as a recommendation for your own work!

### Setting up PyXSPEC

Firstly, we configure how PyXSPEC is going to behave. This includes:
- Setting the 'chatter' to zero, to minimize the outputs that XSPEC prints. **Note that, at the time of writing, there are some PyXSPEC outputs that cannot be suppressed.**
- Telling PyXSPEC to use the Cash statistic; generally considered a good choice for low-count spectra.
- Making sure that PyXSPEC won't ask us for input at any point (`xs.Fit.query = "no"`).

```{code-cell} python
# The strange comment on the end of this line is for the benefit of our
#  automated code-checking processes. You shouldn't import modules anywhere but
#  the top of your file, but this is unfortunately necessary at the moment
import xspec as xs  # noqa: E402

# Limits the amount of output from XSPEC that PyXspec will display
xs.Xset.chatter = 0

# Other xspec settings
xs.Plot.area = True
xs.Plot.xAxis = "keV"
xs.Plot.background = True
xs.Fit.statMethod = "cstat"
xs.Fit.query = "no"
xs.Fit.nIterations = 500
```

We also define a variable that will control whether warnings of our own creation are
displayed; see the code cell in [the next section](#loading-spectra-and-fitting-models). By default, we will _not_
display those warnings, but the possible problems they describe will still be
dealt with:

```{code-cell} python
show_warn = False
```

### Loading spectra and fitting models

This section contains the entirety of our interaction with PyXSPEC in this
notebook; the slightly ugly for loop below will load each spectrum individually,
restrict the energy range, fit models, and create the visualization data (though it
won't plot it).

What we're doing here represents a fairly simple use of PyXSPEC; some of our other
demonstrations, such as '{doc}`Getting started with Swift-XRT <../swift/getting-started-swift-xrt>`,
contain more complex examples; the simultaneous fitting of a model to multiple spectra, for instance.

The most important steps are:
1. Once a spectrum is loaded, we restrict our analysis to data points between 0.11–2.02 keV, also excluding any marked as 'bad' by `ftgrouppha`.
2. Plotting information for the **data** is then generated and stored for later.
3. We move on to model fitting only if $>2$ channels are valid (very low SNR spectra may have one or two); having the same number of channels (or fewer) as there are model parameters would mean an invalid fit.
4. Looping through models (_power law_ and _blackbody_ in this case), they are fit to the data (**using default starting parameter values**), parameter errors and then model fluxes are calculated, and the results are stored in dictionaries.
5. Plotting information for the **models** is generated and stored for later.
6. Finally, the dictionaries of model parameters, uncertainties, and fluxes for each source are combined into Pandas DataFrames, for easier visualization, interaction, and saving.

The low SNR of some of these spectra will almost inevitably lead to poor fits in some
cases, or cause trouble calculating model parameter uncertainties. We keep an eye out
for the latter case especially, as we can use a string returned by XSPEC's `error`
command to check if it has flagged any potential problems with the uncertainties.

We can display a warning when this happens (the visibility of which is controlled
by the `show_warn` variable defined in the [last section](#setting-up-pyxspec)), but
we always write a boolean value to the storage dictionaries indicating if there were
problems with a particular model parameter's uncertainty calculation.

```{code-cell} python
---
tags: [hide-output]
jupyter:
  output_hidden: true
---
spec_plot_data = {}
fit_plot_data = {}
fit_parameters = {}
fit_fluxes = {}

# Iterating through all the ObsIDs
with tqdm(desc="PyXspec - loading RASS spectra", total=len(grp_spec_paths)) as onwards:
    for gsp_ind, cur_name_spec in enumerate(grp_spec_paths.items()):
        cur_src_name, cur_grp_spec = cur_name_spec

        xs.AllData.clear()
        xs.AllModels.clear()

        with contextlib.chdir(os.path.dirname(cur_grp_spec)):
            xs.AllData(cur_grp_spec)
            spec = xs.AllData(1)

        spec.ignore("**-0.11 2.02-**")
        xs.AllData.ignore("bad")

        num_chan_noticed = len(xs.AllData(1).noticed)

        xs.Plot()
        spec_plot_data[cur_src_name] = [
            xs.Plot.x(1),
            xs.Plot.xErr(1),
            xs.Plot.y(1),
            xs.Plot.yErr(1),
        ]

        fit_parameters.setdefault(cur_src_name, {})
        fit_fluxes.setdefault(cur_src_name, {})
        if num_chan_noticed > 2:
            fit_plot_data.setdefault(cur_src_name, {})

            for cur_model_name in ["bbody", "powerlaw"]:

                xs.Model(cur_model_name)
                xs.Fit.perform()

                xs.Plot()
                fit_plot_data[cur_src_name][cur_model_name] = xs.Plot.model(1)

                xs.Fit.error("2.706 1 2")

                xs.AllModels.calcFlux("0.5 2.0 err")
                en_fl, en_fl_min, en_fl_max, ph_fl, ph_fl_min, ph_fl_max = spec.flux

                fit_fluxes[cur_src_name].update(
                    {
                        f"{cur_model_name}_0.5-2.0keV_flux": en_fl,
                        f"{cur_model_name}_0.5-2.0keV_flux_err-": en_fl_min,
                        f"{cur_model_name}_0.5-2.0keV_flux_err+": en_fl_max,
                    }
                )

                for cur_par_id in range(1, xs.AllModels(1).nParameters + 1):
                    cur_par_name = xs.AllModels(1)(cur_par_id).name

                    cur_par_val = xs.AllModels(1)(cur_par_id).values[0]
                    cur_par_lims_out = xs.AllModels(1)(cur_par_id).error

                    # Check the error string output by XSPEC's error command and
                    #  show a warning if there might be a problem
                    error_good = True
                    if cur_par_lims_out[2] != "FFFFFFFFF":
                        if show_warn:
                            warn(
                                f"Error calculation for the {cur_par_name} parameter "
                                f"of {cur_model_name} indicated a possible problem "
                                f"({cur_par_lims_out[2]}) [{cur_src_name}]",
                                stacklevel=2,
                            )
                        error_good = False

                    fit_parameters[cur_src_name].update(
                        {
                            f"{cur_model_name}_{cur_par_name}": cur_par_val,
                            f"{cur_model_name}_{cur_par_name}_err-": cur_par_val
                            - cur_par_lims_out[0],
                            f"{cur_model_name}_{cur_par_name}_err+": cur_par_lims_out[1]
                            - cur_par_val,
                            f"{cur_model_name}_{cur_par_name}_good_err": error_good,
                        }
                    )

        onwards.update(1)

fit_parameters = pd.DataFrame.from_dict(fit_parameters, orient="index")
fit_fluxes = pd.DataFrame.from_dict(fit_fluxes, orient="index")
```

```{note}
We ignore any channels that are outside the 0.11-2.02 keV energy range, which was
chosen using advice from the [ROSAT-PSPC energy calibration table](https://heasarc.gsfc.nasa.gov/docs/rosat/faqs/pspc_cal_faq1.html). Any channels
that have been marked as 'bad' by `ftgrouppha` are also excluded.
```

### Visualizing fitted spectra

At this stage we have fitted two models to most of the spectra (those with $<3$ valid
channels were excluded). We now might want to take a look at them plotted on top of
the spectrum they were fitted to (we also have the necessary data to plot those spectra
with no model fits).

We set up a many-panel figure to display the fitted, background-subtracted, spectrum
for each of our M dwarfs. The x-axis energy scales are consistent across all panels,
and the y-axis scale is consistent across _rows_.

Here we set up the colors assigned to each model:

```{code-cell} python
nice_model_names = {"bbody": "Blackbody", "powerlaw": "Power-law"}
nice_model_colors = {"bbody": "firebrick", "powerlaw": "teal"}
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
num_sps = len(grp_spec_paths)
num_cols = 2
num_rows = int(np.ceil(num_sps / num_cols))

fig_side_size = 5
width_multi = 1.4

fig, ax_arr = plt.subplots(
    ncols=num_cols,
    nrows=num_rows,
    figsize=((fig_side_size * width_multi) * num_cols, fig_side_size * num_rows),
    sharey="row",
    sharex=True,
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

ax_ind = 0
for ax_arr_ind, ax in np.ndenumerate(ax_arr):
    if ax_ind >= num_sps:
        ax.set_visible(False)
        continue

    cur_src_name, cur_seq_id = list(src_seq_ids.items())[ax_ind]
    cur_actual_name = id_name_to_actual[cur_src_name]

    cur_sp_data = spec_plot_data[cur_src_name]

    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="minor", labelsize=8)

    ax.errorbar(
        cur_sp_data[0],
        cur_sp_data[2],
        xerr=cur_sp_data[1],
        yerr=cur_sp_data[3],
        fmt="kx",
        capsize=1.5,
        label="ROSAT All-Sky Data",
        lw=0.6,
        alpha=0.7,
    )

    if cur_src_name in fit_plot_data:
        for cur_model_name, cur_fit_data in fit_plot_data[cur_src_name].items():
            ax.plot(
                cur_sp_data[0],
                cur_fit_data,
                label="XSPEC " + nice_model_names[cur_model_name] + " fit",
                color=nice_model_colors[cur_model_name],
            )

    ax.legend(loc="upper right")

    ax.set_xlim(0.098, 2.08)
    ax.set_xscale("log")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

    ax.set_xlabel("Energy [keV]", fontsize=15)
    if ax_arr_ind[1] == 0:
        ax.set_ylabel(r"Spectrum [ct cm$^{-2}$ s$^{-1}$ keV$^{-1}$]", fontsize=15)

    names_title = f"{cur_src_name}/\n{cur_actual_name}"
    ax.set_title(
        names_title,
        y=0.8,
        x=0.02,
        fontsize=15,
        color="dimgrey",
        fontweight="bold",
        horizontalalignment="left",
    )

    ax_ind += 1

plt.show()
```

### Saving PyXSPEC fit results

If you're fitting X-ray spectra as part of a research project, you're probably going to
want to save the properties you derived to a file, so you can use them later without
re-running the analysis.

In the section where we
[loaded and fit the spectra](#loading-spectra-and-fitting-models), we mentioned
converting the parameter storage dictionaries into Pandas DataFrame objects, one for
fluxes (`fit_fluxes`) and another for model parameters and
uncertainties (`fit_parameters`).

Here we combine them into a single dataframe by performing a table merge, matching the
dataframe indexes (which, due to the way we created the dataframes, are the
"CARMENES-{ID}" style names we [assigned to each source earlier](#preparing-the-carmenes-catalog-for-upload)).

```{code-cell} python
fit_parameters = fit_parameters.round(5)
fit_fluxes = fit_fluxes.round(16)
rass_results = pd.merge(fit_parameters, fit_fluxes, left_index=True, right_index=True)

rass_results = pd.merge(
    carm_cat.to_pandas()[["Karmn", "Name", "id_name"]],
    rass_results,
    right_index=True,
    left_on="id_name",
)

rass_results = rass_results.set_index("id_name")
```

The combined dataframe is then saved as a comma-separated values (CSV) file:

```{code-cell} python
# Using a convenience method of the Pandas DataFrame class
rass_results.to_csv("carmenes_mdwarf_rass_properties.csv", index=True)
```

We can also take a peek at the first few rows, to see what information it contains:

```{code-cell} python
rass_results.head(6)
```

### Briefly examining the fit results

The dataframe of fit results can very easily be leveraged to examine the X-ray
property distributions of the CARMENES M dwarf sample. One possible place to start
is to examine the distributions of fitted model parameter values.

We've not taken a very rigorous approach to fitting and extracting these parameters; if
you're working on a 'real' project, you might want to take more care. That could
include:
- Setting reasonable starting values for the model parameters in the [fitting section](#loading-spectra-and-fitting-models).
- Extracting, saving, and examining goodness-of-fit information for each fit.
- Checking the background spectrum regions for contaminating sources [when defining them](#setting-up-astropy-regions-representing-source-and-background-regions).

Here all we're going to do is check for obviously unphysical parameter values, as well as
excluding any that were flagged as having potentially problematic uncertainties at
the fitting stage.

#### Blackbody temperature

The blackbody temperatures of our M dwarfs might be quite interesting, so we'll make
a quick histogram to check them out!

To hopefully ensure we only include values from successful fits, we're going
to filter our dataframe.

We're first going to make a boolean numpy array (`sel_posi`) that we will use as a
mask to select only the rows where both the blackbody temperature, and both of its
uncertainties, are positive (to avoid any unphysical results).

The `> 0` check on the three columns we retrieve from the dataframe will produce a
new dataframe with those same column names but the values will have been replaced by
True or False, depending on whether the entry was positive or not.

That information is retrieved as a numpy array (using the `.values` property), and the
`.all(axis=1)` call will return a 1D array of booleans, one entry per row, indicating
whether all three column values were positive.

We then use the already-boolean "bbody_kT_good_err" column as another mask, to make
sure that we don't include any data points that had possibly problematic uncertainties:

```{code-cell} python
res_bbtx_cut = rass_results.copy()

sel_posi = (
    res_bbtx_cut[["bbody_kT", "bbody_kT_err-", "bbody_kT_err+"]] > 0
).values.all(axis=1)
res_bbtx_cut = res_bbtx_cut[sel_posi]

res_bbtx_cut = res_bbtx_cut[res_bbtx_cut["bbody_kT_good_err"]]
res_bbtx_cut.info()
```

Now we can use that filtered dataframe to make a histogram of blackbody temperature,
which shows that most of the M dwarfs (or at least their hot outer atmospheres) have a
temperature of around 0.15 keV (~1.7 million Kelvin).

That is consistent with the range of temperatures considered for M dwarfs stars in a
recent census of local M dwarfs by
[Caramazza M. et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...676A..14C/abstract).

There may also be another collection of M dwarfs with slightly (but only slightly)
cooler outer atmospheres, peaking around 0.12 keV (~1.4 million Kelvin):

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Create a histogram of the blackbody temperature distribution
kt_bins = np.arange(0, res_bbtx_cut["bbody_kT"].max() * 1.1, 0.0125)

plt.figure(figsize=(6, 6))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.hist(
    res_bbtx_cut["bbody_kT"],
    bins=kt_bins,
    histtype="stepfilled",
    fc="seagreen",
    ec="black",
)

plt.ylabel("N", fontsize=15)
plt.xlabel(r"Blackbody $T_{\rm{X}}$ [keV]", fontsize=15)

plt.tight_layout()
plt.show()
```

#### Power-law index

We repeat the same exercise as in [the last section](#blackbody-temperature), this time
for the power-law index parameter.

Unfortunately, it's a little harder to define an 'unphysical' index value than it
was to define an unphysical temperature.

The maximum value of an XSPEC power law photon index (by default) is 10. If a
value is in that territory, it's _usually_ (though not guaranteed to be) the product
of a bad fit, so we exclude any rows where the index is $>9.5$.

We also exclude rows where either of the power-law index uncertainties are
negative, or where the uncertainty was flagged as potentially bad at the fitting stage:

```{code-cell} python
res_plind_cut = rass_results.copy()

sel_good = (res_plind_cut["powerlaw_PhoIndex"] < 9.5).values

sel_err_posi = (
    res_plind_cut[["powerlaw_PhoIndex_err-", "powerlaw_PhoIndex_err+"]] > 0
).values.all(axis=1)

res_plind_cut = res_plind_cut[sel_good & sel_err_posi]

res_plind_cut = res_plind_cut[res_plind_cut["powerlaw_PhoIndex_good_err"]]

res_plind_cut.info()
```

Finally, we make the histogram:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Create a histogram of the power law photon index distribution
pho_bins = np.arange(
    res_plind_cut["powerlaw_PhoIndex"].min() * 0.9,
    res_plind_cut["powerlaw_PhoIndex"].max() * 1.1,
    0.1,
)

plt.figure(figsize=(6, 6))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.hist(
    rass_results["powerlaw_PhoIndex"], bins=pho_bins, histtype="step", ec="peru", lw=1.8
)

plt.ylabel("N", fontsize=15)
plt.xlabel(r"Power-law Photon Index", fontsize=15)

plt.tight_layout()
plt.show()
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Author: Mike Corcoran, Associate Research Professor

Updated On: 2026-03-12

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[PyVO GitHub Repository](https://github.com/astropy/pyvo)

[Latest PyVO Documentation](https://pyvo.readthedocs.io/en/latest/)

[ROSAT-PSPC energy calibration table](https://heasarc.gsfc.nasa.gov/docs/rosat/faqs/pspc_cal_faq1.html)

[Astropy `regions` GitHub](https://github.com/astropy/regions)

[Astropy `regions` documentation](https://astropy-regions.readthedocs.io/en/stable/)

[HEASoft quzcif help](https://heasarc.gsfc.nasa.gov/docs/software/ftools/caldb/quzcif.html)

[HEASARC Calibration Database (CALDB) introduction](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_intro.html)

### Acknowledgements


### References

[Alonso-Floriano F. J., Morales J. C., Caballero J. A., Montes D. et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015A%26A...577A.128A/abstract) - _CARMENES input catalogue of M dwarfs. I. Low-resolution spectroscopy with CAFOS_

[Boller T., Freyberg M.J., Trümper J. et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract) - _Second ROSAT all-sky survey (2RXS) source catalogue_

[The VizieR service DOI: 10.26093/cds/vizier](https://doi.org/10.26093/cds/vizier)

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) - _astroquery: An Astronomical Web-querying Package in Python_

[Belloni T., Hasinger G., Izzo C. (1994)](https://ui.adsabs.harvard.edu/abs/1994A%26A...283.1037B/abstract) - _Procedures for the interactive analysis of point sources from the ROSAT XRT/PSPC all-sky survey_

[Caramazza M., Stelzer B., Magaudda E., Raetz St., Güdel M., Orlando S., Poppenhäger K. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...676A..14C/abstract) - _Complete X-ray census of M dwarfs in the solar neighborhood. I. GJ 745 AB: Coronal-hole stars in the 10 pc sample_
