---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: "Anna Ogorza\u0142ek"
  affiliations: ['University of Maryland, College Park', 'XRISM GOF, NASA Goddard']
  website: https://www.astro.umd.edu/people/anna-ogorzalek
  orcid: 0000-0003-4504-2557
date: '2026-06-11'
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
title: Getting started with XRISM-Resolve
---

# Getting started with XRISM-Resolve

## Learning Goals

By the end of this tutorial, you will be able to:

- Search for and acquire XRISM observations of a target of interest.
- Process the observation data to a science ready state.
- Make informed decisions on which XRISM-Resolve events are best for your science case.
- Generate XRISM-Resolve data products (images, exposure maps, spectra and supporting files).
- Fit a simple PyXspec model to a XRISM-Resolve spectrum.

## Introduction

The 'X-Ray Imaging and Spectroscopy Mission' (**XRISM**) is an X-ray telescope
designed for high-energy-resolution spectroscopic observations of astrophysical
sources, as well as wide-field X-ray imaging.

XRISM, launched in 2023, is the result of a JAXA-NASA partnership (with involvement
from ESA), and serves as a nearly like-for-like replacement of the **Hitomi**
telescope, which was lost shortly after its launch in 2016.

There are two main XRISM instruments, **Resolve** and **Xtend**. In this
tutorial, we will focus on **Resolve**, which is a completely unique high-energy
resolution spectroscopic X-ray microcalorimeter instrument, capable of spatially
resolved observations (though admittedly at low angular resolution).

The other instrument, **Xtend**, has its own dedicated demonstration notebook.

Our goal with this 'getting started' notebook is to give you the skills required
to prepare XRISM-Resolve observations for scientific use and to generate data
products tailored to your science goals. It can also serve as a template notebook
that you can use as a foundation to build your own analyses.

We make use of the HEASoftPy interface to HEASoft tasks throughout this demonstration.

### Inputs

- The name of the source of interest, in this case *NGC 1365*.

### Outputs

- Processed XRISM-Resolve event lists.
- Figures illustrating different properties of, or issues with, XRISM-Resolve data.
- New data products:
  - Images and exposure maps in specified energy ranges.
  - Simple region files defining which XRISM pixels should be extracted from.
  - XRISM-Resolve spectra, ancillary response files (ARFs) and response matrix files (RMFs).
- XSPEC model fit results:
  - Parameter values measured for a simple model fit to the new spectra.
  - Visualizations of the fitted spectrum.

### Runtime

As of 13th May 2026, this notebook takes ~45-minutes to run to completion on Fornax using the 'Default Astrophysics' image and the small server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os
from random import randint
from shutil import rmtree
from typing import List, Optional, Tuple, Union
from warnings import warn

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from packaging.version import Version
from xga.products import EventList, Image
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def process_xrism_resolve(
    obs_dir: str,
    cur_obs_id: str,
    out_dir: str,
    file_stem: str,
    rslmpcor_caldb: str = None,
):
    """

    :param str obs_dir:
    :param str cur_obs_id: The ObsID of the XRISM observation to be processed.
    :param str out_dir: The directory where output files should be written.
    :param str file_stem:
    :param str rslmpcor_caldb:
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

    # Ensure that the observation directory passed by the user is absolute before
    #  we start changing directories.
    # Once we use the chdir context to switch directories during processing, we'll
    #  retrieve a relative path to limit the number of characters in the string
    #  passed to xapipeline (long paths can sometimes cause problems)
    obs_dir = os.path.abspath(obs_dir)

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xapipeline_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Check whether a path to CALDB file that the 'rslmpcor' HEASoft task requires
    #  has been passed. This is necessary because HEASoft v6.36 and below have a bug
    #  where the 'rslmpcor' task is not compatible with the remote CALDB, so for this
    #  demonstration to work we have to download it.
    if rslmpcor_caldb is not None:
        rslmpcor_caldb = os.path.abspath(rslmpcor_caldb)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():

        # The processing/preparation stage of any X-ray telescope's data is the most
        #  likely to go wrong, and we use a Python try-except as an automated way to
        #  collect ObsIDs that had an issue during processing.
        try:
            out = hsp.xapipeline(
                instrument="RESOLVE",
                indir=os.path.relpath(obs_dir),
                outdir=".",
                entry_stage=1,
                exit_stage=2,
                steminputs=file_stem,
                stemoutputs=file_stem,
                rsl_mpcorfile=(
                    "NONE"
                    if rslmpcor_caldb is None
                    else os.path.relpath(rslmpcor_caldb)
                ),
                clobber=True,
                chatter=TASK_CHATTER,
                noprompt=True,
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


def screen_xrism_resolve_evts(
    event_file: str,
    out_dir: str,
    lo_pi: Optional[Union[int, Quantity]] = 600,
    hi_pi: Optional[Union[int, Quantity]] = None,
    status4_screen: bool = True,
    rise_time_screen: bool = True,
    unfilt_elec_coinc_evt_screen: bool = False,
    exclude_pix27: bool = True,
):
    """
    This function wraps the HEASoftPy interface to the generic ftcopy task, which is
    used to make a copy of the input events list file that contains a subset of events
    which pass the screening criteria specified by the other arguments of this function.

    We note that this function does not apply event grade screening (e.g. just selecting
    high-resolution primary events), as grade-screened event lists are not recommended
    for all product generation purposes (e.g. making images) - grade screening can
    be applied at the time of product generation.

    :param str event_file:
    :param str out_dir:
    :param int/Quantity lo_pi:
    :param int/Quantity hi_pi:
    :param bool status4_screen:
    :param bool rise_time_screen:
    :param bool unfilt_elec_coinc_evt_screen:
    :param bool exclude_pix27: Whether Pixel 27 should be excluded (currently exhibits
        gain problems and is not recommended for scientific use). Default is True.
    """

    # We can extract the ObsID directly from the header of the event list - it is
    #  safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        # We extract the filter value from the header, then pass it through a
        #  dictionary defined in the constants section to convert it to the
        #  format you see in XRISM file names.
        cur_filter = RESOLVE_FILTERS[read_evto["EVENTS"].header["FILTER"]]

    # NOTES TO SELF
    # elec_coinc_evt_screen = True means screening on STATUS[13] - should
    #  recommend for 'bright' sources
    # DO NOT CURRENTLY UNDERSTAND THE RECOMMENDED RISE TIME CUTS - THEY AREN'T
    #  EXPLAINED IN THE ABC THAT I CAN SEE...

    # Checks the validity of the input lower PI channel value, and converts
    #  to energy values.
    if isinstance(lo_pi, int):
        pass
    elif isinstance(lo_pi, Quantity) and lo_pi.unit.is_equivalent("chan"):
        lo_pi = lo_pi.to("chan").astype(int).value
    elif isinstance(lo_pi, Quantity) and lo_pi.unit.is_equivalent("eV"):
        lo_pi = (lo_pi.to("eV") / RSL_EV_PER_CHAN).to("chan").astype(int).value
    elif lo_pi is not None:
        raise ValueError(
            "If set, 'lo_pi' must be an integer PI value, or an Astropy quantity "
            "convertible to 'eV' - otherwise it must be None."
        )
    # Checks the validity of the input upper PI channel value, and converts
    #  to energy values.
    if isinstance(hi_pi, int):
        pass
    elif isinstance(hi_pi, Quantity) and hi_pi.unit.is_equivalent("chan"):
        hi_pi = hi_pi.to("chan").astype(int).value
    elif isinstance(hi_pi, Quantity) and hi_pi.unit.is_equivalent("eV"):
        hi_pi = (hi_pi.to("eV") / RSL_EV_PER_CHAN).to("chan").astype(int).value
    elif hi_pi is not None:
        raise ValueError(
            "If set, 'hi_pi' must be an integer PI value, or an Astropy quantity "
            "convertible to 'eV' - otherwise it must be None."
        )

    # We'll append filter expressions to this list
    filt_expr = []

    if lo_pi is not None:
        filt_expr.append(f"(PI>={lo_pi})")
    if hi_pi is not None:
        filt_expr.append(f"(PI<={hi_pi})")

    if rise_time_screen:
        filt_expr.append(
            "(((((RISE_TIME+0.00075*DERIV_MAX)>46)&&"
            "((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))"
        )

    if exclude_pix27:
        filt_expr.append("(PIXEL!=27)")

    if status4_screen:
        filt_expr.append("(STATUS[4]==b0)")

    if unfilt_elec_coinc_evt_screen:
        filt_expr.append("(STATUS[13]==b0)")

    evt_file_filt = f"{event_file}[EVENTS][{'&&'.join(filt_expr)}]"

    evt_out = os.path.basename(SCR_EVT_PATH_TEMP).format(
        oi=cur_obs_id,
        xrf=cur_filter,
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "cleaning_evts_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.ftcopy(
            infile=os.path.relpath(evt_file_filt),
            outfile=evt_out,
            copyall=True,
            clobber=True,
            history=True,
            noprompt=True,
            chatter=TASK_CHATTER,
        )

    # Move the event list up from the temporary directory
    os.rename(os.path.join(temp_work_dir, evt_out), os.path.join(out_dir, evt_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_resolve_image(
    event_file: str,
    out_dir: str,
    lo_en: Quantity,
    hi_en: Quantity,
    include_evt_grades: list = None,
    sub_pixel: bool = False,
    im_bin_sub_pixel: int = 1,
):
    """
    This function wraps the HEASoft 'extractor' tool and is used to spatially bin
    XRISM-Resolve event lists into images. The HEASoftPy interface to 'extractor'
    is used.

    The ObsID and X-ray filter are extracted from the header of the passed event
    list file.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate an image from. ObsID and dataclass information
        will be extracted from the EVENTS table header.
    :param str out_dir: The directory where output files should be written.
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the image.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the image.
    :param List[int] include_evt_grades:
    :param bool sub_pixel: If False (default), then the output image pixels match
        will match the Resolve's array's pixels. If True, then the output image
        will be generated using the XY coordinates, and binned according to the
        'im_bin_sub_pixel' argument, over-sampling the instruments spatial resolution.
    :param int im_bin_sub_pixel: Number of XRISM-Resolve SKY X-Y coordinate system
        'pixels' to bin into a single image pixel. Only used if 'sub_pixel' is True.
    """

    # We can extract the ObsID directly from the header of the event list - it is
    #  safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        # We extract the filter value from the header, then pass it through a
        #  dictionary defined in the constants section to convert it to the
        #  format you see in XRISM file names.
        cur_filter = RESOLVE_FILTERS[read_evto["EVENTS"].header["FILTER"]]

    # Make sure the lower and upper energy limits make sense
    if lo_en > hi_en:
        raise ValueError(
            "The lower energy limit must be less than or equal to the upper "
            "energy limit."
        )
    else:
        lo_en_val = lo_en.to("keV").value
        hi_en_val = hi_en.to("keV").value

    # The image binning factor will depend on how we're generating these images
    if not sub_pixel:
        im_bin = 1
        im_bin_name = "PIX"
        bin_coord_sys = "DET"
    else:
        im_bin = im_bin_sub_pixel
        im_bin_name = im_bin
        bin_coord_sys = ""

    # The default behavior is to use all event grades
    if include_evt_grades is None:
        include_evt_grades = [0, 1, 2, 3, 4]

    # Normalize the input of event grades to be included
    if isinstance(include_evt_grades, int):
        include_evt_grades = [include_evt_grades]
    elif isinstance(include_evt_grades, list):
        include_evt_grades = [int(cur_gr) for cur_gr in include_evt_grades]
    else:
        raise TypeError(
            "The 'include_evt_grades' argument must be a list of integer "
            "ITYPE event grades."
        )

    # Convert the energy limits to channel limits, rounding down and up to the nearest
    #  integer channel for the lower and upper bounds respectively.
    lo_ch = np.floor((lo_en / RSL_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / RSL_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the image we're about to generate.
    im_out = os.path.basename(IM_PATH_TEMP).format(
        oi=cur_obs_id, xrf=cur_filter, ibf=im_bin_name, lo=lo_en_val, hi=hi_en_val
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
            xcolf=bin_coord_sys + "X",
            ycolf=bin_coord_sys + "Y",
            gcol="ITYPE",
            gstring=",".join(np.array(include_evt_grades).astype(str)),
            gti="GTI",
            chatter=TASK_CHATTER,
        )

    # Move the output image file to the proper output directory from
    #  the temporary working directory
    os.rename(os.path.join(temp_work_dir, im_out), os.path.join(out_dir, im_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_resolve_expmap(
    event_file: str,
    out_dir: str,
    pix_gti_file: str = None,
    radial_delta: Union[float, Quantity] = Quantity(20.0, "arcmin"),
    num_phi_bin: int = 1,
):
    """
    Function that wraps the HEASoftPy interface to the XRISM-Resolve 'xaexpmap'
    task, which is used to generate exposure maps for XRISM-Resolve observations.

    :param str event_file: Event list of the observation + dataclass you wish to
        generate an exposure map for. No event data are used in the creation of the
        event list, but some information in the file headers is useful.
    :param str out_dir: The directory where output files should be written.
    :param str pix_gti_file:
    :param float/Quantity radial_delta: Radial increment for the annular grid for
        which the attitude histogram will be calculated.
    :param int num_phi_bin: Number of azimuth (phi) bins in the first annular region
        over which attitude histogram bins will be calculated
    """

    # We can extract the ObsID directly from the header of the event list - it is
    #  safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        # We extract the filter value from the header, then pass it through a
        #  dictionary defined in the constants section to convert it to the
        #  format you see in XRISM file names.
        cur_filter = RESOLVE_FILTERS[read_evto["EVENTS"].header["FILTER"]]

    ext_hk_file = os.path.join(
        ROOT_DATA_DIR, cur_obs_id, "auxil", f"xa{cur_obs_id}.ehk.gz"
    )
    gti_file = event_file

    if pix_gti_file is None:
        pix_gti_file = os.path.join(out_dir, f"xa{cur_obs_id}rsl_{cur_filter}_exp.gti")
    else:
        pix_gti_file = os.path.abspath(pix_gti_file)

    # Make sure the radial_delta value is in arcminutes/is convertible to arcmins
    #  Also will assume that radial_delta is in arcmin if it is not a Quantity object
    if not isinstance(radial_delta, Quantity):
        radial_delta = Quantity(radial_delta, "arcmin")
    elif radial_delta.unit.is_equivalent("arcmin"):
        radial_delta = radial_delta.to("arcmin")
    else:
        raise ValueError(
            f"The 'radial_delta' argument must be in arcmin or convertible to "
            f"arcmin, not {radial_delta.unit}."
        )

    # Now we're certain of 'radial_delta's unit, we read out the value
    radial_delta = radial_delta.value.astype(float)

    # Two variants of exposure map can be generated by the function we're about to
    #  call; the default is a map of the integrated exposure time for each pixel, and
    #  the second (not recommended by the documentation) is a flat-fielding map
    out_map_type = "EXPOSURE"

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaexpmap_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Set up the output file name for the exposure map we're about to generate.
    ex_out = os.path.basename(EX_PATH_TEMP).format(
        oi=cur_obs_id, xrf=cur_filter, rd=radial_delta, npb=num_phi_bin
    )

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():

        out = hsp.xaexpmap(
            instrume="RESOLVE",
            ehkfile=ext_hk_file,
            gtifile=gti_file,
            pixgtifile=pix_gti_file,
            delta=radial_delta,
            numphi=num_phi_bin,
            outfile=ex_out,
            outmaptype=out_map_type,
            badimgfile="NONE",
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

    # Move the up to the final output directory
    os.rename(os.path.join(temp_work_dir, ex_out), os.path.join(out_dir, ex_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_resolve_spectrum(
    event_file: str,
    out_dir: str,
    include_evt_grades: list = [0, 1, 2, 3, 4],
    include_pixels: list = None,
    exclude_pixel_27: bool = True,
):
    """


    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate a XRISM-Resolve spectrum from. ObsID and
        dataclass information will be extracted from the EVENTS table header.
    :param str out_dir: The directory where output files should be written.
    """

    # We can extract the ObsID directly from the header of the event list - it is
    #  safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        # We extract the filter value from the header, then pass it through a
        #  dictionary defined in the constants section to convert it to the
        #  format you see in XRISM file names.
        cur_filter = RESOLVE_FILTERS[read_evto["EVENTS"].header["FILTER"]]

    # Normalize the input of event grades to be included
    if isinstance(include_evt_grades, int):
        include_evt_grades = [include_evt_grades]
    elif isinstance(include_evt_grades, list):
        include_evt_grades = [int(cur_gr) for cur_gr in include_evt_grades]
    else:
        raise TypeError(
            "The 'include_evt_grades' argument must be a list of integer "
            "ITYPE event grades."
        )

    # Also make the selected grades into a string for the output file name
    include_evt_grades_str = "_".join(
        [RESOLVE_EVT_GRADES_INT_SHORT[cur_gr] for cur_gr in include_evt_grades]
    )

    # If the special-case of ignoring pixel 27 is activated, we modify the
    #  include_pixels variable. Either from list form or from the all-inclusive None
    #  INTO a list of all pixels except 27.
    if exclude_pixel_27 and include_pixels is not None:
        include_pixels = [p for p in include_pixels if p != 27]
    elif exclude_pixel_27 and include_pixels is None:
        include_pixels = [p for p in range(0, 36) if p != 27]

    if include_pixels is None:
        include_pixels = "0:11,13:35"
    else:
        # Convert to a set of pixel ranges, as requested in the XRISM rslmkrmf docs
        groups = np.split(
            u := np.unique(include_pixels), np.where(np.diff(u) > 1)[0] + 1
        )
        include_pixels = ",".join(
            f"{g[0]}:{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups
        )

    filt_expr = f"[PIXEL={include_pixels}]"

    # Set up the output file names for the source spectrum we're about to generate.
    sp_out = os.path.basename(SP_PATH_TEMP).format(
        oi=cur_obs_id,
        xrf=cur_filter,
        slp=include_pixels.replace(":", "to").replace(",", "_"),
        grd=include_evt_grades_str,
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
        src_out = hsp.extractor(
            filename=os.path.relpath(event_file) + filt_expr,
            phafile=sp_out,
            ecol="PI",
            gti="GTI",
            gcol="ITYPE",
            gstring=",".join(np.array(include_evt_grades).astype(str)),
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

        # Add DSTYP and DSVAL header keys to the spectrum file describing the
        #  pixel ranges - this is convenient for our implementation of the
        #  RMF generation wrapper function.
        with fits.open(sp_out, mode="update") as new_speco:
            spec_hdr = new_speco["SPECTRUM"].header

            # First find the next available DSTYP/DSVAL index to avoid overwriting
            #  existing entries
            dstyp_ind = 1
            while f"DSTYP{dstyp_ind}" in spec_hdr:
                dstyp_ind += 1

            # Convert the pixel ranges back to a string format for the header
            #  include_pixels is already in the format used for the FITS filter
            #  expression (e.g., "0:10,15:20,27" or "0:35")
            spec_hdr[f"DSTYP{dstyp_ind}"] = "PIXEL"
            spec_hdr[f"DSVAL{dstyp_ind}"] = include_pixels

    # Move the spectra up from the temporary directory
    os.rename(os.path.join(temp_work_dir, sp_out), os.path.join(out_dir, sp_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return src_out, os.path.join(out_dir, sp_out), event_file, cur_obs_id, cur_filter


def gen_xrism_resolve_rmf(
    event_file: str,
    spec_file: str,
    out_dir: str,
    rmf_type: str = "L",
    include_evt_grades: List[int] = None,
    include_pixels: List[int] = None,
):
    """
    A wrapper around the XRISM-Resolve-specific RMF generation tool implemented as
    part of HEASoft (and called here through HEASoftPy).

    :param str spec_file: The path to the spectrum file for which to generate an RMF.
    :param str out_dir: The directory where output files should be written.
    :param List[int] rel_pixels:
    """

    if rmf_type.lower() == "x":
        raise NotImplementedError(
            "This convenience function does not currently support the generation "
            "of X-large RMFs."
        )

    #
    sp_include_evt_grades = None
    sp_include_pixels = None

    with fits.open(spec_file) as read_speco:
        ds_mask = np.array(
            ["DSTYP" in hdr_key for hdr_key in read_speco["SPECTRUM"].header]
        )

        dtyp_hdrs = np.array(list(read_speco["SPECTRUM"].header.keys()))[ds_mask]

        dtype_hdr_ens = {
            read_speco["SPECTRUM"].header[rel_key]: rel_key for rel_key in dtyp_hdrs
        }

        if "ITYPE" in dtype_hdr_ens:
            sp_grade_ranges = read_speco["SPECTRUM"].header[
                dtype_hdr_ens["ITYPE"].replace("DSTYP", "DSVAL")
            ]

            sp_grade_ranges = [
                np.array(list(set(ran_pair.split(":")))).astype(int)
                for ran_pair in sp_grade_ranges.split(",")
            ]
            sp_include_evt_grades = np.concat(
                [
                    (
                        cur_gr
                        if len(cur_gr) == 1
                        else list(range(cur_gr[0], cur_gr[1] + 1))
                    )
                    for cur_gr in sp_grade_ranges
                ]
            ).astype(int)
            sp_include_evt_grades = sp_include_evt_grades.tolist()

        if "PIXEL" in dtype_hdr_ens:
            sp_pix_ranges = read_speco["SPECTRUM"].header[
                dtype_hdr_ens["PIXEL"].replace("DSTYP", "DSVAL")
            ]
            sp_pix_ranges = [
                np.array(list(set(ran_pair.split(":")))).astype(int)
                for ran_pair in sp_pix_ranges.split(",")
            ]
            sp_include_pixels = np.concat(
                [
                    (
                        cur_pr
                        if len(cur_pr) == 1
                        else list(range(cur_pr[0], cur_pr[1] + 1))
                    )
                    for cur_pr in sp_pix_ranges
                ]
            ).astype(int)

            sp_include_pixels = sp_include_pixels.tolist()

    if sp_include_evt_grades is not None and include_evt_grades is not None:
        raise ValueError(
            f"Event grades used to generate the spectrum "
            f"([{sp_include_evt_grades}]) have been inferred from the "
            f"file header, and so the 'include_evt_grades' argument "
            f"should be None."
        )
    elif sp_include_evt_grades is not None:
        include_evt_grades = sp_include_evt_grades

    if sp_include_pixels is not None and include_pixels is not None:
        raise ValueError(
            f"Pixels used to generate the spectrum "
            f"([{sp_include_pixels}]) have been inferred from the "
            f"file header, and so the 'include_pixels' argument "
            f"should be None."
        )
    elif sp_include_pixels is not None:
        include_pixels = sp_include_pixels
    elif include_pixels is None:
        raise ValueError(
            "No pixel information could be identified in the spectrum "
            "file header, so 'include_pixels' cannot be None."
        )

    # Check that the RMF type passed by the user is valid
    if not isinstance(rmf_type, str):
        raise TypeError("The 'rmf_type' argument must be of type string.")
    elif rmf_type not in ["S", "M", "L", "X"]:
        raise ValueError("'rmf_type' must be 'S', 'M', 'L', or 'X'.")

    # Enforce correct types for input grades
    if isinstance(include_evt_grades, str) or (
        isinstance(include_evt_grades, list)
        and any([isinstance(gr, str) for gr in include_evt_grades])
    ):
        try:
            include_evt_grades = [int(gr) for gr in list(include_evt_grades)]
        except ValueError:
            # Error message doesn't exactly match the error we caught, but if the
            #  user just replaces entries with integers this issue will be solved
            raise TypeError(
                "Entries in the 'include_evt_grades' list must be integers."
            )
    elif isinstance(include_evt_grades, int):
        include_evt_grades = [include_evt_grades]
    elif not isinstance(include_evt_grades, list) and any(
        [isinstance(gr, str) for gr in include_evt_grades]
    ):
        raise TypeError("Only pass lists of integers to 'include_evt_grades'.")

    # Now make sure that the input grades are all in the valid range
    include_evt_grades = np.array(include_evt_grades)
    if (include_evt_grades < 0).any() or (include_evt_grades > 4).any():
        raise ValueError(
            "XRISM-Resolve events are assigned an integer grade from "
            "0 to 4, and at least one entry in 'include_evt_grades' is "
            "outside this range."
        )

    # Turn the grades into a string that can be passed to the XRISM task
    include_evt_grades = ",".join(include_evt_grades.astype(str))

    if include_pixels is not None:
        if not isinstance(include_pixels, (list, str, int)):
            raise TypeError("'include_pixels' must be a list of integer pixel IDs.")
        elif not isinstance(include_pixels, list) and isinstance(
            include_pixels, (str, int)
        ):
            include_pixels = [include_pixels]

        try:
            include_pixels = np.array([int(rp) for rp in include_pixels])
        except ValueError:
            raise TypeError(
                "All entries in 'include_pixels' must be integer "
                "XRISM-Resolve pixel IDs."
            )

        # Convert to a set of pixel ranges, as requested in the XRISM rslmkrmf docs
        groups = np.split(
            u := np.unique(include_pixels), np.where(np.diff(u) > 1)[0] + 1
        )
        include_pixels = ",".join(
            f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups
        )

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "rslrmf_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Set up the RMF file name by cannibalising the name of the spectrum file - this
    #  means we don't have to worry about identifying the ObsID
    rmf_out = os.path.basename(spec_file).replace("-spectrum.fits", ".rmf")

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.rslmkrmf(
            infile=os.path.relpath(event_file),
            whichrmf=rmf_type,
            resolist=include_evt_grades,
            regionfile="NONE",
            pixlist=include_pixels,
            outfileroot=rmf_out.split(".")[0],
            noprompt=True,
            clobber=True,
        )

    # Move the RMF up from the temporary directory
    os.rename(os.path.join(temp_work_dir, rmf_out), os.path.join(out_dir, rmf_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out, os.path.join(out_dir, rmf_out)


def gen_xrism_resolve_arf(
    out_dir: str,
    rel_coord: SkyCoord,
    expmap_file: str,
    spec_file: str,
    rmf_file: str,
    pix_reg_file: str,
    num_photons: int,
    min_photons: int,
):
    """
    A wrapper function for the HEASoft `xaarfgen` task, which we use to generate
    ARFs for XRISM-Resolve spectra.

    IMPORTANT: The way we have set up the call to `xaarfgen` implicitly assumes that
    the spectrum was generated for a POINT SOURCE. Using this setup to generate
    an ARF for an extended source WOULD NOT BE VALID.

    This function can take a long time to run, primarily because of the ray-tracing
    step (and the acquisition of a large CalDB file necessary for this step, if
    using remote CalDB). The ray-tracing time will scale with the value
    of 'num_photons', with the XRISM team estimating ~1 minute per 100,000 photons
    (though note this does not include time to download the previously mentioned
    CalDB file).

    :param str out_dir: The directory where output files should be written.
    :param SkyCoord rel_coord:
    :param str pix_reg_file:
    :param str expmap_file: The path to the exposure map file necessary to generate
        the ARF.
    :param str spec_file: The path to the spectrum file for which to generate an ARF.
    :param str rmf_file: The path to the RMF file necessary to generate an ARF.
    :param int num_photons: The number of photons, per energy grid point, per
        attitude histogram, to simulate in the ray-tracing portion of
        XRISM-Resolve ARF generation.
    :param int min_photons: The minimum number of photons, per energy grid point, per
        attitude histogram, that is required to continue to calculating an ARF at
        the end of the ray-tracing portion.
    """

    # We can extract the ObsID directly from the header of the spectrum file - it is
    #  safer than having the user pass it separately
    with fits.open(spec_file) as read_speco:
        cur_obs_id = read_speco[0].header["OBS_ID"]

    pix_reg_file = os.path.abspath(pix_reg_file)
    expmap_file = os.path.abspath(expmap_file)
    spec_file = os.path.abspath(spec_file)
    rmf_file = os.path.abspath(rmf_file)

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaarfgen_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # We can use the spectrum file name to set up the output ARF file name
    arf_out = os.path.basename(spec_file).replace("-spectrum.fits", ".arf")

    # Set up a name for the ray-traced simulated event file required for
    #  XRISM ARF generation
    ray_traced_evt_out = (
        f"xrism-resolve-obsid{cur_obs_id}-numphoton{num_photons}-"
        f"enALL-raytracedevents.fits"
    )

    # If a ray-traced event file with the same already exists, we're just going
    #  to point to it with the absolute path (saves on re-running expensive
    #  ray tracing).
    if os.path.exists(os.path.join(os.path.abspath(out_dir), ray_traced_evt_out)):
        ray_traced_exists = True
        ray_traced_evt_out = os.path.abspath(os.path.join(out_dir, ray_traced_evt_out))
    else:
        ray_traced_exists = False

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.xaarfgen(
            xrtevtfile=ray_traced_evt_out,
            outfile=arf_out,
            sourcetype="POINT",
            numphotons=num_photons,
            minphotons=min_photons,
            source_ra=rel_coord.ra.value,
            source_dec=rel_coord.dec.value,
            regionfile=os.path.relpath(pix_reg_file),
            telescop="XRISM",
            instrume="RESOLVE",
            emapfile=os.path.relpath(expmap_file),
            rmffile=os.path.relpath(rmf_file),
            regmode="DET",
            noprompt=True,
            clobber=True,
        )

    # Move the ARF and ray traced event files up from the temporary directory
    os.rename(os.path.join(temp_work_dir, arf_out), os.path.join(out_dir, arf_out))
    # If the ray traced file already existed, we don't need to move anything
    if not ray_traced_exists:
        os.rename(
            os.path.join(temp_work_dir, ray_traced_evt_out),
            os.path.join(out_dir, ray_traced_evt_out),
        )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out, os.path.join(out_dir, arf_out)


def det_region_from_pixels(new_reg_path: str, include_pixels: List[int] = None):
    """
    Simply generates a region file in XRISM-Resolve detector coordinates from an
    input set of XRISM-Resolve pixel IDs.

    :param List[int] include_pixels: Controls which XRISM-Resolve pixels are included
        in the output region file. Default is None, in which case all pixels, aside
        from the calibration pixel 12, are included. Otherwise pass a list of
        integer pixel IDs.
    """

    if include_pixels is not None:
        if not isinstance(include_pixels, (list, str, int)):
            raise TypeError("'include_pixels' must be a list of integer pixel IDs.")
        elif not isinstance(include_pixels, list) and isinstance(
            include_pixels, (str, int)
        ):
            include_pixels = [include_pixels]

        try:
            include_pixels = np.array([int(rp) for rp in include_pixels])
        except ValueError:
            raise TypeError(
                "All entries in 'include_pixels' must be integer "
                "XRISM-Resolve pixel IDs."
            )

    else:
        include_pixels = np.append(np.arange(0, 12), np.arange(13, 36))

    #
    rel_pix_regs = [RESOLVE_PIX_DET_REGIONS[cur_pix] for cur_pix in include_pixels]

    with open(new_reg_path, "w") as new_rego:
        new_rego.writelines(cur_reg + "\n" for cur_reg in rel_pix_regs)


def plot_fit_spec(
    plot_data: dict,
    sp_color: str = "navy",
    mod_color: str = "firebrick",
    res_color: str = "navy",
    x_lims: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Tuple[float, float]] = None,
    inst_name: Optional[str] = None,
    mod_expr: Optional[str] = None,
    fig_size: Optional[Tuple[Union[float, int]]] = None,
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
    :param str mod_expr: Optionally, the 'expression' of the fitted model - to be
        added to its legend label.
    :param Optional[Tuple[Union[float, int]]] fig_size: Optionally, a tuple controlling
        the size of the figure producted by this function. Default is None, which
        corresponds to a size of (7, 6).
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
    ]
    # Raise an error before we get started plotting if any entries are missing
    if any([en not in plot_data for en in req_ents]):
        raise KeyError(
            f"Plot data must contain the following keys: {', '.join(req_ents)}"
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

    if fig_size is None:
        fig_size = (7, 6)

    fig, ax_arr = plt.subplots(
        nrows=2, figsize=fig_size, height_ratios=(3, 1.5), sharex=True
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

    spec_ax.plot(
        plot_data["energy"],
        plot_data["model"],
        color=mod_color,
        label=mod_label,
        alpha=0.8,
        zorder=2,
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
    # We don't set the spectrum to be normalized by area in this notebook, so
    #  have y-axis labels of ct/s/keV
    res_ax.set_ylabel(r"Residuals [$\frac{\rm{ct}}{\rm{s} \: \rm{keV}}$]", fontsize=15)

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
# The name of the source we're examining in this demonstration
SRC_NAME = "NGC1365"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 2

# The approximate linear relationship between Resolve PI and event energy
RSL_EV_PER_CHAN = (1 / Quantity(2000, "chan/keV")).to("eV/chan")

# Expansion of event grade entries in event lists to something
#  a little more descriptive
DESCRIPTIVE_RESOLVE_EVT_GRADES = {
    "Hp": "High-resolution Primary [0]",
    "Mp": "Mid-resolution Primary [1]",
    "Ms": "Mid-resolution Secondary [2]",
    "Lp": "Low-resolution Primary [3]",
    "Ls": "Low-resolution Secondary [4]",
    "Bl": "Baseline event (diagnostic) [5]",
    "El": "Lost event [6]",
    "Rj": "Rejected event [7]",
}

RESOLVE_EVT_GRADES_INT_SHORT = {0: "Hp", 1: "Mp", 2: "Ms", 3: "Lp", 4: "Ls"}

# Relation of XRISM-Resolve fits header FILTER values to the equivalent
#  XRISM-Resolve file naming scheme
RESOLVE_FILTERS = {
    "OPEN": "px1000",
    "FE55": "px5000",
    "BE": "px4000",
    "ND": "px3000",
    "POLY": "px2000",
    "UNDEF": "px0000",
}

RESOLVE_PIX_DET_REGIONS = {
    0: "box(4,3,1,1,0)",
    1: "box(6,3,1,1,0)",
    2: "box(5,3,1,1,0)",
    3: "box(6,2,1,1,0)",
    4: "box(5,2,1,1,0)",
    5: "box(6,1,1,1,0)",
    6: "box(5,1,1,1,0)",
    7: "box(4,2,1,1,0)",
    8: "box(4,1,1,1,0)",
    9: "box(1,3,1,1,0)",
    10: "box(2,3,1,1,0)",
    11: "box(1,2,1,1,0)",
    13: "box(2,2,1,1,0)",
    14: "box(2,1,1,1,0)",
    15: "box(3,2,1,1,0)",
    16: "box(3,1,1,1,0)",
    17: "box(3,3,1,1,0)",
    18: "box(3,4,1,1,0)",
    19: "box(1,4,1,1,0)",
    20: "box(2,4,1,1,0)",
    21: "box(1,5,1,1,0)",
    22: "box(2,5,1,1,0)",
    23: "box(1,6,1,1,0)",
    24: "box(2,6,1,1,0)",
    25: "box(3,5,1,1,0)",
    26: "box(3,6,1,1,0)",
    27: "box(6,4,1,1,0)",
    28: "box(5,4,1,1,0)",
    29: "box(6,5,1,1,0)",
    30: "box(6,6,1,1,0)",
    31: "box(5,5,1,1,0)",
    32: "box(5,6,1,1,0)",
    33: "box(4,5,1,1,0)",
    34: "box(4,6,1,1,0)",
    35: "box(4,4,1,1,0)",
}
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
    ROOT_DATA_DIR = "../../../_data/XRISM/"
else:
    ROOT_DATA_DIR = "XRISM/"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("XRISM_output")
os.makedirs(OUT_PATH, exist_ok=True)
# --------------------------------------------------------------


# ------------- Set up output file path templates --------------
# ------ XAPIPELINE -------
# Cleaned event list path template - obviously going to be useful later
EVT_PATH_TEMP = os.path.join(OUT_PATH, "{oi}", "xa{oi}rsl_p0{xrf}_cl.evt")

# The path to the bad pixel map, useful for excluding dodgy pixels from data products
BADPIX_PATH_TEMP = os.path.join(OUT_PATH, "{oi}", "xa{oi}rsl_p{sc}{xrf}.bimg")
# --------------------------

# -- SCREENED EVENT LISTS --
# Template for the path to screened event lists
# TODO NEED TO FINISH THIS OBV.
SCR_EVT_PATH_TEMP = os.path.join(
    OUT_PATH, "{oi}", "xrism-resolve-obsid{oi}-filter{xrf}--clean-events.fits"
)
# --------------------------

# --------- IMAGES ---------
IM_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-resolve-obsid{oi}-filter{xrf}-imbinfactor{ibf}-en{lo}_{hi}keV-image.fits",
)
# --------------------------


# -------- EXPMAPS ---------
EX_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-resolve-obsid{oi}-filter{xrf}-attraddelta{rd}arcmin-"
    "attphibin{npb}-enALL-expmap.fits",
)
# --------------------------


# ------ LIGHTCURVES -------
LC_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-resolve-obsid{oi}-filter{xrf}-ra{ra}-dec{dec}-radius{rad}deg-"
    "en{lo}_{hi}keV-expthresh{lct}-tb{tb}s-lightcurve.fits",
)

BACK_LC_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-resolve-obsid{oi}-filter{xrf}-ra{ra}-dec{dec}-"
    "en{lo}_{hi}keV-expthresh{lct}-tb{tb}s-back-lightcurve.fits",
)

NET_LC_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-resolve-obsid{oi}-filter{xrf}-ra{ra}-dec{dec}-radius{rad}deg-"
    "en{lo}_{hi}keV-expthresh{lct}-tb{tb}s-net-lightcurve.fits",
)
# --------------------------


# -------- SPECTRA ---------
SP_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-resolve-obsid{oi}-filter{xrf}-pix{slp}-res{grd}-enALL-spectrum.fits",
)

# BACK_SP_PATH_TEMP = os.path.join(
#    OUT_PATH,
#    "{oi}",
#    "xrism-resolve-obsid{oi}-filter{xrf}-ra{ra}-dec{dec}-enALL-back-spectrum.fits",
# )
# --------------------------

# ----- GROUPEDSPECTRA -----
GRP_SP_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum", "-{gt}grp{gs}-spectrum")
# --------------------------

# ---------- RMF -----------
# TODO THIS IS NOT CORRECT
RMF_PATH_TEMP = os.path.join(
    OUT_PATH, "{oi}", "xrism-resolve-obsid{oi}-filter{xrf}.rmf"
)
# --------------------------

# ---------- ARF -----------
ARF_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum.fits", ".arf")
# --------------------------
# --------------------------------------------------------------
```

***

## 1. Finding and downloading XRISM observations of NGC 1365

Our first task is to determine which XRISM observations are relevant to the source
that we are interested in.

We are going in with the knowledge that NGC 1365 has been observed by XRISM, but of
course, there is no guarantee that _your_ source of interest has been, so this is
an important exploratory step.

### Determining the name of the XRISM observation summary table

HEASARC maintains tables that contain information about every observation taken by
each of the missions in its archive. We will use XRISM's table to find observations
that should be relevant to our source.

The name of the XRISM observation summary table is 'xrismmastr', but as you may not
know that a priori, we demonstrate how to identify the correct table for a given
mission.

Using the AstroQuery Python module (specifically this Heasarc object), we list all
catalogs that are **(a)** related to XRISM, and **(b)** are flagged as 'master' (meaning the
table summarising all observations). This should only return one catalog for any
mission you pass to 'keywords':

```{code-cell} python
catalog_name = Heasarc.list_catalogs(master=True, keywords="xrism")[0]["name"]
catalog_name
```

### What are the coordinates of NGC 1365?

To search for relevant observations, we have to know the coordinates of our
source. The astropy module allows us to look up a source name in CDS' Sesame name
resolver and retrieve its coordinates.

```{hint}
You could also set up a SkyCoord object directly, if you already know the coordinates.
```

```{code-cell} python
SRC_COORD = SkyCoord.from_name(SRC_NAME).transform_to("icrs")
# This will be useful later on in the notebook, for functions that take
#  coordinates as an astropy Quantity.
SRC_COORD_QUANT = Quantity([SRC_COORD.ra, SRC_COORD.dec])
SRC_COORD
```

### Searching for relevant observations

Now that we know which catalog to search, and the coordinates of our source, we use
AstroQuery to retrieve those lines of the summary table that are within some radius
of the source coordinate. We're using the default search radius for
the XRISM summary table, but you can pass a `radius` argument to set your own.

In this case, we also define a custom set of columns to retrieve, as the default set
does not contain some Resolve-specific columns that we might need later. You may also
pass a wildcard `columns='*'` to retrieve all available columns.

```{code-cell} python
col_str = (
    "__row,obsid,name,ra,dec,time,exposure,status,public_date,"
    "rsl_datamode,rsl_fil_be,rsl_fil_fe55,rsl_fil_nd,rsl_fil_poly,"
    "rsl_fil_open,rsl_fil_undef"
)

all_xrism_obs = Heasarc.query_region(SRC_COORD, catalog_name, columns=col_str)
all_xrism_obs
```

For an active mission (i.e., actively collecting data and adding to the archive), we
will, at some point, probably come across observations that have been taken, but are
currently only available to their proposers (still in the proprietary period).

Such proprietary observations will still appear in the XRISM summary table, and the
files could even be downloaded, but unless we took those data, we won't have the
key necessary to decrypt the files.

As such, we are going to use the 'public_date' column to filter out any observations
that are not yet publicly available:

```{code-cell} python
public_times = Time(all_xrism_obs["public_date"], format="mjd")
avail_xrism_obs = all_xrism_obs[public_times <= Time.now()]

avail_xrism_obs
```

It's also worth noting that _scheduled but not yet taken_ observations are also included
in the XRISM master table, and the filtering we just performed will also exclude those
cases.

We can see that there are two public XRISM observations of NGC 1365
(as of May 2026) – with ObsIDs of **300075010** and _300075020_. To ensure that this
demonstration notebook will run in a reasonable length of time, we
will restrict ourselves to using one observation (300075010; chosen primarily
because it illustrates the [problem with pixel 27](#pixel-27-of-xrism-resolve-is-broken)
better than the other), by filtering the `avail_xrism_obs` table:

```{code-cell} python
avail_xrism_obs = avail_xrism_obs[avail_xrism_obs["obsid"] == "300075010"]

# Create an array of the relevant ObsIDs
rel_obsids = avail_xrism_obs["obsid"].value.data

# Create a dictionary storing which filters were used for each ObsID
rel_filters = {
    row["obsid"]: [
        RESOLVE_FILTERS[f]
        for f in RESOLVE_FILTERS
        if row[f"rsl_fil_{f.lower()}"] == "Y"
    ]
    for row in avail_xrism_obs
}
```

```{important}
Though we have chosen to demonstrate using a **single observation** of NGC 1365, we
note that the notebook is designed so that it can handle any number of observations. As such, if you
wish to adapt this demonstration to examine a different source, with multiple observations, it
should work without modification.
```

### Downloading the XRISM observation

The AstroQuery `Heasarc` module makes it easy to download the data we need. Our
cut-down table of observations can be passed to the `locate_data()` method, which
will return the access links for the data on several different platforms:

```{code-cell} python
data_links = Heasarc.locate_data(avail_xrism_obs)
data_links
```

That data links table can now be passed straight to the `download_data()` method, which
will do what it says on the tin and download the files. We can also specify which
platform to pull the observations from, and in this case we select the HEASARC AWS S3 bucket:

```{code-cell} python
Heasarc.download_data(links=data_links, host="aws", location=ROOT_DATA_DIR)
```

```{note}
We choose to download the data from the HEASARC AWS S3 bucket, but you could
pass 'heasarc' to acquire data from the FTP server. Additionally, if you are working
on SciServer, you may pass 'sciserver' to use the pre-mounted HEASARC dataset.
```

### What do the downloaded data directories contain?

Now we can take a quick look at the contents of the directory we just downloaded:

```{code-cell} python
glob.glob(os.path.join(ROOT_DATA_DIR, rel_obsids[0], "") + "*")
```

```{code-cell} python
glob.glob(os.path.join(ROOT_DATA_DIR, rel_obsids[0], "resolve", "") + "**/*")
```

## 2. Processing XRISM-Resolve data

There are multiple steps involved in processing XRISM-Resolve data into a
science-ready state.

As with many NASA-affiliated high-energy missions, HEASoft
includes a beginning-to-end pipeline to streamline this process for XRISM data - the
XRISM-Resolve and Xtend instruments both have their own pipelines.

XRISM also has an overall pipeline that orchestrates the running of both instrument
specific pipelines, as well as automatically determining the paths to the various
housekeeping files included in the data download necessary for processing the data.

We will show you how to run this top-level XRISM pipeline (`xapipeline`), but
will limit it to processing only XRISM-Resolve data (though it is quite capable of
preparing both Resolve and Xtend data).

The Python interface to HEASoft, HEASoftPy, is used throughout this tutorial, and we
will implement parallel observation processing wherever possible (even though we have
only selected a single observation).

### HEASoft and HEASoftPy versions

```{warning}
XRISM is a relatively new mission, and as such the analysis software and recommended
best practises are still immature and evolving. We are checking and updating this tutorial
on a regular basis, but please report any issues, or make suggestions, to
the [XRISM Help Desk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xrism).
```

Both the HEASoft and HEASoftPy package versions can be retrieved from the
HEASoftPy module.

The HEASoftPy version:

```{code-cell} python
hsp.__version__
```

The HEASoft version:

```{code-cell} python
fver_out = hsp.fversion()
fver_out
```

It is likely that this tutorial will not run all the way through if you are using
a version of HEASoft older than **v6.36**, so we will check for that and raise an
error if it is the case. First, extract the version string from the `fversion` output, and
set up a `Version` object:

```{code-cell} python
fver_out.output[0].split("_")[-1]
HEA_VER = Version(fver_out.output[0].split("_")[-1])
HEA_VER
```

We can now check that `HEA_VER` is greater than the minimum required version:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
if HEA_VER < Version("v6.36"):
    raise ValueError(
        "We strongly recommend using HEASoft v6.36 or later for this "
        "tutorial - you may run rest of the notebook yourself, but "
        "ARF generation will either fail or produce an incorrect result."
    )
```

```{important}
We are also aware of a problem in HEASoft **v6.36** where one step of the
XRISM-Resolve pipeline (which we use in [the next section](#running-the-xrism-pipeline-for-resolve))
is not compatible with remote CALDB files.

This issue will be fixed in a future HEASoft release.
```

As a workaround we use HEASoftPy's interface to the `quzcif` task to fetch the single
CALDB file that the failing step requires; though only if the version of HEASoft is
v6.36 or older, and if the CALDB is currently set up to use remote files.

If we _do_ have to download the file, the path is passed to the
`process_xrism_resolve(...)` wrapper function defined in [Global Setup: Functions](#functions)
and discussed in the [Running the XRISM pipeline for Resolve](#running-the-xrism-pipeline-for-resolve) section.

```{code-cell} python
# A constant that will be passed to the wrapper function for 'xapipeline', only needs
#  to be not None if using a remote CALDB and HEASoft v6.36 or lower
RSLMPCOR_PATH = None

# Have to fetch file if HEASoft v6.36 or lower, and using a remote CALDB setup
if HEA_VER <= Version("v6.36") and (
    os.environ["CALDB"].startswith("https") or ".com/" in os.environ["CALDB"]
):
    warn(
        "Downloading the XRISM-Resolve 'RSLMPCOR' CALDB file to avoid HEASoft v6.36's "
        "small incompatibility with a remote XRISM-Resolve CALDB.",
        stacklevel=1,
    )

    # This will find and download (retrieve=True) the XRISM-Resolve mid-resolution
    #  primary/secondary event channel correction file
    with contextlib.chdir(ROOT_DATA_DIR):
        caldb_ret = hsp.quzcif(
            mission="xrism",
            instrument="resolve",
            codename="RSLMPCOR",
            date="-",
            retrieve=True,
            noprompt=True,
            clobber=True,
        )

    # Set the path to the downloaded file
    RSLMPCOR_PATH = os.path.join(ROOT_DATA_DIR, caldb_ret.output[0].split(" ")[0])

    # We include a small validity check to make sure we get an informative error if
    #  something goes wrong when downloading the CALDB file.
    if not RSLMPCOR_PATH.endswith(".fits") or not os.path.exists(RSLMPCOR_PATH):
        # Show the output to give us a clue of what happened
        print(caldb_ret)
        raise FileNotFoundError(
            "Download of the XRISM-Resolve 'RSLMPCOR' CALDB file has failed."
        )
```

```{note}
This notebook is configured to acquire XRISM CALDB files from the HEASARC
Amazon Web Services S3 bucket - this can greatly improve the speed of some
steps later in the notebook, particularly when running on the Fornax Science Console.

CALDB location configuration can be found in the [Global Setup: Configuration](#configuration) section.
```

### Running the XRISM pipeline for Resolve

`xapipeline` needs the 'stem' of the input file names to be defined, so that it
can identify the relevant event list files. The way we call the pipeline, the input
stem will also be used to format output file names.

```{code-cell} python
file_stem_temp = "xa{oi}"
```

The pipeline has three stages and provides the option to start and stop the processing
at any of those stages; this can be useful if you wish to re-run a stage with slightly
different configuration without repeating the entire pipeline run.

A stage is a collection of different tasks, and have the following general goals:
- **Stage 1** - Calibrating the events.
- **Stage 2** - Screening the events.
- **Stage 3** - Producing quick-look data products.

The [**`xapipeline` documentation**](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xapipeline.html) breaks down
exactly which HEASoft tools are during each stage.

```{note}
We will stop the execution of `xapipeline` at **Stage 2**, as the latter part of this
demonstration will show you how to make more customised data products than are output
by default.
```

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [
        [
            os.path.join(ROOT_DATA_DIR, oi),
            oi,
            os.path.join(OUT_PATH, oi),
            file_stem_temp.format(oi=oi),
            RSLMPCOR_PATH,
        ]
        for oi in rel_obsids
    ]

    pipe_result = p.starmap(process_xrism_resolve, arg_combs)

xa_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in xa_pipe_problem_ois]

xa_pipe_problem_ois
```

```{warning}
Processing XRISM-Resolve data can take a long time, up to several hours for a single observation.
```

We also include a code snippet that will print the output of the `xapipeline` run for any
observations that appear to have failed:

```{code-cell} python
if len(xa_pipe_problem_ois) != 0:
    for all_out in pipe_result:
        if all_out[0] in xa_pipe_problem_ois:
            print(all_out[1])
            print("\n\n")
```

## 3. Choosing the events to consider for data product generation

Before we can begin filtering out events that we don't want for our particular analysis, we first
have to choose which event **list** we want to use. Many XRISM-Resolve observations will have
multiple event lists associated with them, representing periods of the observation that were
taken with different filters applied; some will contain genuine events detected from the target source(s), and
others will contain events from calibration sources onboard XRISM.

The filter that was active for a particular event list is indicated by the four-digit section of the file name following 'px' (and also in the
event file header) – the substrings map to the following filters:
- **px0000** – Undefined
- **px1000** – Open
- **px2000** – Al/Polyimide
- **px3000** – Neutral Density (ND)
- **px4000** – Be
- **px5000** – Fe 55 calibration source

Exactly which event list you pick will depend on the type of analysis you are performing, but this demonstration
is operating under the assumption that we don't want to directly use the calibration source event lists.

Most XRISM-Resolve observations will have at least undefined-filter and calibration-filter event lists, as
well as an event list taken with the PI's filter of choice (though it is possible that there will be multiple
'science' filters requested by the PI).

In the latter part of the [searching for relevant observations section](#searching-for-relevant-observations) we set
up a dictionary that stores which filters were used for each observation (though this demonstration's default behavior
is to use just one observation, **300075010**). Reminding ourselves of the contents of this dictionary, we can
see that the undefined, calibration, and open filters were used:

```{code-cell} python
rel_filters
```

From here on, we limit ourselves to the event list created from the open-filter observation. The `rel_filters`
dictionary will be cut down to only include filters specified in the `USE_FILTERS` variable defined below - the
code here is set up so that multiple filters can be selected, if you wish to modify this tutorial:

```{code-cell} python
USE_FILTERS = ["px1000"]

cut_rel_filters = {
    oi: [cur_fi for cur_fi in filts if cur_fi in USE_FILTERS]
    for oi, filts in rel_filters.items()
}
cut_rel_filters
```

```{important}
Though we have specified a single filter, we note that this demonstration is designed so that it can process
and deal with multiple event lists, with different filters, for each observation.
```

### Loading event lists into Python

```{code-cell} python
# Set up a two-level nested dictionary (ObsID top level keys, filter as low level keys)
#  with EventList instances as values
evt_lists = {
    oi: {
        cur_filt: EventList(EVT_PATH_TEMP.format(oi=oi, xrf=cur_filt))
        for cur_filt in cur_filts
    }
    for oi, cur_filts in cut_rel_filters.items()
}
evt_lists
```

```{code-cell} python
cur_evt_list = evt_lists[rel_obsids[0]][cut_rel_filters[rel_obsids[0]][0]]
```

### Pixel 12 is a dedicated calibration pixel

XRISM-Resolve is constructed as a **6x6** array of microcalorimeter detector 'pixels', but only
**35 of 36** are exposed to the light focused by the X-ray optics. The one left out pixel, **pixel 12**, is
dedicated only to observations of a calibration source that is built into the Resolve instrument – it is
separate from the Fe 55 calibration source housed in the filter wheel, which we briefly discussed in
the introduction to [Section 3](#3-choosing-the-events-to-consider-for-data-product-generation).

Pixel 12 is most conspicuous by its absence from cleaned event lists – you will not find any events
recorded by the calibration pixel, which is **centered at DETX, DETY = 1, 1** (putting it in the
bottom left of the array). We can quickly demonstrate this by fetching the detector coordinates of
every event recorded in the event list we just produced by
[running the XRISM xapipeline](#running-the-xrism-pipeline-for-resolve), then plotting
a binned detector-coordinate 'image':

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(6.7, 5.5))
ax = plt.gca()
ax.set_axis_off()

detxy_data = cur_evt_list.data[["DETX", "DETY"]]

det_im_arr = np.histogram2d(
    detxy_data["DETX"], detxy_data["DETY"], np.arange(0.5, 7.5)
)[0]

plt.imshow(det_im_arr, origin="lower", cmap="gnuplot2")

ax.add_artist(Line2D([-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], color="snow", lw=3))
plt.text(
    x=0,
    y=0,
    s="12",
    color="snow",
    fontsize=25,
    fontweight="bold",
    horizontalalignment="center",
    verticalalignment="center",
)

ax.add_artist(Line2D([5.5, 4.5, 4.5, 5.5], [3.5, 3.5, 2.5, 2.5], color="gold", lw=3))
plt.text(
    x=5,
    y=3,
    s="27",
    color="gold",
    fontsize=25,
    fontweight="bold",
    horizontalalignment="center",
    verticalalignment="center",
)

cb = plt.colorbar()
cb.set_label("Counts", size=15, rotation=270, va="bottom")

plt.tight_layout()
plt.show()
```

Though there are a great many events attributed to other pixels in the XRISM-Resolve array, the
total count value of the bottom-left pixel is zero.

```{important}
Though the rest of the detector (i.e. pixels 0-35) is not directly exposed to pixel 12's
calibration source, the source can still have an indirect impact on the data recorded by
pixels 11 and 13 via 'electrical cross-talk'. This concept is discussed
[in a later section of this demonstration](#electrical-cross-talk).
```

### Event grades and branching ratios

Every event recorded by the XRISM-Resolve instrument is assessed by the post-observation
pipeline (**not** `xapipeline`, but rather an internal pipeline used to handle the raw
data delivered from the satellite). This assessment is intended to flag potentially
problematic events, which can occur for a whole host of different reasons, so that the
end user (you!) can decide which events are safe to keep for their particular analysis.

The values representing the quality of each event are stored in the event list, under
the **TYPE/ITYPE** (both provide the same information, just in slightly different
formats) and **STATUS** columns. These XRISM-Resolve event list columns (and many others)
are described in detail by [XRISM GOF & SDC (2024)](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html#SECTION00770000000000000000).

The **STATUS** column is addressed in a [later section of this demonstration](#excluding-pixel-pixel-coincident-events).

In the **TYPE/ITYPE** columns, you will find what other missions may refer to as the
event 'grade', which for XRISM-Resolve essentially represents the precision to which the
event's energy can be determined (i.e., the energy resolution). Exactly how XRISM-Resolve
events are graded is discussed in detail by
[XRISM GOF & SDC (2026)](https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html#sec:resolve_eventgrading),
but the most important concepts are:
1. XRISM-Resolve pixels are essentially thermometers.
2. An arriving photon's energy is defined by how it alters the temperature of the pixel.
3. Not being magic, the pixel takes some quantifiable time to both read out the temperature and to cool down after the arrival and detection of a photon.
4. More photons arriving before the pixel has had time to cool down (re-equilibrate) can cause biases in the measurement of those photon's energies, and more uncertainty as to the precise energy, so they are marked as lower resolution events.
5. Even worse, if another photon (or photons) arrive before the temperature can be read out, it is not possible for the pixel to determine that they **are** separate photons, and they will be read out as a single event (**pile-up, usually only a problem for very bright sources**).

The exact grades assigned to lower resolution events are determined by comparing the time-separation between
neighboring temperature 'pulses' in the same pixel to limiting values determined by the mission team. The following
grades (TYPE column in the event list) can be assigned to an event (the value in brackets is the ITYPE equivalent):
- **Hp (0)** – High-resolution primary
- **Mp (1)** – Mid-resolution primary
- **Ms (2)** – Mid-resolution secondary
- **Lp (3)** – Low-resolution primary
- **Ls (4)** – Low-resolution secondary

Other event grades can be assigned, but are less likely to be seen in science data:
- **Bl (5)** – Baseline event (diagnostic)
- **El (6)** – Lost event
- **Rj (7)** – Rejected event

***<span style="color:red">ADD A SHORT DISCUSSION HERE ABOUT THE APPROXIMATE ENERGY RESOLUTIONS OF EACH GRADE</span>***

In an ideal world our entire event list would entirely consist of high-resolution primary events, though unfortunately,
that is not very likely to happen. To get an idea of different event grade's relative occurrence rates, at least
for our observation of NGC 1365, we can construct a histogram from the event list we produced by
[running the XRISM xapipeline in a previous section](#running-the-xrism-pipeline-for-resolve).

On the left side y-axis, the histogram represents the absolute number of events which were assigned
a particular grade, and on the right side y-axis, the 'branching ratio' of each grade. The branching
ratio is defined as the fraction of events that fall into a particular event grade:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(6.5, 5))
plt.tick_params(direction="in", bottom=False)

evt_grade_names, evt_grade_cnts = np.unique(
    cur_evt_list.data["TYPE"], return_counts=True
)
# Now we put the event grades in the same order as they are defined in the
#  DESCRIPTIVE_RESOLVE_EVT_GRADES constant
grade_matches = (
    np.array(list(DESCRIPTIVE_RESOLVE_EVT_GRADES.keys()))[:, None]
    == evt_grade_names.value
)
sort_indices = np.where(grade_matches)[1]
evt_grade_names = evt_grade_names[sort_indices]
evt_grade_cnts = evt_grade_cnts[sort_indices]

grade_colors = plt.cm.plasma(np.linspace(0, 1, len(evt_grade_names)))

grade_bars = plt.bar(evt_grade_names, evt_grade_cnts, width=0.75, color=grade_colors)

for cur_bar_ind, cur_bar in enumerate(grade_bars):
    cur_bar.set_label(DESCRIPTIVE_RESOLVE_EVT_GRADES[evt_grade_names[cur_bar_ind]])

plt.ylabel("Number of Events", fontsize=15)
plt.xlabel("Event Grade", fontsize=15)

sec_ax = plt.gca().secondary_yaxis(
    "right",
    functions=(lambda x: x / evt_grade_cnts.sum(), lambda x: x * evt_grade_cnts.sum()),
)
sec_ax.set_ylabel("Branching Ratio", rotation=270, va="bottom", fontsize=15)
sec_ax.minorticks_on()
sec_ax.tick_params(which="both", direction="in")

plt.ylim(0, evt_grade_cnts.sum())

plt.legend()
plt.title(f"XRISM-Resolve Event Grades [{cur_evt_list.obs_id}]", fontsize=16)

plt.tight_layout()
plt.show()
```

```{note}
Energy scale accuracy is only truly guaranteed for high-resolution events at the
moment, so we currently recommend limiting analysis to events with a grade of **Hp**.
```


### Overabundance of low-resolution secondary (**Ls**) events

The histogram of event grade counts and branching ratios we constructed
[in the last section](#event-grades-and-branching-ratios) demonstrates a serious issue
currently affecting the majority of XRISM-Resolve observations – **a severe overabundance of low-resolution secondary events (Ls).**

This problem is discussed in the XRISM ABC guide
([XRISM GOF & SDC 2024](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Resolve_Data_Analysis.html#SECTION00932000000000000000)),
which provides recommendations on how to handle these anomalous Ls events.

Anomalous Ls events **do not represent real, celestial, X-ray photons**. Instead, they are currently
believed to have two distinct origins:
1. The first component, dominating the 'low count rate' regime ($\lesssim 0.4$ [$\rm{ct}\:\rm{pix}^{-1}\:s^{-1}$]), likely derives "from cosmic-ray particles, or instrumental X-rays induced by cosmic rays".
2. The second, which occurs for observations with count rates significantly higher than $\sim 1$ [$\rm{ct}\:\rm{pix}^{-1}\:s^{-1}$], is likely due to "secondary signals produced by initial (probably energetic) X-rays".

```{note}
As XRISM is a relatively new mission, the best practice of data analysis is still under active development by the mission team. That
is particularly true when dealing with issues like this one, as the large numbers of Ls events we see in XRISM data were not
expected from pre-launch testing. The XRISM demonstration articles in HEASARC-tutorials are kept up to date, but we
strongly recommend that you also check the XRISM ABC guide to see if any new recommendations have been made.
```

#### What to do for faint sources

We currently recommend that **all Ls events be excluded from analysis for observations with per-pixel count rates of
$\lesssim 0.4$ [$\rm{ct}\:\rm{pix}^{-1}\:s^{-1}$]**, which translates to a total of $\lesssim 1$ [$\rm{ct}\:s^{-1}$] for
a point source.


#### What to do for bright sources

For moderately bright emission, up to a total of $\sim 10$ [$\rm{ct}\:s^{-1}$] for a point source, the
recommendation is to:
1. **Remove all Ls events**, just as [for faint sources](#what-to-do-for-faint-sources).
2. Restrict yourself to analyzing **events with energies between 3–10 keV**.
3. Assume that **whatever source fluxes you derive are lower limits**, as a not-entirely-negligible fraction of the excluded Ls events will have been real.

This is because in this count rate regime, the second component of the anomalous Ls event abundance comes into play (see the [beginning of this section](#overabundance-of-low-resolution-secondary-ls-events)).

#### What to do for very bright sources

The low-resolution secondary (Ls) events recorded for very bright sources exhibit
complicated behaviors, and the XRISM-Resolve instrument team is still working to develop
the most effective methods of analyzing Ls data for very bright sources.

This is one area in which the current best practises are likely to evolve rapidly, so be sure
to check the latest version of the XRISM ABC guide. The XRISM-Resolve instrument team is conducting
a comprehensive study to determine the most effective strategies for very bright sources.

Currently available XRISM-Resolve response files are highly uncertain for very high count rate data, which
in makes the derived absolute flux and global spectral shape highly uncertain. Our current advice
is that, when analyzing very high count rate data, users should currently limit analyzes to narrow
energy bands.


```{danger}
It is **essential** to understand that some
XRISM-Resolve analysis tasks - the `rslmkrmf` response matrix file generator, which we
use in [a later section](#producing-redistribution-matrix-files-rmfs) in particular - will behave
poorly if the Ls events are not excluded.

`rslmkrmf`, for instance, normalizes the output response in part by the ratio of the
number events of selected grades (Hp and Mp for instance) to the number of events of
all grades from Hp to Ls (inclusive) - however, this normalization was intended to
be based on 'real' Ls events, rather than the glut of anomalous events we end
up seeing, so if they are not excluded the net effective area of the Resolve spectral
responses could be in error by as much as a factor of $\sim2$.
```

<span style="color:red">***Now need to talk about how we're going to handle this in the demonstration.***</span>


### Pixel 27 of XRISM-Resolve is broken

Another of XRISM-Resolve's teething problems is that the microcalorimeter labeled as 'pixel 27' (we
highlighted the location in the figure
[produced in a previous section](#pixel-12-is-a-dedicated-calibration-pixel)) has significantly
different gain variation characteristics to all the other microcalorimeters in the XRISM-Resolve array.

'Gain' in this context is what describes the relationship between the 'pulse' of temperature
increase-then-decrease recorded by the microcalorimeter's electronics and the actual energy of the
detected photon. You can see the necessity of being able to trust the gain calculated for
every event!

Unfortunately, when tracking the 'gain history' of each pixel during a long observation, strange spikes in
gain versus time were identified for pixel 27, when compared to the smoother gain versus time curves
of all the other pixels. The practical meaning of this is that you **cannot fully trust the
energies assigned to pixel 27 events**.

We can illustrate this problem by showing the 'gain history' of the XRISM-Resolve pixels during
our observation of NGC 1365 – read from the gain history file produced
[when we ran `xapipeline`](#running-the-xrism-pipeline-for-resolve). This figure
plots the temperature of the pixel against time, and we can
clearly see that pixel 27's gain history curve has a significantly different shape to those
of the other pixels. In particular, it jumps up in temperature toward the end of the observation, whereas
all other pixels decrease in temperature:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Set up the path to the gain history file we're using as an example
rel_ghf_path = os.path.join(
    OUT_PATH, rel_obsids[0], f"xa{rel_obsids[0]}rsl_000_fe55.ghf"
)

# Load the file in to memory
with fits.open(rel_ghf_path) as gaino:
    gain_tab = Table(gaino["Drift_energy"].data)[["TIME", "PIXEL", "TEMP_FIT"]]

# Bad way to get the gain curve in x-units of APPROXIMATELY seconds from the
#  beginning of the observation. Do not use this method to calculate
#  time-since-start when the times actually matter
gain_tab["TIME"] = gain_tab["TIME"] - gain_tab["TIME"][0]

plt.figure(figsize=(6, 4.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

for pix_id in set(gain_tab["PIXEL"]):

    cur_sub_gain = gain_tab[gain_tab["PIXEL"] == pix_id]

    if pix_id != 27:
        plt.plot(cur_sub_gain["TIME"], cur_sub_gain["TEMP_FIT"], alpha=0.3, lw=0.5)

    else:
        plt.plot(
            cur_sub_gain["TIME"],
            cur_sub_gain["TEMP_FIT"],
            alpha=1,
            color="dodgerblue",
            lw=2,
            label="PIXEL 27",
        )

plt.xlim(0, gain_tab["TIME"].max())
plt.ylim(gain_tab["TEMP_FIT"].min() * 0.999, gain_tab["TEMP_FIT"].max() * 1.001)

plt.ylabel("Fit Temperature [K]")
plt.xlabel("Time [s]", fontsize=15)

plt.legend(loc=1, fontsize=14)
plt.tight_layout()
plt.show()
```

Investigations into the cause of this problem and potential strategies for mitigation are ongoing, but the
current best practice is to **exclude pixel 27 from analysis entirely**.

```{code-cell} python
remove_pixel_27 = True
```

```{important}
The inclusion of data taken from pixel 27 can degrade the overall spectral resolution
achievable by the Resolve instrument, you should exclude it from your analysis. You
must also remember to exclude pixel 27 from the pixel list or the detector region when
generating RMFs and ARFs (we implement this in [the RMF](#producing-redistribution-matrix-files-rmfs)
and [the ARF](#calculating-ancillary-response-files-arfs) generation sections of this demonstration).
```

### Excluding pixel-pixel coincident events

In this part of the demonstration we're going to discuss a whole category of events that you
should probably remove from your analysis - **pixel-pixel coincident events**.

Pixel-pixel coincident events are, unsurprisingly, events whose detection occurs almost
simultaneously with the detection of another photon in a different XRISM-Resolve pixel.

Multiple events being recorded at or around the same time by multiple pixels is not
inherently a bad thing, and of course, it will be more likely to happen the higher the
flux of your source.

Unfortunately, we often have to err on the side of caution and decide not to trust these
events, as not only are there physical processes **other than the normal arrival of
photons** that can induce XRISM-Resolve pixels to register events near simultaneously, but
even two photons happening to arrive at the same time can, in some cases, cause the
measured properties of the events to be biased.

Amongst their other functions, the various stages of processing applied to XRISM-Resolve data
attempt to identify pixel-pixel coincident events and to determine what caused them – the
results of those searches (alongside other the results of other checks) are stored in
the **STATUS** column of XRISM-Resolve event lists, with one entry per event.

```{code-cell} python
# TODO POTENTIAL DEMO OF GETTING THE EVENTS WHICH HAVE A STATUS[4] FLAG (I KNOW
#  IT SAYS 3 BUT INDEXING IS DIFFERENT)
cur_evt_list.data[cur_evt_list.data["STATUS"][:, 3]]
```

The **STATUS** column entry for a particular event is a *16-bit flag* (with
14 bits actually in use), and each bit represents the result of a different type of
processing check performed for the event. The different flags are described in the XRISM ABC guide
([XRISM GOF & SDC 2024](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html#SECTION00770000000000000000)).

If a bit value is **b0**, that means that the flag which that bit represents
was **not** raised (a good thing), whereas **b1** indicates that the flag _was_ raised.

So, we will be able to remove those events that might be problematic pixel-pixel
coincidences by making cuts based on the value of several of the bits in the **STATUS** column.

```{important}
When using HEASoft tools, the indexing of the STATUS column's flags **begins at 1**, unlike
in Python, which has zero based indexing.
```

We recommend excluding all events with a **STATUS[4]** flag raised - this is a generic check
for event coincidence and will help to filter out the majority of events that might
not have trustworthy properties.

```{code-cell} python
apply_general_coincident_screen = True
```

#### Frame events

The first type of coincident events we'll deal with are 'frame events' – they
occur when a significant amount of energy is absorbed into the **silicon frame *around*
the XRISM-Resolve array**

Absorption of enough energy into the frame will measurably 'pulse' the temperature of
the array's heat sinks, which in turn pulses the temperature of the pixels
themselves. This is a type of 'thermal cross-talk'; we will discuss the related concept
of 'electrical cross-talk' [the next section](#electrical-cross-talk).

Given that the pixels are microcalorimeters (glorified thermometers), you can see how
that might then affect the detection of incident photons, and the quantification of
their energy.

Indeed, for very large depositions of energy into the frame (on the scale of MeV),
the resulting pixel temperature pulses can produce signals that trigger the
'Pulse Shape Processor' (PSP), and result in **false events being recorded**.

If this does happen, the resulting false events tend to be clustered in time (because they occur
when the energy is deposited in the frame), and are referred to as 'frame events'.

Thankfully, there is a fairly easy way to identify frame events – they normally have a
significantly different 'pulse rise time' than 'normal' events do. The pulse rise time
describes how long it takes for a pulse to reach its peak and is stored in the
'RISE_TIME' column of a XRISM-Resolve event list (in units of $20\:\rm{\mu s}$).

Frame events can be effectively removed by applying a XRISM-Resolve-team-defined pulse
rise time cut to the event list, selecting events that fulfil these criteria:
>**The _RISE_TIME_ summed with _DERIV_MAX_ multiplied by a constant factor (currently set to 0.00075)
> should be between **46** and **58** (non-inclusive).**

Where _DERIV_MAX_ is the maximum time derivative of the pulse's rise – it can act as a
proxy for the pulse height.

<span style="color:red">***Would really like to include exactly what drove these choices.***</span>

We also note that low-resolution secondary events (Ls) have a very large spread of rise times, and thus
should not be considered when you perform this particular cut. That is to say, an event can be selected if it is
EITHER a non-Ls grade event and fulfills the above criteria, OR it is a Ls grade event.

Drawing on the event list we produced [when we ran the XRISM pipeline earlier in this demonstration](#running-the-xrism-pipeline-for-resolve),
we can produce histograms showing the distribution of _RISE_TIME_ for several subsets of events:
- **Low-resolution secondary (Ls; grade 4) events**
- **High-resolution primary (Hp; grade 0) to low-resolution primary (Lp; grade 3) events**
- **Hp-Lp events with the recommended _RISE_TIME_ cut applied**.

The total distribution of selected events would be the sum of the histograms labeled **Selected Hp—Lp** and
**Ls** (the wide distribution of Lp rise times that we mentioned earlier is quite noticeable here):

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(5.5, 5.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

rise_time_bins = np.linspace(
    cur_evt_list.data["RISE_TIME"].min(), cur_evt_list.data["RISE_TIME"].max(), 100
) * Quantity(20, "microsecond")

non_low_sec_data = cur_evt_list.data[cur_evt_list.data["ITYPE"] < 4]
non_low_sec_rise_times = non_low_sec_data["RISE_TIME"]

low_sec_data = cur_evt_list.data[cur_evt_list.data["ITYPE"] == 4]
low_sec_rise_times = low_sec_data["RISE_TIME"]

sel_rise_time_mask = (
    (non_low_sec_data["RISE_TIME"] + (0.00075 * non_low_sec_data["DERIV_MAX"])) > 46
) & ((non_low_sec_data["RISE_TIME"] + (0.00075 * non_low_sec_data["DERIV_MAX"])) < 58)
rise_time_sel_data = non_low_sec_data[sel_rise_time_mask]

plt.hist(
    (low_sec_rise_times * Quantity(20, "microsecond")).to("millisecond"),
    bins=rise_time_bins,
    color="crimson",
    label="Ls",
    alpha=1,
    histtype="step",
    lw=1.8,
)

plt.hist(
    (non_low_sec_rise_times * Quantity(20, "microsecond")).to("millisecond"),
    bins=rise_time_bins,
    color="navy",
    alpha=0.7,
    label="Hp—Lp",
    histtype="step",
    lw=1.8,
    hatch="///",
)

plt.hist(
    (rise_time_sel_data["RISE_TIME"] * Quantity(20, "microsecond")).to("millisecond"),
    bins=rise_time_bins,
    color="mediumturquoise",
    alpha=0.8,
    histtype="stepfilled",
    label="Selected Hp—Lp",
)

plt.xlabel("Rise Time [ms]", fontsize=15)
plt.ylabel("N", fontsize=15)

plt.legend(loc="best", fontsize=14)
plt.tight_layout()
plt.show()
```

Here we are interacting with the event list through the `XGA` module's EventList class, but only for convenience, they
cannot yet be directly used to produce XRISM data products.

As a slight aside, we can also compare the PI distributions of the Hp-Lp events that passed our rise time cut to the
subset of Hp-Lp events that did not. You'll note that the distribution of PI values for the **cut events** has a
peak at between 45000–50000 (corresponding to $\sim\:22.5-25.0$ keV); many of these events have been assigned
high energy values, which fits with them being the product of massive energy deposition into the frame:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(5.5, 5.0))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

pi_bins = np.linspace(cur_evt_list.data["PI"].min(), cur_evt_list.data["PI"].max(), 80)

rise_time_removed_data = non_low_sec_data[~sel_rise_time_mask]

plt.hist(
    rise_time_sel_data["PI"],
    bins=pi_bins,
    color="teal",
    alpha=0.8,
    histtype="stepfilled",
    label="Selected Hp—Lp",
)
plt.hist(
    rise_time_removed_data["PI"],
    bins=pi_bins,
    color="goldenrod",
    alpha=0.8,
    histtype="stepfilled",
    label="Cut Hp—Lp",
)

plt.xlabel("PI", fontsize=15)
plt.ylabel("N", fontsize=15)

plt.legend(loc="best", fontsize=14)
plt.tight_layout()
plt.show()
```

We recommend that you apply these rise time cuts in your own analysis. We have created an all-in-one
event list cleaning function for this demonstration, which will optionally apply the various
checks and screen we're discussing at the moment. Here we set the variable that will control
whether the rise time screening is applied [when we call the function](#making-new-cleaned-event-lists):

```{code-cell} python
apply_rise_time_screen = True
```

#### Electrical cross-talk

<span style="color:red">***The ABC guide 'screening out pixel-pixel coincident events' guide mentions
that 'electrical cross-talk screening with status 6 is not recommended'. The status definition for [6] is
coincidence with pixel 12 event	(status 5) & passed energy test for absorption of electron ejected from 12. I
guess that sounds like a form of electrical cross-talk, but status 7's description is "candidate electrical
crosstalk event or its source", which sounds even more relevant? Have to ask the XRISM team.***</span>

We can now move on from the world of photons causing problems to the world of electrons causing problems.

Every XRISM-Resolve pixel is a separate microcalorimeter detector, read out through a dedicated
signal path (though some processing electronics are used by multiple pixels). The signal read-out
paths (i.e. wires) for a XRISM-Resolve pixel will at some points be in close proximity to
those of another pixel (or pixels) – an unfortunate necessity driven by the overall design of the
Resolve array.

It is possible for the voltage pulse produced by a XRISM-Resolve pixel detecting an incident photon to
induce a much smaller pulse in the read-out path of an _electrically adjacent_ pixel, through capacitive
coupling. This is called **electrical cross-talk**, and it can degrade the energy resolution
achievable by Resolve – it is primarily high count-rate observations that are affected, mind you; this
type of cross-talk is likely to be less of a problem for faint sources.

Unlike the 'frame events' we discussed in [a previous section](#frame-events), the input
induced by electrical cross-talk does not usually trigger the pulse shape processor (PSP), and thus
is not generally recorded as a (false) event.

```{note}
A Resolve pixel's read-out path is not necessarily _electrically adjacent_ to those of its
direct neighbors.
```

However, electrical cross-talk induced by an electrically adjacent pixel _can_ contaminate a **real**
signal pulse that has occurred at around the same time - this then corrupts the eventual
measurements of the pulse's characteristics, and everything inferred from them (including the energy
assigned to the event).

The most effective way to deal with electrical cross-talk is to **exclude any events
originating from electrically adjacent pixels that occur near simultaneously**. You
don't have to figure out which events meet that particular criteria for yourself, as the
**STATUS** column that we discussed at the
[beginning of the 'excluding pixel-pixel coincident events' section](#excluding-pixel-pixel-coincident-events)
contains entries with that information:
- **STATUS[13]** being set to **b1** (or True) indicates that the event is likely to be contaminated by an electrical cross-talk signal that was not powerful enough to be recorded as an event (untriggered electrical cross-talk).
- <span style="color:red">**STATUS[6]** (or is it 7?)</span> being set to **b1** (or True) indicates that the event _may_ be the result of an electrical cross-talk signal powerful enough to be recorded as an event.

```{caution}
We do not currently recommend using <span style="color:red">**STATUS[6]** (or is it 7?)</span> to remove
electrical cross-talk *events*, as the strongly recommended **STATUS[4]** cut discussed at the
[beginning of the 'excluding pixel-pixel coincident events' section](#excluding-pixel-pixel-coincident-events)
will achieve much the same result.
```

We recommend that, if you are analyzing a bright source, you explore what effect excluding events
flagged as potentially contaminated by untriggered electrical cross-talk has on the spectra you
produce and the measurements you make from them. <span style="color:red">***Add sentence describing whatever
default we actually go with below***</span>

```{code-cell} python
apply_unfiltered_coincident_screen = False
```

### Excluding periods of high particle background flux

Moving on from pixel-pixel coincidence, another cleaning step that you could _potentially_ apply
to your data is to exclude all events recorded during periods of the observation
that had particularly high particle background fluxes.

We say *potentially*, because practically speaking, you might not need to worry about
this if your observation is of a bright source.

If your source of interest **is relatively faint**, however, you should consider comparing
your output spectra and measured properties with and without this extra step, and make a
decision based on the source flux and your particular science goals. We cannot give a
blanket recommendation to apply this to all faint sources, because you can end up
dramatically decreasing the overall exposure time of your observation, and every
kilo-second is precious. So, if removing the background particle flux does not improve
your source's signal-to-noise, the trade-off may not be worth it.

Now that we've philosophized about whether you _should_ apply this step, we can move on to how
it works. XRISM traces the particle background level by exploiting its inverse correlation with the
'geomagnetic cutoff rigidity' (COR), when in low Earth orbits.

The cut-off rigidity itself is a property of the Earth's geomagnetic field and is a measure
of how much the magnetic field 'shields' XRISM from cosmic rays
([Smart D. F., Shea M. A. 2005](https://ui.adsabs.harvard.edu/abs/2005AdSpR..36.2012S/abstract)), as a function of the spacecraft's location. It is
not measured directly by XRISM, but rather is calculated from the **2020 version of
the 'International Geomagnetic Reference Field' (IGRF-13) model** ([Alken P. et al. 2021](https://ui.adsabs.harvard.edu/abs/2021EP&S...73...49A)).

All the COR information recorded for a given XRISM observation is stored in the
'extended housekeeping file' (EHK); there are several COR-related
columns – _COR_, _COR2_, _COR3_, and ***CORTIME***. We will use the ***CORTIME***
column, as it is calculated from the model mentioned above, whereas the other estimates
of COR are based on older maps and models.

So, to exclude periods of high particle background flux, we want to select events from
only those periods of the observation that have a COR value **above a certain threshold** (recall
that higher COR means lower particle background flux).

We recommend selecting time periods that have a **CORTIME value of $>8$** (though you can experiment
with this threshold and observe the effect of it on your data).

The housekeeping file does not store a COR value for every single event; instead, all the
parameters that it keeps track of are recorded on a regular time cadence.

<span style="color:red">***ACTUALLY I DON'T UNDERSTAND WHY WE CAN'T JUST USE MAKETIME?***</span>><>


Thus, the most effective way of selecting events based on the COR information stored in the EHK
is to generate a new 'Good Time Interval' (GTI) file that can be applied during event list
cleaning. We will use two HEASoft tools to achieve this:
1. `makefilter` - Will be used to create an FTOOLS filter file that selects time steps when **CORTIME > 8**.
2. `maketime` - Converts the filter file created by `makefilter` into a GTI file that can be applied to the event list.

<span style="color:red">***NEED TO CHECK IF NEW FILTER FILE SHOULD BE COMBINED WITH EXISTING, AND HOW TO APPLY THE NEW GTI USING EXTRACTOR***</span>><>

<span style="color:red">***Makefilter requires a config file to describe which columns to copy into the filter file.***</span>><>

```{code-cell} python
# hsp.makefilter(configure=, noprompt=True)

# cur_ehk = os.path.join(OUT_PATH, cur_evt_list.obs_id, f"xa{cur_evt_list.obs_id}.ehk")
# print(cur_ehk)
#
# hsp.maketime(infile=cur_ehk,
#              outfile='testo-gti.fits',
#              expr="CORTIME.gt.8",
#              noprompt=True)
```

### Selecting events within PI limits

When creating event lists that will be used to generate data products such as spectra, images,
light curves, etc. we recommend selecting the widest possible energy range (or rather, PI
channel range, as is stored in event lists) - in most cases it is best to select the entire valid
energy range of the instrument, it provides the most flexibility.

With that said, you will find that XRISM-Resolve event lists (like most X-ray instruments) records
events that are outside the viable range of the detector, so we do need to apply _some_ PI
filtering - indeed the second figure in [the 'frame events' section](#frame-events) highlights
that many events have been recorded with a PI of zero. Definitely not physical.

The exact PI filtering applied will vary depending on your science case - we do not recommend setting
a limit any lower than PI=600, as excluding those very low energy events helps, in concert with other
screening methods (such as those discussed in [the electrical cross-talk section](#electrical-cross-talk))
to screen out coincident events.

Our upper limit is set to PI=20000, which corresponds to 10 keV - XRISM-Resolve is not currently
well calibrated at energies much higher than this.

```{code-cell} python
# Define lower and upper PI channel limits for product extraction
pi_chan_limits = Quantity([600, 20000], "chan")

# Show the PI limits as energies, for context
print((RSL_EV_PER_CHAN * pi_chan_limits).to("keV"))
```

```{important}
The lower bound of XRISM-Resolve's effective energy range is currently limited to
approximately **1.7 keV**, as the **gate valve** that protected the XRISM-Resolve instrument
during launch failed to open. This gate valve is a highly effective absorber of low
energy X-rays.
```

### Making new 'cleaned' event lists

<span style="color:red">***IF I MENTION AND GENERATE COR-BASED GTIS, I NEED TO BE ABLE TO PASS THEM TO THE FUNCTION BELOW***</span>

```{code-cell} python
arg_combs = [
    [
        EVT_PATH_TEMP.format(oi=oi, xrf=xf),
        os.path.join(OUT_PATH, oi),
        *pi_chan_limits,
        apply_general_coincident_screen,
        apply_rise_time_screen,
        apply_unfiltered_coincident_screen,
        remove_pixel_27,
    ]
    for oi, xfs in cut_rel_filters.items()
    for xf in xfs
]

with mp.Pool(NUM_CORES) as p:
    sev_result = p.starmap(screen_xrism_resolve_evts, arg_combs)
```

### There are considerations for extended sources

<span style="color:red">***THIS DOESN'T BELONG HERE, AND DOESN'T MATCH WHAT I ORIGINALLY THOUGHT I'D WRITE HERE***</span>

The source we are using for our example, NGC 1365, is a point source. However, one of
XRISM-Resolve's unique capabilities is that of performing **spatially resolved** very high energy
resolution (high resolution for X-ray observations at least) for extended sources.

Other high energy resolution X-ray instruments use fundamentally different technologies to
the microcalorimeters that make up XRISM-Resolve, they are typically dispersive grating
spectrometers.

It is extremely difficult to perform spatially resolved spectroscopic
analyses with such instruments, and they are poorly suited to the observation
of extended sources. Such observations of tend to run afoul of instrumental line broadening
effects, rendering the derivation of well constrained line widths much harder.

All this is to say that many XRISM-Resolve observations will be of extended sources, and while
they will provide many measurements impossible with previous missions, their analysis
is far more complex, with extra considerations, and we will not demonstrate it in this notebook.


## 4. Generating new XRISM-Resolve images and exposure maps

At this point we have processed the raw XRISM-Resolve data, and then applied various cleaning
steps to remove anomalous or unhelpful events. From here on we will produce data products
useful for analysis of the observed source.

### Setting up for image generation

We start by producing XRISM-Resolve images, within specified energy bands. Due to the
very low spatial resolution of XRISM-Resolve, they will appear neither spectacular nor
particularly informative, but they will at least allow us to ascertain if most of
the emission is concentrated in the central few pixels, as we expect for a well-targeted
point source.

First, we decide which energy bounds we wish to generate images within. Those we
choose here have no particular meaning, but in order to demonstrate the generation
of multiple images from multiple energy bands in parallel, we define two.

You can easily adjust these limits, if you're using this notebook as a template or a
basis for your own analysis - this Astropy Quantity is a set of lower and upper
bounds, and will result in images between 3.0-10.0 keV and 6.0-7.0 keV being
generated. If you wish to specify a single energy band, simply define the variable
as `Quantity([[3.5, 5.5]], "keV")`.

```{code-cell} python
# Define pairs of lower/upper energy bounds within which to generate images
im_en_bounds = Quantity([[3.0, 10.0], [6.0, 7.0]], "keV")
```

Next, we decide which grades of event to include in the final images. In our case
we choose to include all grades, which is **recommended for the estimation of
source flux from images**.

<span style="color:red">That last may not be right?</span>

If you are modifying this demonstration and wish to define which grades
should be used, you may set the variable to a list of integer grade identifiers (e.g.
`im_evt_grades = [0, 1]` for high-resolution primary and medium-resolution primary events).

```{code-cell} python
im_evt_grades = None
```

### Running image generation

We have implemented a convenient function to generate XRISM-Resolve images in the
['Global Setup: Functions'](#functions) section near the beginning of this notebook.

It makes use of the HEASoft `extractor` task behind the scenes, and is designed to be
easily run in parallel, which is what we will be doing below.

As the event lists we're using have already been screened for anomalous events (see
[the previous section](#3-choosing-the-events-to-consider-for-data-product-generation)),
we just need to pass the variables defined in
[the 'setting up for image generation' subsection](#setting-up-for-image-generation) to
the image generation function.

This will parallelize image generation so that different combinations of ObsID, X-ray
filter, and specified energy bounds are run simultaneously across as many cores as
are available (though by default this demonstration only uses one ObsID and one filter):

```{code-cell} python
arg_combs = [
    [
        SCR_EVT_PATH_TEMP.format(oi=oi, xrf=xf),
        os.path.join(OUT_PATH, oi),
        *cur_bnds,
        im_evt_grades,
    ]
    for oi, xfs in cut_rel_filters.items()
    for xf in xfs
    for cur_bnds in im_en_bounds
]

with mp.Pool(NUM_CORES) as p:
    sp_result = p.starmap(gen_xrism_resolve_image, arg_combs)
```

### Setting up for exposure map generation

Exposure maps describe the effective exposure of each pixel in the XRISM-Resolve
array, and are a prerequisite for the generation of ancillary response files ([which we
will be doing in a later section](#calculating-ancillary-response-files-arfs)).

Exposure maps are also a useful way to tell exactly which parts of the sky
are covered by the observation.

Unlike for image creation, there is a dedicated HEASoft task for the generation of
XRISM exposure maps; `xaexpmap` - just as we did with image generation in
[the last subsection](#running-image-generation), we have set up a wrapper function
for this task in the ['Global Setup: Functions'](#functions) section near the beginning
of this notebook, allowing us to easily run generation of different exposure maps in
parallel.

There are two `xaexpmap` configuration options which control how the
attitude (essentially where the telescope is pointing) of XRISM over the course of
the observation is binned spatially. These bins ('off-axis wedges' as the
[`xaexpmap` documentation](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xaexpmap.html)
describes them) are where the initial 'time intervals' of observation coverage are calculated:
- **Radial Delta** - Passed to `xaexpmap` as `delta`. Radial increment (in arcmin) for the annular grid for which the attitude histogram will be calculated. The annuli are centered on the optical axis (off-axis angle = 0), and the central circle has a radius equal to `delta`.
- **Number of azimuthal bins** - Passed to `xaexpmap` as `numphi`. Number of azimuth (phi) bins in the first annular region over which attitude histogram bins will be calculated (i.e., this annular region lies between `delta` and 2*`delta` arcmin from the center of the annuli). The zeroth annular region is a full circle of radius `delta` and the nth annular region has an outer radius of (n+1)*`delta`, and `numphi`*n azimuthal bins.

The documentation for `xaexpmap` notes that you can force the attitude histogram to have a single bin, by choosing a radial delta that is much larger than any expected attitude variation during an observation.

We choose to create exposure maps from only one attitude histogram bin, by passing a large radial delta and requiring a single azimuthal bin:

```{code-cell} python
expmap_rad_delta = Quantity(20, "arcmin")
expmap_phi_bins = 1
```

### Running exposure map generation

Now to run our exposure map generation, parallelizing over all relevant ObsIDs
and X-ray filters – note that this will only produce a single exposure map for
the default configuration of this demonstration notebook, as we selected a single
ObsID and filter earlier on, though it will handle having multiple ObsIDs/filters
selected.

```{code-cell} python
arg_combs = [
    [
        SCR_EVT_PATH_TEMP.format(oi=oi, xrf=xf),
        os.path.join(OUT_PATH, oi),
        None,
        expmap_rad_delta,
        expmap_phi_bins,
    ]
    for oi, xfs in cut_rel_filters.items()
    for xf in xfs
]

with mp.Pool(NUM_CORES) as p:
    ex_result = p.starmap(gen_xrism_resolve_expmap, arg_combs)
```

### Visualizing a new image

Our final act of this section is to take a quick look at one of the XRISM-Resolve images
that we just generated, with the pixel IDs overlaid on top. That's going to give us some
context when we define the region file that we want to extract spectra from.

Firstly, we load a just generated image into an XGA Image instance (just because there
are useful inbuilt visualization capabilities). Which exact image we've selected isn't particularly
important in this case.

```{code-cell} python
cur_im_path = IM_PATH_TEMP.format(
    oi=rel_obsids[0],
    xrf=cut_rel_filters[rel_obsids[0]][0],
    ibf="PIX",
    lo=im_en_bounds[0][0].to("keV").value,
    hi=im_en_bounds[0][1].to("keV").value,
)

cur_im = Image(cur_im_path, rel_obsids[0], "Resolve", "", "", "", *im_en_bounds[0])
```

Usually we would just call `cur_im.view()` to make a visualization of the image, but in this
case we also want to overlay the pixel IDs, so we will fetch the base visualization produced
for the Image view method, by calling `cur_im.get_view()`, and then modify the figure before
displaying it:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(6.7, 5.5))
cur_ax = plt.gca()
cur_im.get_view(
    cur_ax, zoom_in=True, manual_zoom_xlims=[-0.5, 5.5], manual_zoom_ylims=[-0.5, 5.5]
)

for pix_id, det_reg_str in RESOLVE_PIX_DET_REGIONS.items():
    cur_pix_detxy = np.array(det_reg_str.strip("box(").split(",")[:2]).astype(int) - 1

    pix_txt_col = "black" if pix_id != 27 else "snow"

    plt.text(
        x=cur_pix_detxy[0],
        y=cur_pix_detxy[1],
        s=str(pix_id),
        color=pix_txt_col,
        fontsize=20,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )

cbar = plt.colorbar(cur_ax.images[0])
cbar.ax.set_ylabel("ct", fontsize=15)

plt.tight_layout()
plt.show()
```

So now we can easily find each pixel's ID, and from there decide whether to include
or exclude it from our spectral extraction.


## 5. Generating new XRISM-Resolve spectra

Generating and analyzing high-resolution X-ray spectra is the most likely reason for you
to use XRISM-Resolve data, so the next data products we create are spectra and all the
supporting files required for their analysis.

### Defining the extraction region/pixels

As is the case for generation of spectra from any high-energy mission's data, the first
decision we have to make is what spatial region the events used to create the spectrum
are to be extracted from.

In this demonstration we're only addressing the analysis of point sources, so we don't
have to concern ourselves with splitting the observation of NGC 1365 into multiple
spatial regions. Instead, we're going to make a simple 'global' spectrum by using
events from **all valid pixels**.

We define a list below that includes every valid pixel ID, excluding
**pixel 12** [as it is a calibration pixel](#pixel-12-is-a-dedicated-calibration-pixel) and
**pixel 27** [as it is broken](#pixel-27-of-xrism-resolve-is-broken).

```{code-cell} python
chosen_pixels = [pix_id for pix_id in range(0, 36) if pix_id not in [12, 27]]
chosen_pixels
```

### Setting up region files

We have defined a simple convenience function called `det_region_from_pixels` in the
[Global Setup: Functions](#functions) section near the top of this notebook. As you
may have gathered from its name, the function takes a list of XRISM-Resolve pixel IDs
and uses them to produce a region file **in detector coordinates**, selecting every pixel
specified.

This is useful because HEASoft tool we use for spectral extraction wants
the input region files to be in detector coordinates, and because there are so few
pixels in the XRISM-Resolve array, we can often just look at a visualization (such as
[was shown at the end of the last section](#visualizing-a-new-image)) and determine
which pixel IDs we want to use.

```{code-cell} python
chosen_pixel_det_reg_path = os.path.join(OUT_PATH, "chosen_pixel_detxy.reg")
det_region_from_pixels(chosen_pixel_det_reg_path, chosen_pixels)
```

### Choosing which event grades to include

We [discussed XRISM-Resolve event grading in a previous section](#event-grades-and-branching-ratios), and you will
ultimately have to make your own choices as to which are appropriate to include in spectrum extraction for
your science case.

Here, we are only going to use the very highest resolution events, and we generally recommend that you do the same if
your observation is of a high enough SNR. The list definition below will be passed to a convenience
function for XRISM-Resolve spectral generation and should include at least one entry of an integer
grade ID (e.g. the ITYPE column in XRISM-Resolve event lists):

```{code-cell} python
chosen_evt_grades = [0]
```

### Generating spectral files

As with the generation of images and exposure maps, we have set up spectrum generation in
this notebook so that extraction from different ObsID-filter combinations can be performed
in parallel, across as many cores as are available.

Also, as with those previous data product generation steps, we are not making full use
of the parallelization capability as early on in this tutorial we selected as a single
ObsID and a single filter to minimize the default run time. If you decide to alter this
notebook to use multiple ObsIDs/filters, then more cores will be utilized.

```{code-cell} python
arg_combs = [
    [
        SCR_EVT_PATH_TEMP.format(oi=oi, xrf=xf),
        os.path.join(OUT_PATH, oi),
        chosen_evt_grades,
        chosen_pixels,
    ]
    for oi, xfs in cut_rel_filters.items()
    for xf in xfs
]

with mp.Pool(NUM_CORES) as p:
    sp_result = p.starmap(gen_xrism_resolve_spectrum, arg_combs)
```

The return from the `gen_xrism_resolve_spectrum` function includes the following information:
- HEASoft logs of the spectral extraction process.
- File path of the newly generated spectrum.
- File path to the source event list (this is for convenience when generating the spectra's ancillary files).
- The ObsID that the spectrum is associated with.
- The X-ray filter that the spectrum is associated with.

As we are using a multiprocessing pool to enable the generation of multiple spectra in parallel, all
returns from `gen_xrism_resolve_spectrum` are stored in the `sp_result` list. This list
has one entry per function call, and is in the same order as the `arg_combs` variable.

### Producing redistribution matrix files (RMFs)

There are two crucial ancillary files required to analyze high-energy spectra – the first
is the redistribution matrix file (RMF). This is what describes the mapping between
detector channel and incident photon energy, including uncertainties introduced by
the fact that no detector (or its electronics) is entirely perfect.

Without this, we would be only be able to deal with spectra in terms of the channel,
rather than energy, assigned to an event – this would essentially remove our ability
to draw physical conclusions about the origin of the emission.

XRISM-Resolve RMFs are uniquely large (in terms of storage and memory use) because of
the very high energy resolution of the detector – higher resolution means more detector
channels, which not only increases the length of a one table in the RMF file, but dramatically
increases the size of the 'matrix' part of the RMF.

All that said, anything that involves producing, reading, or using an RMF is going to
take longer than you are used to, if you have never used XRISM-Resolve data before.

With that in mind, the HEASoft task used to generate the RMFs for Resolve allows the
user to select the 'size' of RMF, with the different sizes having different levels
of complexity in their modeling of the detector channel to energy mapping:
- **Small [S]** – Suitable **only for analysis development**, but not final scientific results, this RMF only models the Gaussian core of the line broadening function.
- **Medium [M]** – Additionally includes the exponential tail and Si K alpha emission line.
- **Large [L]** – Suitable for some scientific analyses, this also includes the escape peaks.
- **X-Large [X]** – Includes the electron loss continuum in addition to every other effect.

The line spread function is discussed in the proposer's observatory guide
[XRISM GOF & SDC 2026](https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html#sec:resolve_LSF).


Larger RMFs reproduce the XRISM-Resolve response more accurately, but unfortunately increase the fitting
time required for each spectrum. As such we recommend that exploratory model fits, and the development
of your analyses, make use of 'small' or 'medium' RMFs, and that larger RMFs are only used when
you are sure of your fitting setup.

The X-large class of XRISM-Resolve RMF often produces files with sizes in excess of 7GB. Consider
the implications of that in terms of storage use, and in memory use when loaded into XSPEC, before
utilizing them. We **strongly** recommend that X-large RMFs are split into multiple files using
the `splitrmf` and `splitcomb` arguments of the `rslmkrmf` task, though the convenience function
for RMF generation that we are about to use does not support this.

For demonstrative purposes, we will use the **small** RMF type:

```{code-cell} python
chosen_rmf_size = "S"
```

```{seealso}
See the [HEASoft `rslmkrmf` help file](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/rslmkrmf.html), specifically
the parts about the 'whichrmf' parameter, for more information on RMF sizes.
```

```{code-cell} python
arg_combs = [
    [
        sp_gen_output[2],
        sp_gen_output[1],
        os.path.join(OUT_PATH, sp_gen_output[3]),
        chosen_rmf_size,
    ]
    for sp_gen_output in sp_result
]

with mp.Pool(NUM_CORES) as p:
    rmf_result = p.starmap(gen_xrism_resolve_rmf, arg_combs)
```

```{caution}
We **do not** recommend using 'small' XRISM-Resolve RMFs for the measurement of final
scientific results, but they are very useful for initial exploration and preparation
of your analyses.
```

### Calculating ancillary response files (ARFs)

```{danger}
The HEASoft task we use to generate ARFs is called **`xaarfgen`**. There is
another, very similarly named, HEASoft tool related to the construction of XRISM
ARFs, **`xaxmaarfgen`**. Be sure which one you are using!
```

ARFs are the final type of supporting file required to make our spectra usable and
describe the effective area (i.e., the sensitivity) of XRISM-Xtend as a function of
energy.

The effective area has to be understood (and well calibrated) as we need it to help
map a spectral model, which hopefully describes what the object of interest
is _actually_ emitting (and how), to the _observed_ spectrum; that observed spectrum
has been altered across its energy range by how good XRISM-Resolve is at detecting
photons at different points in that range.

The sensitivity of an X-ray detector is a combination of the X-ray optic's (on XRISM
this is the called X-ray Mirror Assembly, or XMA) effective area and the detector's
quantum efficiency. They are both independently a function of energy.

ARFs are standard products for most high-energy missions, but the methods implemented
to calculate them for XRISM's instruments are quite unusual.

The HEASoft task we need to call (`xaarfgen`) calls further HEASoft tools that perform
ray-tracing simulations of XRISM XMAs, for the location of your source on the
detector, and use those to define the X-ray optic's collecting area for a wide range
of energies.

```{note}
If you have to generate multiple ARFs for the same source, in the same observation, you
should be aware that the raytraced event lists can be re-used (though only in this
particular scenario).
```

Raytracing can be a slow process, as individual events and their path through the
XMA are being simulated, but it does help to produce very accurate ARFs. There are ways
that it can be sped up, though at the cost of that accuracy – the most direct way is
to limit the number of events that are simulated.

Rather than setting an overall number of events to simulate, the `xaarfgen` task provides
an argument ('numphoton') to set the number of photons allocated to each attitude
histogram bin (in the exposure map file), per grid point in the internal energy grid.

An argument specifying the number of events ('numphoton') can be passed to `xaarfgen`, and for
our demonstration we are going to use a very small sample - this is primarily so the
notebook can run in a reasonable amount of time.

A second argument, `minphoton`, specifies the minimum acceptable number of raytracing photons that
successfully reach the focal plane for each raytracing energy grid point. If that minimum number is
not reached for each energy grid point during the raytracing process, ARF production will fail.

The `xaarfgen` documentation provides the following guidance on choosing the number of
events to simulate:

```{seealso}
Note that even if `minphoton` is exceeded at all energies, this does not guarantee
that the resulting ARF is robust and sufficiently accurate.

In general, about 5000 or more photons per energy (over the extraction region) give
good results, but the actual minimum number varies case-by-case, and fewer may be
sufficient in some cases.

The default value of `minphoton` is deliberately very small, in order that the
ARF is made and available for diagnostic evaluation. In general, it is not
recommended to set `minphoton` to a high value in the first place, because it is
not possible to reliably estimate what `minphoton` should be in advance of
running raytracing within `xaarfgen`, in order for that value of 'photon' to be
satisfied for all energies, which could result in repeated failures after very long
run times. It could also run into memory problems and/or a raytracing file size that
is unmanageable.
```

We choose the default values for both the `minphoton` and `numphoton` arguments:

```{code-cell} python
arf_rt_num_photons = 20000
arf_rt_min_photons = 100
```

So now we move onto actually running the ARF generation – using the
`gen_xrism_resolve_arf` function defined in the Global Setup: Functions section (near the top of
the notebook), which wraps the HEASoftPy interface to the `xaarfgen` task. We now use it
to generate ARFs in parallel for all of our new spectra:

```{code-cell} python
arg_combs = []
for sp_gen_output in sp_result:
    oi = sp_gen_output[3]
    xf = sp_gen_output[4]

    args = [
        os.path.join(OUT_PATH, oi),
        SRC_COORD,
        EX_PATH_TEMP.format(
            oi=oi, xrf=xf, rd=expmap_rad_delta.to("arcmin").value, npb=expmap_phi_bins
        ),
        sp_gen_output[1],
        sp_gen_output[1].replace("-spectrum.fits", ".rmf"),
        chosen_pixel_det_reg_path,
        arf_rt_num_photons,
        arf_rt_min_photons,
    ]
    arg_combs.append(args)

with mp.Pool(NUM_CORES) as p:
    arf_result = p.starmap(gen_xrism_resolve_arf, arg_combs)
```

### Grouping our newly generated spectra

We also need to group the spectra [we just generated](#generating-spectral-files). Grouping
essentially combines spectral channels until some minimum quality threshold is reached. We
use the HEASoft `ftgrouppha` tool to do this, once again through HEASoftPy.

Various quality metrics can be used; for instance, it is quite common to group high-energy
spectra so that there is at least one count per channel, in order to make the use of the
Cash statistic during spectral fitting valid.

In this case we select the 'optmin' binning technique, which implements the optimum binning
method described by [Kaastra J. S. and Bleeker J. A. M. (2016)](https://ui.adsabs.harvard.edu/abs/arXiv:1601.05309), while
also including a requirement for a minimum number of counts per channel (10 in this case).

```{code-cell} python
spec_group_type = "optmin"
spec_group_scale = 10
```

We do not parallelize the grouping of spectra, as it is a fairly computationally
inexpensive task. However, if you are dealing with many spectra you may wish to implement
a multicore version, taking the parallelized functions we have implemented in this
notebook as a template.

This loops through the spectra produced in the ['generating spectral files'](#generating-spectral-files)
subsection and applies the grouping, while also writing the relative paths to RMF and ARF files
into each spectrum's header:

```{code-cell} python
grp_spec_paths = []
for sp_gen_ind, sp_gen_output in enumerate(sp_result):

    cur_sp_path = sp_gen_output[1]
    cur_rmf_path = rmf_result[sp_gen_ind][1]
    cur_arf_path = arf_result[sp_gen_ind][1]

    new_grp_sp_path = cur_sp_path.replace(
        "-spectrum", f"-{spec_group_type}grp{spec_group_scale}-spectrum"
    )

    hsp.ftgrouppha(
        infile=cur_sp_path,
        outfile=new_grp_sp_path,
        grouptype=spec_group_type,
        groupscale=spec_group_scale,
        respfile=cur_rmf_path,
        clobber=True,
        chatter=TASK_CHATTER,
        noprompt=True,
    )

    # Populate the RESPFILE and ANCRFILE headers
    with fits.open(new_grp_sp_path, mode="update") as speco:
        del speco["SPECTRUM"].header["RESPFILE"]
        speco["SPECTRUM"].header["RESPFILE"] = os.path.basename(cur_rmf_path)

        del speco["SPECTRUM"].header["ANCRFILE"]
        speco["SPECTRUM"].header["ANCRFILE"] = os.path.basename(cur_arf_path)

    grp_spec_paths.append(new_grp_sp_path)
```

## 6. Fitting a model with PyXspec

In this section we will perform a simple model fit to our new XRISM-Resolve
spectra – or rather, spectrum, as even though the rest of the demonstration is designed
to scale to the analysis of multiple observations/filters (we selected just one to
make the tutorial faster), this section will only handle a single spectrum.

```{code-cell} python
chosen_grp_sp_path = grp_spec_paths[0]
```

As you might imagine, spectral analysis of XRISM-Resolve data can be considerably more
complex than spectro-imaging CCD spectra. ***We defer a full exploration of more
in-depth spectral analysis to future demonstration notebooks.***

### Configuring PyXspec

Before we start using PyXspec in earnest, we configure some of its behaviors:
- _xs.Plot.xAxis = "keV"_ – Ensures that the x-axis of any plot data we retrieve from PyXspec is in energy, rather than channel, units.
- _xs.Fit.statMethod = "cstat"_ – Tells PyXspec to use the Cash statistic for model fits.
- _xs.Fit.nIterations = 1000_ – Sets the maximum number of iterations during a model fit.
- _xs.Fit.query = "no"_ – Disables PyXspec prompts asking whether to continue or not.

```{code-cell} python
xs.Plot.xAxis = "keV"
xs.Fit.statMethod = "cstat"

xs.Fit.nIterations = 1000
xs.Fit.query = "no"
```

### Loading the spectrum into PyXspec

Just loading a XRISM-Resolve spectrum (or more correctly, its response matrix) into
PyXspec can take a long time, **even for the 'small' Resolve RMF** that
[we generated in a previous section](#producing-redistribution-matrix-files-rmfs). Expect
it to take considerably longer to declare an XSPEC `Spectrum` instance when using the
larger RMF sizes.

You can see that we use the `chdir` context here to switch to the directory containing
our chosen spectrum, then set up the PyXspec `Spectrum` instance. Once complete, the
current working directory is then changed back to its original location.

```{code-cell} python
xs.AllData.clear()

with contextlib.chdir(os.path.dirname(chosen_grp_sp_path)):
    cur_sp = xs.Spectrum(os.path.basename(chosen_grp_sp_path))
```

### Initial visual examination of the spectrum

It's always a good idea to examine the newly generated spectrum, just as a validity check, and
to get an idea of what spectral features you have to work with/might find interesting.

Here we use the PyXspec plot manager to set up the data arrays necessary for a visualization
of the spectrum, prior to any models being fit or energy limits being applied. We then
retrieve the relevant data from PyXspec and store it in a dictionary, for easy access later:

```{code-cell} python
xs.Plot("data")

spec_plot_data = {
    "energy": np.array(xs.Plot.x()),
    "energy_delta": np.array(xs.Plot.xErr()),
    "rate": np.array(xs.Plot.y()),
    "rate_err": np.array(xs.Plot.yErr()),
}
```

You may find the '{doc}`PyXspec basics <../../useful_high_energy_tools/pyxspec/finding_relevant_heasarc_catalog>`' tutorial useful.

```{seealso}
THIS IS A CHECK TO SEE IF IT WORKS IN AN ADMONITION
You may find the '{doc}`PyXspec basics <../../useful_high_energy_tools/pyxspec/finding_relevant_heasarc_catalog>`' tutorial useful.
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

plt.errorbar(
    spec_plot_data["energy"],
    spec_plot_data["rate"],
    xerr=spec_plot_data["energy_delta"],
    yerr=spec_plot_data["rate_err"],
    fmt="+",
    color="navy",
    label="XRISM-Resolve data",
)

plt.xscale("log")

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
ax.xaxis.set_minor_formatter(
    FuncFormatter(lambda inp, _: "{:g}".format(inp) if inp >= 1 else "")
)

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(
    r"Spectrum [$\frac{\rm{ct}}{\rm{s} \: \rm{cm}^{2} \: \rm{keV}}$]", fontsize=15
)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
```

### Constraining the continuum

This simple demonstration is not meant to be a comprehensive guide to fitting high-resolution
X-ray spectra, and the many pieces of work produced by the XRISM performance verification team
contain much more sophisticated approaches applicable to a range of different source types.

However, we will take a slightly more sophisticated approach than just fitting a single model
and calling it a day.

```{code-cell} python
xs.AllData.ignore("bad")
cur_sp.ignore("**-3. 6.-8. 10.0-**")
```

```{code-cell} python
pl_cont_mod = xs.Model("powerlaw")
```

```{code-cell} python
xs.Fit.renorm()
xs.Fit.perform()
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
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

plot_fit_spec(
    fit_pl_plot_data,
    inst_name="XRISM-Resolve",
    mod_expr=pl_cont_mod.expression,
    mod_color="firebrick",
    sp_color="teal",
)
```

```{code-cell} python
pl_norm = pl_cont_mod.powerlaw.norm.values[0]
pl_index = pl_cont_mod.powerlaw.PhoIndex.values[0]

print(pl_norm, pl_index)
```

### Adding Gaussian emission line model components

```{code-cell} python
# Resetting the noticed channels
xs.AllData.notice("all")

# Applying new ignore commands (very similar to the first)
xs.AllData.ignore("bad")
cur_sp.ignore("**-3. 10.0-**")
```

```{code-cell} python
xs.AllModels.clear()

pl_gauss_mod = xs.Model("powerlaw+gauss+gauss+gauss")

# Set up the powerlaw component
pl_gauss_mod.powerlaw.norm.values = pl_norm
pl_gauss_mod.powerlaw.norm.frozen = True
pl_gauss_mod.powerlaw.PhoIndex.values = pl_index
pl_gauss_mod.powerlaw.PhoIndex.frozen = True

# First gaussian
pl_gauss_mod.gaussian.LineE.values = [6.38, 0.01, 0.0, 6.20, 6.50, 1000000.0]
pl_gauss_mod.gaussian.Sigma.values = [0.02, 0.01, 0, 0.0, 0.1, 20.0]

# Second gaussian
pl_gauss_mod.gaussian_3.LineE.values = [6.65, 0.01, 0.0, 6.6, 6.75, 1000000.0]
pl_gauss_mod.gaussian_3.Sigma.values = [0.02, 0.01, 0, 0.0, 0.05, 20.0]

# Third gaussian
pl_gauss_mod.gaussian_4.LineE.values = [7.05, 0.01, 0.0, 7, 7.1, 1000000.0]
pl_gauss_mod.gaussian_4.Sigma.values = [0.02, 0.005, 0, 0.0, 0.05, 20.0]
```

```{code-cell} python
pl_gauss_mod.show()
```

```{code-cell} python
xs.Fit.renorm()
xs.Fit.perform()
```

### Visualizing the final fit

```{code-cell} python
xs.Plot("data resid")

fit_pl_ggg_plot_data = {
    "energy": np.array(xs.Plot.x(plotWindow=1)),
    "energy_delta": np.array(xs.Plot.xErr(plotWindow=1)),
    "rate": np.array(xs.Plot.y(plotWindow=1)),
    "rate_err": np.array(xs.Plot.yErr(plotWindow=1)),
    "model": np.array(xs.Plot.model(plotWindow=1)),
    "residual": np.array(xs.Plot.y(plotWindow=2)),
    "residual_err": np.array(xs.Plot.yErr(plotWindow=2)),
}

plot_fit_spec(
    fit_pl_ggg_plot_data,
    inst_name="XRISM-Resolve",
    mod_expr=pl_gauss_mod.expression,
    mod_color="firebrick",
    sp_color="teal",
    fig_size=(10, 6),
)
plot_fit_spec(
    fit_pl_ggg_plot_data,
    inst_name="XRISM-Resolve",
    mod_expr=pl_gauss_mod.expression,
    mod_color="firebrick",
    sp_color="teal",
    fig_size=(10, 6),
    x_lims=[5.9, 8.1],
)
```

## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Anna Ogorzałek, XRISM GOF Scientist.

Updated On: 2026-06-11

+++

### Additional Resources

**XRISM Help Desk**: [https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xrism](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xrism)

**XRISM Data Reduction (ABC) Guide**: [https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide)

**HEASoftPy GitHub Repository**: [https://github.com/HEASARC/heasoftpy](https://github.com/HEASARC/heasoftpy)

**HEASoftPy HEASARC Page**: [https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy.html](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy.html)

**HEASoft XRISM Resolve/Xtend `xapipeline` help file**: [https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xapipeline.html](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xapipeline.html)

**HEASoft XRISM `xaexpmap` help file**: [https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xaexpmap.html](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xaexpmap.html)

**HEASoft XRISM `rslmkrmf` help file**: [https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/rslmkrmf.html](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/rslmkrmf.html)

**XSPEC Model Components**: [https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node128.html](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node128.html)

### Acknowledgements


### References

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE RESOLVE ENERGY-CHANNEL MAPPING_ [ACCESSED 25-Mar-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Resolve_Data_Analysis.html#SECTION00943000000000000000)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE FILE NAMING CONVENTIONS_ [ACCESSED 11-DEC-2025]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE EVENT TABLE COLUMNS_ [ACCESSED 26-Mar-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html#SECTION00770000000000000000)

[XRISM GOF & SDC (2026) - _XRISM POG EVENT GRADING_ [ACCESSED 05-May-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html#sec:resolve_eventgrading)

[XRISM GOF & SDC (2026) - _XRISM POG RESOLVE LINE SPREAD FUNCTION_ [ACCESSED 11-June-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html#sec:resolve_LSF)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE REMOVING ANOMALOUS LS EVENTS_ [ACCESSED 05-May-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Resolve_Data_Analysis.html#SECTION00932000000000000000)

[Smart D. F., Shea M. A. (2005) - _A review of geomagnetic cutoff rigidities for earth-orbiting spacecraft_](https://ui.adsabs.harvard.edu/abs/2005AdSpR..36.2012S/abstract)

[Alken P. et al. (2021) - _International Geomagnetic Reference Field: the thirteenth generation_](https://ui.adsabs.harvard.edu/abs/2021EP&S...73...49A)

[Kaastra J. S. and Bleeker J. A. M. (2016) - _Optimal binning of X-ray spectra and response matrix design_](https://ui.adsabs.harvard.edu/abs/arXiv:1601.05309)
