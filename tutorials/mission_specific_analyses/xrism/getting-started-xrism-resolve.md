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
date: '2026-04-09'
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

- Find...

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

~~Other tutorials in this series will explore how to perform more complicated generation and analysis
of XRISM-Xtend data, but here we will focus on making single aperture light curves and spectra for an
object that can be semi-reasonably treated as a 'point' source; the supernova-remnant LMC N132D.~~

***~~NOT FINAL, BUT THE TARGET IS RX J1856.5-3754, ONE OF THE MAGNIFICENT SEVEN~~***

***NOT FINAL, BUT PDS 456 IS THE TARGET - local ish radio quiet quasar***

We make use of the HEASoftPy interface to HEASoft tasks throughout this demonstration.

### Inputs

- The name of the source of interest, in this case *PDS 456*.

### Outputs

- THINGS

### Runtime

As of 25th March 2026, this notebook takes ~***????*** m to run to completion on Fornax using the 'Default Astrophysics' image and the medium server with 16GB RAM/ 4 cores.

## Imports

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os
from random import randint
from shutil import rmtree
from typing import List, Optional, Union
from warnings import warn

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

# from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity

# , UnitConversionError
from astroquery.heasarc import Heasarc

# from matplotlib.ticker import FuncFormatter
from packaging.version import Version
from xga.products import EventList  # , Image
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
    rise_time_screen: bool = True,
    exclude_frame_evts: bool = True,
    elec_coinc_evt_screen: bool = False,
    exclude_pix27: bool = True,
):
    """
    DOES NOT DO GRADE SELECTION - AS NOT RECOMMENDED FOR MAKING IMAGES/LCS, SO I
    DON'T WANT TO ENFORCE IT.
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

    if exclude_frame_evts:
        filt_expr.append("(STATUS[4]==b0)")

    if elec_coinc_evt_screen:
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
    sub_pixel: bool = False,
    im_bin_sub_pixel: int = 1,
    include_evt_grades: list = None,
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
    :param bool sub_pixel: If False (default), then the output image pixels match
        will match the Resolve's array's pixels. If True, then the output image
        will be generated using the XY coordinates, and binned according to the
        'im_bin_sub_pixel' argument, over-sampling the instruments spatial resolution.
    :param int im_bin_sub_pixel: Number of XRISM-Resolve SKY X-Y coordinate system
        'pixels' to bin into a single image pixel. Only used if 'sub_pixel' is True.
    :param List[int] include_evt_grades:
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
        pix_gti_file = os.path.join(
            out_dir, cur_obs_id, f"xa{cur_obs_id}rsl_{cur_filter}_exp.gti"
        )
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

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaexpmap_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

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
        include_pixels = "0:35"
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

    return src_out


def gen_xrism_resolve_rmf(
    event_file: str,
    spec_file: str,
    out_dir: str,
    include_evt_grades: List[int] = None,
    include_pixels: List[int] = None,
    rmf_type: str = "L",
):
    """
    A wrapper around the XRISM-Resolve-specific RMF generation tool implemented as
    part of HEASoft (and called here through HEASoftPy).

    :param str spec_file: The path to the spectrum file for which to generate an RMF.
    :param str out_dir: The directory where output files should be written.
    :param List[int] rel_pixels:
    """

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

    return out


def gen_xrism_resolve_arf(
    out_dir: str,
    expmap_file: str,
    spec_file: str,
    rmf_file: str,
    src_radec_reg_file: str,
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
    :param str expmap_file: The path to the exposure map file necessary to generate
        the ARF.
    :param str spec_file: The path to the spectrum file for which to generate an ARF.
    :param str rmf_file: The path to the RMF file necessary to generate an ARF.
    :param str src_radec_reg_file: The path to the region file defining the source
        region for which to generate an ARF.
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

    # Spectrum files generated in this demonstration notebook contain RA-Dec
    #  information in their file name, so we will read it out from there
    radec_sec = os.path.basename(spec_file).split("-radius")[0].split("-ra")[1]
    cen_strs = radec_sec.split("-dec")
    ra_val, dec_val = [float(crd) for crd in cen_strs]

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaarfgen_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # We can use the spectrum file name to set up the output ARF file name
    arf_out = os.path.basename(spec_file).replace("-spectrum.fits", ".arf")

    # Set up a name for the ray-traced simulated event file required for
    #  XRISM ARF generation
    ray_traced_evt_out = (
        f"xrism-xtend-obsid{cur_obs_id}-numphoton{num_photons}-"
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
            source_ra=ra_val,
            source_dec=dec_val,
            telescop="XRISM",
            instrume="XTEND",
            emapfile=os.path.relpath(expmap_file),
            rmffile=os.path.relpath(rmf_file),
            regionfile=os.path.relpath(src_radec_reg_file),
            regmode="RADEC",
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

    return out
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# The name of the source we're examining in this demonstration
SRC_NAME = "PDS456"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 2

# The approximate linear relationship between Resolve PI and event energy
RSL_EV_PER_CHAN = (1 / Quantity(2000, "chan/keV")).to("eV/chan")

# Expansion of event grade entries in event lists to something
#  a little more descriptive
DESCRIPTIVE_RESOLVE_EVT_GRADES = {
    "Hp": "High-resolution Primary [0]",
    "Lp": "Low-resolution Primary [3]",
    "Ls": "Low-resolution Secondary [4]",
    "Mp": "Mid-resolution Primary [1]",
    "Ms": "Mid-resolution Secondary [2]",
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

## 1. Finding and downloading XRISM observations of PDS 456

Our first task is to determine which XRISM observations are relevant to the source
that we are interested in.

We are going in with the knowledge that PDS 456 has been observed by XRISM, but of
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
summary table of observations). This should only return one catalog for any
mission you pass to 'keywords':

```{code-cell} python
catalog_name = Heasarc.list_catalogs(master=True, keywords="xrism")[0]["name"]
catalog_name
```

### What are the coordinates of PDS 456?

To search for relevant observations, we have to know the coordinates of our
source. The astropy module allows us to look up a source name in CDS' Sesame name
resolver and retrieve its coordinates.

```{hint}
You could also set up a SkyCoord object directly, if you already know the coordinates.
```

```{code-cell} python
src_coord = SkyCoord.from_name(SRC_NAME).transform_to("icrs")
# This will be useful later on in the notebook, for functions that take
#  coordinates as an astropy Quantity.
src_coord_quant = Quantity([src_coord.ra, src_coord.dec])
src_coord
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

all_xrism_obs = Heasarc.query_region(src_coord, catalog_name, columns=col_str)
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

We can see that there is a single public XRISM observation of PDS 456
(as of March 2026) - its observation ID is **300072010**.

***DECIDE WHETHER TO RESTRICT TO THIS ONE OBSID IN CASE OF FURTHER OBS, PROBABLY YES?***

```{code-cell} python
avail_xrism_obs = avail_xrism_obs[avail_xrism_obs["obsid"] == "300072010"]

# Create an array of the relevant ObsIDs
rel_obsids = avail_xrism_obs["obsid"].value.data

# Create a dictionary mapping ObsIDs to the filters used in each observation
rel_filters = {
    row["obsid"]: [
        RESOLVE_FILTERS[f]
        for f in RESOLVE_FILTERS
        if row[f"rsl_fil_{f.lower()}"] == "Y"
    ]
    for row in avail_xrism_obs
}
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


~~As with many NASA-affiliated high-energy missions, HEASoft
includes a beginning-to-end pipeline to streamline this process for XRISM data - the
XRISM-Resolve and Xtend instruments both have their own pipelines.~~

~~In this tutorial we are focused only on preparing and using data from XRISM's Xtend
instrument and will not discuss how to handle XRISM-Resolve data; we note however that
there is a third XRISM pipeline task in HEASoft called `xapipeline`, which can be used
to run either or both the Xtend and Resolve pipelines. It contains some convenient
functionality that can identify and automatically pass the attitude, housekeeping, etc. files.~~

We will show you how to run the top-level pipeline `xapipeline` XRISM pipeline, but
will limit it to processing only XRISM-Resolve data.

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
        stacklevel=2,
    )

    # This will find and download (retrieve=True) the XRISM-Resolve mid-resolution
    #  primary/secondary event channel correction file
    with contextlib.chdir(ROOT_DATA_DIR):
        caldb_ret = hsp.quzcif(
            mission="xrism",
            instrument="resolve",
            codename="RSLMPCOR",
            retrieve=True,
            noprompt=True,
            clobber=True,
        )

    # Set the path to the downloaded file
    RSLMPCOR_PATH = os.path.join(ROOT_DATA_DIR, caldb_ret.output[0].split(" ")[0])
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
- **Stage 1** -
- **Stage 2** -
- **Stage 3** -

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

```{note}
This notebook is configured to acquire XRISM CALDB files from the HEASARC
Amazon Web Services S3 bucket - this can greatly improve the speed of some
steps later in the notebook, particularly when running on the Fornax Science Console.

CALDB location configuration can be found in the [Global Setup: Configuration](#configuration) section.
```

## 3. Choosing the events to consider for data product generation

```{code-cell} python
evt_lists = {oi: EventList(EVT_PATH_TEMP.format(oi=oi)) for oi in rel_obsids}
evt_lists
```

```{code-cell} python
cur_evt_list = evt_lists[rel_obsids[0]]
```

### Event grades and branching ratios

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

### Overabundance of low-resolution secondary (**Ls**) events

***Highly recommended for Relatively Weak Sources/Temporary Measure***

The branching ratios plot demonstrates an important known issue with XRISM-Resolve data....

```
ftcopy infile="xa000126000 rsl_p0px1000_cl.evt[EVENTS][ITYPE$<$4]"
outfile=xa000126000 rsl_p0px1000_wols_cl.evt copyall=yes clobber=yes
history=yes
```

### XRISM-Resolve's pixel 27 is broken

***<span style="color:red">Obviously recommend that it is excluded, but what is the best way to do it in this particular notebook?</span>***

```{code-cell} python

```

### Excluding pixel-pixel coincident events

***ARRRGH THIS SECTION IN THE ABC FEELS LIKE ITS BEEN COPIED AND PASTED FROM VARIOUS
PLACES AND NOT EDITED - THEY SEEM TO BE TALKING ABOUT VARIOUS TYPES OF COINCIDENT EVENTS
AND NOT MAKING CLEAR WHICH ONE THEY REFER TO AT A PARTICULAR TIME***

(((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))

```
ftcopy infile="xa000126000rsl_p0px1000_cl.evt[EVENTS][(PI>=600)
&&(((((RISE_TIME+0.00075*DERIV_MAX)>46)&&
((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))&&
STATUS[4]==b0]" outfile=xa000126000rsl_p0px1000_cl2.evt
copyall=yes clobber=yes history=yes
```


#### Frame events

***<span style="color:red">CHUNK BELOW CURRENTLY JUST RIPPED FROM SECTION 6.3 OF THE XRISM ABC GUIDE</span>***
When energy is absorbed into the silicon frame around the Resolve array, it pulses the
temperature of the heat sink of all of the pixels, resulting in pulses in the
temperatures of the pixels themselves. For very large depositions of energy (MeV scale),
the resulting pulses on the pixels can produce signals that trigger in the Pulse Shape
Processor (PSP). We refer to the resulting clusters of events as frame events. Most of
these events have significantly different pulse rise times from regular events, so
they can be removed efficiently with a rise time cut. Because Ls events have a very
large spread in rise times, Ls events are excluded from the cut.

```{code-cell} python

```

#### Electrical cross-talk


#### Real events contaminated by untriggered crosstalk



### Avoiding periods of high particle background flux

**Depending on source flux or science goal - might be useful for low brightness sources, to improve SNR**

```{code-cell} python

```

### Making new 'cleaned' event lists

```{code-cell} python

```

### Further considerations for spatially-resolved analyses

Something something out of the scope something something but still need you to be
aware that there _are_ further considerations.

```{code-cell} python

```

## 4. Generating new XRISM-Resolve images

### Images from cleaned-**un**screened event lists

```{code-cell} python

```

### Images from 'for science' cleaned-screened event lists

```{code-cell} python

```

### Comparing the new images

```
# test_im = "XRISM_output/xrism-resolve-obsid300072010-filterpx1000-imbinfactorPIX-en4.0_10.0keV-image.fits"
# im = Image(test_im, '300072010', 'Resolve', "", "", "", Quantity(4, 'keV'), Quantity(10, 'keV'))
# im.view(zoom_in=True, manual_zoom_xlims=[-0.5, 5.5], manual_zoom_ylims=[-0.5, 5.5])
```

## 5. Generating new XRISM-Resolve spectra

### Defining the extraction region/pixels

```{code-cell} python

```

### Generating spectral files

```{code-cell} python
# arg_combs = [
#     [
#         EVT_PATH_TEMP.format(oi=oi, xrf=xf),
#         os.path.join(OUT_PATH, oi),
#         src_coord,
#         src_reg_rad,
#         obs_src_reg_path_temp.format(oi=oi),
#         obs_back_reg_path_temp.format(oi=oi),
#     ]
#     for oi, xfs in rel_filters.items()
#     for xf in xfs
# ]
#
# with mp.Pool(NUM_CORES) as p:
#     sp_result = p.starmap(gen_xrism_resolve_spectrum, arg_combs)
```

### Producing redistribution matrix files (RMFs)

```{code-cell} python

```

### Calculating ancillary response files (ARFs)

```{code-cell} python

```

## 4. Fitting a model with PyXspec

In this section we will perform a simple model fit to our new XRISM-Resolve spectra.

As you might imagine, spectral analysis of XRISM-Resolve data can be considerably more
complex than spectro-imaging CCD spectra. ***We defer a full exploration of more
in-depth spectral analysis to other demonstration notebooks.***

### Configuring PyXspec

```{code-cell} python

```

### Loading spectral data

```{code-cell} python

```

## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Anna Ogorzałek, XRISM GOF Scientist.

Updated On: 2026-04-09

+++

### Additional Resources

**XRISM Help Desk**: https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xrism

**XRISM Data Reduction (ABC) Guide**: https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide

**HEASoftPy GitHub Repository**: https://github.com/HEASARC/heasoftpy

**HEASoftPy HEASARC Page**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy.html

**HEASoft XRISM Resolve/Xtend `xapipeline` help file**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xapipeline.html

**HEASoft XRISM `xaexpmap` help file**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xaexpmap.html

**XSPEC Model Components**: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node128.html

### Acknowledgements


### References

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE RESOLVE ENERGY-CHANNEL MAPPING_ [ACCESSED 25-Mar-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Resolve_Data_Analysis.html#SECTION00943000000000000000)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE FILE NAMING CONVENTIONS_ [ACCESSED 11-DEC-2025]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE EVENT TABLE COLUMNS_ [ACCESSED 26-Mar-2026]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html#SECTION00770000000000000000)
