---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: Kenji Hamaguchi
  affiliations: ['University of Maryland, Baltimore County', 'XRISM GOF, NASA Goddard']
  website: https://science.gsfc.nasa.gov/sci/bio/kenji.hamaguchi-1
  orcid: 0000-0001-7515-2779
date: '2026-06-13'
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
title: Getting started with XRISM-Xtend
---

# Getting started with XRISM-Xtend

## Learning Goals

By the end of this tutorial, you will be able to:

- Identify and download XRISM observations of an interesting source.
- Prepare the XRISM-Xtend data for analysis.
- Generate XRISM-Xtend data products:
  - Images
  - Exposure maps
  - Light curves
  - Spectra and supporting files
- Perform a simple spectral analysis of a XRISM-Xtend spectrum

## Introduction

The 'X-Ray Imaging and Spectroscopy Mission' (**XRISM**) is an X-ray telescope designed for high-energy-resolution
spectroscopic observations of astrophysical sources, as well as wide-field X-ray imaging.

XRISM, launched in 2023, is the result of a JAXA-NASA partnership (with involvement from ESA), and serves as a nearly like-for-like replacement
of the **Hitomi** telescope, which was lost shortly after its launch in 2016.

There are two main XRISM instruments, **Xtend** and **Resolve**. In this tutorial, we will focus on **Xtend**, which is
a wide-field CCD spectro-imaging instrument similar in concept to instruments included on many other X-ray
telescopes (XMM's EPIC detectors, Chandra's ACIS, Swift's XRT, etc.) The other instrument, **Resolve**, has its own
dedicated demonstration notebook.

Our goal with this 'getting started' notebook is to give you the skills required to prepare XRISM-Xtend
observations for scientific use and to generate data products tailored to your science goals. It can also serve as a
template notebook to build your own analyses on top of.

Other tutorials in this series will explore how to perform more complicated generation and analysis
of XRISM-Xtend data, but here we will focus on making single aperture light curves and spectra for an
object that can be semi-reasonably treated as a 'point' source; the supernova-remnant LMC N132D.

We make use of the HEASoftPy interface to HEASoft tasks throughout this demonstration.

### Inputs

- The name of the source of interest, in this case *LMC N132D*

### Outputs

- Processed, cleaned, and calibrated XRISM-Xtend event lists.
- XRISM-Xtend images, exposure maps, light curves, spectra, and supporting files.
- Simple region files that define where light curves and spectra are extracted from.

### Runtime

As of 2nd March 2026, this notebook takes ~50 m to run to completion on Fornax using the 'Default Astrophysics' image and the medium server with 16GB RAM/ 4 cores.

## Imports

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os
from random import randint
from shutil import rmtree
from typing import Union

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity, UnitConversionError
from astroquery.heasarc import Heasarc
from matplotlib.ticker import FuncFormatter
from packaging.version import Version
from regions import CircleSkyRegion, Regions
from xga.products import Image, LightCurve
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def process_xrism_xtend(
    cur_obs_id: str,
    out_dir: str,
    evt_dir: str,
    attitude: str,
    orbit: str,
    obs_gti: str,
    mkf_filter: str,
    file_stem: str,
    extended_housekeeping: str,
    xtend_housekeeping: str,
    calc_modegti: bool = True,
):
    """
    A wrapper for the HEASoftPy xtdpipeline task, which is used to prepare and process
    XRISM-Xtend observation data. This wrapper function is primarily to enable the
    use of multiprocessing.

    This function is set to run xtdpipeline until the end of stage 2, excluding the
    final stage that generates the 'quick-look' data products.

    :param str cur_obs_id: The ObsID of the XRISM observation to be processed.
    :param str out_dir: The directory where output files should be written.
    :param str evt_dir: The directory containing the raw, unfiltered, event list
        files for the observation.
    :param str attitude: XRISM attitude file for the observation.
    :param str orbit: XRISM orbit file for the observation.
    :param str obs_gti: XRISM base good-time-invterval file for the observation.
    :param str mkf_filter: XRISM overall filter file for the observation.
    :param str file_stem: The stem of the input event list files (also used for
        output file names).
    :param str extended_housekeeping: Extended housekeeping file for the
        XRISM observation.
    :param str xtend_housekeeping: Instrument-specific Xtend housekeeping file
        for the observation.
    :param bool calc_modegti: Whether to execute `xtdmodegti` to create the
        segment and mode GTI for Xtend using the exposure file. Default is True.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xtdpipeline_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():

        # The processing/preparation stage of any X-ray telescope's data is the most
        #  likely to go wrong, and we use a Python try-except as an automated way to
        #  collect ObsIDs that had an issue during processing.
        try:
            out = hsp.xtdpipeline(
                entry_stage=1,
                exit_stage=2,
                steminputs=file_stem,
                stemoutputs=file_stem,
                indir=evt_dir,
                outdir=".",
                attitude=attitude,
                orbit=orbit,
                obsgti=obs_gti,
                makefilter=mkf_filter,
                extended_housekeeping=extended_housekeeping,
                housekeeping=xtend_housekeeping,
                calc_modegti=calc_modegti,
                clobber=True,
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


def gen_xrism_xtend_image(
    event_file: str,
    out_dir: str,
    lo_en: Quantity,
    hi_en: Quantity,
    im_bin: int = 1,
):
    """
    This function wraps the HEASoft 'extractor' tool and is used to spatially bin
    XRISM-Xtend event lists into images. The HEASoftPy interface to 'extractor' is used.

    Both the energy band and the image binning factor, which controls how
    many 'pixels' in the native SKY X-Y coordinate of the event list are binned into
    a single image pixel, can be specified.

    The ObsID and dataclass are extracted from the header of the passed event list file.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate an image from. ObsID and dataclass information
        will be extracted from the EVENTS table header.
    :param str out_dir: The directory where output files should be written.
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the image.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the image.
    :param int im_bin: Number of XRISM-Xtend SKY X-Y pixels to bin into a single image
        pixel.
    """

    # We can extract the ObsID and data class directly from the header of the event
    #  list - it is safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        cur_xtend_data_class = read_evto["EVENTS"].header["DATACLAS"]

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
    lo_ch = np.floor((lo_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the image we're about to generate.
    im_out = os.path.basename(IM_PATH_TEMP).format(
        oi=cur_obs_id, xdc=cur_xtend_data_class, ibf=im_bin, lo=lo_en_val, hi=hi_en_val
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
            gti="GTI",
        )

    # Move the output image file to the proper output directory from
    #  the temporary working directory
    os.rename(os.path.join(temp_work_dir, im_out), os.path.join(out_dir, im_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_xtend_expmap(
    event_file: str,
    out_dir: str,
    gti_file: str,
    extend_hk_file: str,
    bad_pix_file: str,
    pix_gti_file: str = "NONE",
    im_bin: int = 1,
    radial_delta: Union[float, Quantity] = Quantity(20.0, "arcmin"),
    num_phi_bin: int = 1,
):
    """
    Function that wraps the HEASoftPy interface to the XRISM-Xtend 'xaexpmap'
    task, which is used to generate exposure maps for XRISM-Xtend observations.

    :param str event_file: Event list of the observation + dataclass you wish to
        generate an exposure map for. No event data are used in the creation of the
        event list, but some information in the file headers is useful.
    :param str out_dir: The directory where output files should be written.
    :param str gti_file: File defining the good-time-intervals of the observation
        and observation dataclass for which we are generating an exposure map (often
        the event list itself is passed).
    :param str extend_hk_file:
    :param str bad_pix_file:
    :param str pix_gti_file: Optional file defining the good-time-intervals of
        individual XRISM-Xtend pixels. If not provided, the default value of 'NONE' is
        passed to 'xaexpmap'.
    :param im_bin: Number of XRISM-Xtend SKY X-Y pixels to bin into a single exposure
        map pixel. Defaults to 1, and any other value will also result in an
        'im_bin=1' being generated.
    :param float/Quantity radial_delta: Radial increment for the annular grid for
        which the attitude histogram will be calculated.
    :param int num_phi_bin: Number of azimuth (phi) bins in the first annular region
        over which attitude histogram bins will be calculated
    """

    # We can extract the ObsID and data class directly from the header of the event
    #  list - it is safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        cur_xtend_data_class = read_evto["EVENTS"].header["DATACLAS"]

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
    # TODO REINSTATE WHEN WE HAVE A BETTER UNDERSTANDING OF POTENTIAL USER USES
    # ex_type = "expmap" if out_map_type == "EXPOSURE" else "flatfieldmap"

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaexpmap_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Set up the output file name for the exposure map we're about to generate.
    ex_out = os.path.basename(EX_PATH_TEMP).format(
        oi=cur_obs_id, xdc=cur_xtend_data_class, rd=radial_delta, npb=num_phi_bin, ibf=1
    )

    # If the user wants to bin up the exposure map, we'll need to set up another
    #  output file name with the bin factor set to the input value (this variable
    #  is not used if the user does not want to bin the map)
    binned_ex_out = os.path.basename(EX_PATH_TEMP).format(
        oi=cur_obs_id,
        xdc=cur_xtend_data_class,
        rd=radial_delta,
        npb=num_phi_bin,
        ibf=im_bin,
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaexpmap_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.xaexpmap(
            instrume="XTEND",
            ehkfile=extend_hk_file,
            gtifile=gti_file,
            pixgtifile=pix_gti_file,
            delta=radial_delta,
            numphi=num_phi_bin,
            outfile=ex_out,
            badimgfile=bad_pix_file,
            outmaptype=out_map_type,
            noprompt=True,
            clobber=True,
        )

        # If the user wants a spatially binned exposure map, we run the fimgbin task
        if im_bin != 1:
            rebin_out = hsp.fimgbin(
                infile=ex_out,
                outfile=binned_ex_out,
                xbinsize=im_bin,
                noprompt=True,
                clobber=True,
            )
            out = [out, rebin_out]

    # Move the im_bin=1 exposure map (guaranteed to have been generated) up to the
    #  final output directory
    os.rename(os.path.join(temp_work_dir, ex_out), os.path.join(out_dir, ex_out))
    # Then do the same for the spatially binned exposure map, if it was requested
    if im_bin != 1:
        os.rename(
            os.path.join(temp_work_dir, binned_ex_out),
            os.path.join(out_dir, binned_ex_out),
        )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_xtend_lightcurve(
    event_file: str,
    out_dir: str,
    rel_src_coord: SkyCoord,
    rel_src_radius: Quantity,
    src_reg_file: str,
    back_reg_file: str,
    lo_en: Quantity = Quantity(0.6, "keV"),
    hi_en: Quantity = Quantity(13, "keV"),
    time_bin_size: Quantity = Quantity(200, "s"),
    lc_bin_thresh: float = 0.0,
):
    """
    Function that wraps the HEASoftPy interface to the HEASoft extractor tool, set
    up to generate light curves from XRISM-Xtend observations. The function will
    generate a light curve for the source region and a background light curve for
    the background region.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate a XRISM-Xtend light curve from. ObsID and
        dataclass information will be extracted from the EVENTS table header.
    :param str out_dir: The directory where output files should be written.
    :param SkyCoord rel_src_coord: The source coordinate (RA, Dec) of the
        source region for which we wish to generate a light curve.
    :param Quantity rel_src_radius: The radius of the source region for which we wish
        to generate a light curve.
    :param str src_reg_file: Path to the region file defining the source region for
        which we wish to generate a light curve.
    :param str back_reg_file: Path to the region file defining the background region
        for which we wish to generate a light curve.
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the light curve.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the light curve.
    :param Quantity time_bin_size: The size of the time bins used to generate the
        light curve.
    :param float lc_bin_thresh: When constructing a light curve, any bins whose
        exposure is less than lc_bin_thresh*time_bin_size are ignored.
    """

    # We can extract the ObsID and data class directly from the header of the event
    #  list - it is safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        cur_xtend_data_class = read_evto["EVENTS"].header["DATACLAS"]

    # Get RA, Dec, and radius values in the right format
    ra_val = rel_src_coord.ra.to("deg").value.round(6)
    dec_val = rel_src_coord.dec.to("deg").value.round(6)
    rad_val = rel_src_radius.to("deg").value.round(4)

    # Check the units of the passed time bin size - also if the passed value is
    #  a float or integer, we'll assume it is in seconds
    if not isinstance(time_bin_size, Quantity):
        time_bin_size = Quantity(time_bin_size, "s")
    elif not time_bin_size.unit.is_equivalent("s"):
        raise UnitConversionError(
            f"The 'time_bin_size' argument ({time_bin_size}) "
            "must be an astropy Quantity that is convertible "
            "to seconds."
        )

    # Convert the time bin size to seconds and convert it to a simple integer/float
    time_bin_size = time_bin_size.to("s").value

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
    # We will use these to make a channel selection in the event list passed
    #  to the tool
    lo_ch = np.floor((lo_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)

    # Set up the output file name for the light curve we're about to generate.
    lc_out = os.path.basename(LC_PATH_TEMP).format(
        oi=cur_obs_id,
        xdc=cur_xtend_data_class,
        ra=ra_val,
        dec=dec_val,
        rad=rad_val,
        lo=lo_en_val,
        hi=hi_en_val,
        lct=lc_bin_thresh,
        tb=time_bin_size,
    )

    # The same file name, but with 'lightcurve' changed to 'back-lightcurve', and the
    #  radius information information removed, for the background light curve.
    lc_back_out = os.path.basename(BACK_LC_PATH_TEMP).format(
        oi=cur_obs_id,
        xdc=cur_xtend_data_class,
        ra=ra_val,
        dec=dec_val,
        lo=lo_en_val,
        hi=hi_en_val,
        lct=lc_bin_thresh,
        tb=time_bin_size,
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "lightcurve_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        # Create modified input event list file path, where we use the just-calculated
        #  PI channel limits to subset the events
        # This is within the chdir context, as we want to pass a relative path
        #  as very long string arguments can be cut off when passed to HEASoft tools
        evt_file_chan_sel = os.path.relpath(event_file) + f"[PI={lo_ch}:{hi_ch}]"

        src_out = hsp.extractor(
            filename=evt_file_chan_sel,
            fitsbinlc=lc_out,
            binlc=time_bin_size,
            lcthresh=lc_bin_thresh,
            regionfile=os.path.relpath(src_reg_file),
            xcolf="X",
            ycolf="Y",
            gti="GTI",
            noprompt=True,
            clobber=True,
        )

        # Now for the background light curve
        back_out = hsp.extractor(
            filename=evt_file_chan_sel,
            fitsbinlc=lc_back_out,
            binlc=time_bin_size,
            lcthresh=lc_bin_thresh,
            regionfile=os.path.relpath(back_reg_file),
            xcolf="X",
            ycolf="Y",
            gti="GTI",
            noprompt=True,
            clobber=True,
        )

    # Move the light curves up from the temporary directory
    os.rename(os.path.join(temp_work_dir, lc_out), os.path.join(out_dir, lc_out))
    os.rename(
        os.path.join(temp_work_dir, lc_back_out), os.path.join(out_dir, lc_back_out)
    )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return [src_out, back_out]


def gen_xrism_xtend_spectrum(
    event_file: str,
    out_dir: str,
    rel_src_coord: SkyCoord,
    rel_src_radius: Quantity,
    src_reg_file: str,
    back_reg_file: str,
):
    """
    Function that wraps the HEASoftPy interface to the HEASoft extractor tool, set
    up to generate spectra from XRISM-Xtend observations. The function will
    generate a spectrum for the source region and a background spectrum for
    the background region.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate a XRISM-Xtend spectrum from. ObsID and
        dataclass information will be extracted from the EVENTS table header.
    :param str out_dir: The directory where output files should be written.
    :param SkyCoord rel_src_coord: The source coordinate (RA, Dec) of the
        source region for which we wish to generate a spectrum.
    :param Quantity rel_src_radius: The radius of the source region for which we wish
        to generate a spectrum.
    :param str src_reg_file: Path to the region file defining the source region for
        which we wish to generate a spectrum.
    :param str back_reg_file: Path to the region file defining the background region
        for which we wish to generate a spectrum.
    """

    # We can extract the ObsID and data class directly from the header of the event
    #  list - it is safer than having them be passed to this function separately.
    with fits.open(event_file) as read_evto:
        cur_obs_id = read_evto["EVENTS"].header["OBS_ID"]
        cur_xtend_data_class = read_evto["EVENTS"].header["DATACLAS"]

    # Get RA, Dec, and radius values in the right format
    ra_val = rel_src_coord.ra.to("deg").value.round(6)
    dec_val = rel_src_coord.dec.to("deg").value.round(6)
    rad_val = rel_src_radius.to("deg").value.round(4)

    # Set up the output file names for the source and background spectra we're
    #  about to generate.
    sp_out = os.path.basename(SP_PATH_TEMP).format(
        oi=cur_obs_id, xdc=cur_xtend_data_class, ra=ra_val, dec=dec_val, rad=rad_val
    )
    sp_back_out = os.path.basename(BACK_SP_PATH_TEMP).format(
        oi=cur_obs_id, xdc=cur_xtend_data_class, ra=ra_val, dec=dec_val
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
            filename=os.path.relpath(event_file),
            phafile=sp_out,
            regionfile=os.path.relpath(src_reg_file),
            xcolf="X",
            ycolf="Y",
            ecol="PI",
            gti="GTI",
            noprompt=True,
            clobber=True,
        )

        # Now for the background spectrum
        back_out = hsp.extractor(
            filename=os.path.relpath(event_file),
            phafile=sp_back_out,
            regionfile=os.path.relpath(back_reg_file),
            xcolf="X",
            ycolf="Y",
            ecol="PI",
            gti="GTI",
            noprompt=True,
            clobber=True,
        )

    # Move the spectra up from the temporary directory
    os.rename(os.path.join(temp_work_dir, sp_out), os.path.join(out_dir, sp_out))
    os.rename(
        os.path.join(temp_work_dir, sp_back_out), os.path.join(out_dir, sp_back_out)
    )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return [src_out, back_out]


def gen_xrism_xtend_rmf(spec_file: str, out_dir: str):
    """
    A wrapper around the XRISM-Xtend-specific RMF generation tool implemented as
    part of HEASoft (and called here through HEASoftPy).

    :param str spec_file: The path to the spectrum file for which to generate an RMF.
    :param str out_dir: The directory where output files should be written.
    """

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xtdrmf_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Set up the RMF file name by cannibalising the name of the spectrum file - this
    #  means we don't have to worry about identifying the ObsID
    rmf_out = os.path.basename(spec_file).split("-ra")[0] + ".rmf"

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.xtdrmf(
            infile=spec_file,
            outfile=rmf_out,
            noprompt=True,
            clobber=True,
        )

    # Move the RMF up from the temporary directory
    os.rename(os.path.join(temp_work_dir, rmf_out), os.path.join(out_dir, rmf_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_xtend_arf(
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
    ARFs for XRISM-Xtend spectra.

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
        XRISM-Xtend ARF generation.
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
SRC_NAME = "LMCN132D"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3

# The approximate linear relationship between Xtend PI and event energy
XTD_EV_PER_CHAN = (1 / Quantity(166.7, "chan/keV")).to("eV/chan")
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
# ------ XTDPIPELINE -------
# Cleaned event list path template - obviously going to be useful later
EVT_PATH_TEMP = os.path.join(OUT_PATH, "{oi}", "xa{oi}xtd_p{sc}{xdc}_cl.evt")

# The path to the bad pixel map, useful for excluding dodgy pixels from data products
BADPIX_PATH_TEMP = os.path.join(OUT_PATH, "{oi}", "xa{oi}xtd_p{sc}{xdc}.bimg")
# --------------------------

# --------- IMAGES ---------
IM_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-imbinfactor{ibf}-en{lo}_{hi}keV-image.fits",
)
# --------------------------


# -------- EXPMAPS ---------
EX_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-attraddelta{rd}arcmin-"
    "attphibin{npb}-imbinfactor{ibf}-enALL-expmap.fits",
)
# --------------------------


# ------ LIGHTCURVES -------
LC_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-ra{ra}-dec{dec}-radius{rad}deg-"
    "en{lo}_{hi}keV-expthresh{lct}-tb{tb}s-lightcurve.fits",
)

BACK_LC_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-ra{ra}-dec{dec}-"
    "en{lo}_{hi}keV-expthresh{lct}-tb{tb}s-back-lightcurve.fits",
)

NET_LC_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-ra{ra}-dec{dec}-radius{rad}deg-"
    "en{lo}_{hi}keV-expthresh{lct}-tb{tb}s-net-lightcurve.fits",
)
# --------------------------


# -------- SPECTRA ---------
SP_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-ra{ra}-dec{dec}-radius{rad}deg-"
    "enALL-spectrum.fits",
)

BACK_SP_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-ra{ra}-dec{dec}-enALL-back-spectrum.fits",
)
# --------------------------

# ----- GROUPEDSPECTRA -----
GRP_SP_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum", "-{gt}grp{gs}-spectrum")
# --------------------------

# ---------- RMF -----------
RMF_PATH_TEMP = os.path.join(
    OUT_PATH, "{oi}", "xrism-xtend-obsid{oi}-dataclass{xdc}.rmf"
)
# --------------------------

# ---------- ARF -----------
ARF_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum.fits", ".arf")
# --------------------------
# --------------------------------------------------------------
```

***

## 1. Finding and downloading XRISM observations of LMC N132D

Our first task is to determine which XRISM observations are relevant to the source
that we are interested in.

We are going in with the knowledge that LMC N132D has been observed by XRISM, but of
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

### What are the coordinates of LMC N132D?

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
does not contain the 'xtd_dataclas*' columns that we might need later. You may also
pass a wildcard `columns='*'` to retrieve all available columns.

```{code-cell} python
col_str = (
    "__row,obsid,name,ra,dec,time,exposure,status,public_date,"
    "xtd_dataclas1,xtd_dataclas2"
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

We can see that there are three public XRISM observations of LMC N132D
(as of December 2025) that we could make use of for this demonstration.

To make sure that this notebook can run in a reasonable amount of time, we
are only going to choose one of them; observation 000128000.

Please note that we have written this notebook in such a way that you could remove the
first line of the next cell (which selects only one ObsID) and run the notebook
for all public observations.

```{code-cell} python
avail_xrism_obs = avail_xrism_obs[avail_xrism_obs["obsid"] == "000128000"]

# Define a couple of useful variables that make accessing information in the
#  table a little easier later on in the notebook
# Create an array of the relevant ObsIDs
rel_obsids = avail_xrism_obs["obsid"].value.data
# Create a dictionary connecting ObsIDs to their associated Xtend data classes
rel_dataclasses = {
    oi: [
        dc
        for dc in avail_xrism_obs[oi_ind][["xtd_dataclas1", "xtd_dataclas2"]].values()
        if dc != ""
    ]
    for oi_ind, oi in enumerate(rel_obsids)
}
```

### Downloading the selected XRISM observations

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
Heasarc.download_data(data_links, "aws", ROOT_DATA_DIR)
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
glob.glob(os.path.join(ROOT_DATA_DIR, rel_obsids[0], "xtend", "") + "**/*")
```

## 2. Processing XRISM-Xtend data

There are multiple steps involved in processing XRISM-Xtend data into a
science-ready state. As with many NASA-affiliated high-energy missions, HEASoft
includes a beginning-to-end pipeline to streamline this process for XRISM data - the
XRISM-Xtend and Resolve instruments both have their own pipelines.

In this tutorial we are focused only on preparing and using data from XRISM's Xtend
instrument and will not discuss how to handle XRISM-Resolve data; we note however that
there is a third XRISM pipeline task in HEASoft called `xapipeline`, which can be used
to run either or both the Xtend and Resolve pipelines. It contains some convenient
functionality that can identify and automatically pass the attitude, housekeeping, etc. files.

We will show you how to run the Xtend-specific pipeline, `xtdpipeline`, but the
use of `xapipeline` is nearly identical.

The Python interface to HEASoft, HEASoftPy, is used throughout this tutorial, and we
will implement parallel observation processing wherever possible (even though we have
only selected a single observation).

### HEASoft and HEASoftPy versions

```{warning}
XRISM is a relatively new mission, and as such the analysis software and recommended
best practises are still immature and evolving. We are checking and updating this tutorial
on a regular basis, but please report any issues, or make suggestions, to the [XRISM Help Desk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xrism).
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

### Setting up file paths to pass to the XRISM-Xtend pipeline

In order to properly prepare and calibrate XRISM-Xtend data, `xtdpipeline` must
make use of a number of housekeeping files that describe the observatory's status.

Here we set up template file path variables to the required files so that we can
more easily pass observation-specific file paths to the XRISM-Xtend processing
function in the next section.

The only expected difference in file name between the equivalent files of different
observations is the included ObsID string, represented by the `{oi}` placeholder. This
placeholder will be replaced by the relevant ObsID for each observation being processed.

In summary, the supporting files required by `xtdpipeline` are:
- **Attitude file** - Describes the pointing of XRISM in many short time steps throughout the observation.
- **Orbit file** - Orbital telemetry of the XRISM spacecraft during the observation.
- **Observation good-time-intervals (GTI) file** - Contains base GTIs for the observation; used to exclude times when the spacecraft was slewing, or its attitude was inconsistent with that required to observe the target.
- **Filter file (MKF)** - The base filters used to exclude times when the instruments or spacecraft were not operating normally.
- **Extended housekeeping (EHK) file** - Contains extra information about the observation derived from attitude and orbit files, used to screen events. Much of the data relates to attitude, the South Atlantic Anomaly (SAA), and cut-off rigidity (COR).
- **Xtend housekeeping (HK) file** - An instrument-specific housekeeping file that summarizes the electrical and thermal state of Xtend in small time steps throughout the observation.

```{code-cell} python
# File containing XRISM pointing information
att_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.att.gz")

# File containing XRISM orbital telemetry
orbit_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.orb.gz")

# The base XRISM observation GTI file
obs_gti_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}_gen.gti.gz")

# The overall XRISM observation filter file
mkf_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.mkf.gz")

# The XRISM extended housekeeping file
ehk_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.ehk.gz")

# The Xtend housekeeping file
xtd_hk_path_temp = os.path.join(
    ROOT_DATA_DIR, "{oi}", "xtend", "hk", "xa{oi}xtd_a0.hk.gz"
)
```

`xtdpipeline` also needs the 'stem' of the input file names to be defined, so that it
can identify the relevant event list files. The way we call the pipeline, the input
stem will also be used to format output file names.

```{code-cell} python
file_stem_temp = "xa{oi}"
```

Finally, we set up a template variable for the directory containing the raw
Xtend event information for each observation. It contains several files, and
`xtdpipeline` will identify the ones it needs to use:

```{code-cell} python
raw_evt_dir_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "xtend", "event_uf")
```

### Running the XRISM-Xtend pipeline

`xtdpipeline` will take us from a brand-new set of raw XRISM-Xtend data files, all the way
through to generating the 'quick-look' data products (images, spectra, and light curves)
included in HEASARC's XRISM archive 'products' directories.

This `xtdpipeline` task will prepare **all** data taken during a particular
observation - that means that if you are using an XRISM-Xtend observation that
was running in a data mode other than full-window, you still only have to
run `xtdpipeline` once.

Full-window should be considered XRISM-Xtend's default data mode, but you are likely
to come across data taken in other modes. Those modes are described in the XRISM
ABC guide ([XRISM GOF & SDC 2024](https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Xtend_SXI.html#SECTION00920000000000000000)),
and we summarize them here:

- **Full-window** - The entire Xtend detector is in the same data mode, producing a 640x640 (in raw CCD coordinates) image, at a 4-second time resolution.
- **1/8th window [NO BURST]** - Half of the detector (2 CCDs) operates 'normally' and the other has only 1/8th of the pixel rows operating. Produces a 640x80 image, at 0.5-second time resolution.
- **1/8th window [BURST]** - Similar to the 1/8th window mode but collects exposures during only a small fraction of the effective detector exposures (the 0.5-second time resolution). Avoids pile-up for very bright sources and allows the determination of photon arrival times with ~0.06-second accuracy.

Data taken in each data mode is assigned a different 'dataclass' so that the multiple
event lists produced when using a 1/8th window mode can be distinguished from the event
list of the half of the detector that is operating 'normally'. The dataclasses are discussed in the XRISM ABC guide
([XRISM GOF & SDC 2024](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html)) and are summarized below:
- **30000010** - All four CCDs are in full window mode.
- **31100010** - CCD1 & CCD2 in 1/8 window mode.
- **31200010** - CCD1 & CCD2 in full window + 0.1 sec burst mode.
- **31300010** - CCD1 & CCD2 in 1/8 window + 0.1 sec burst mode
- **32000010** - CCD3 & CCD4 in full window mode

The pipeline has three stages and provides the option to start and stop the processing
at any of those stages; this can be useful if you wish to re-run a stage with slightly
different configuration without repeating the entire pipeline run.

A different set of tasks is encapsulated by each stage, and they have the following general goals:
- **Stage 1** - Calibration and preparation of raw Xtend data.
- **Stage 2** - Screening and filtering of the prepared Xtend event lists.
- **Stage 3** - Generation of quick-look data products.

```{note}
We will stop the execution of `xtdpipeline` at **Stage 2**, as the latter part of this
demonstration will show you how to make more customised data products than are output
by default.
```

There are a great many arguments that can be passed to the `xtdpipeline` task to
modify its behaviors and exercise finer control over its outputs - please see the
[`xtdpipeline` documentation](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xtdpipeline.html)
for a full overview.

One optional argument that we change from its default value is `calc_modegti`, which
is normally set to `True` (or 'yes', if you're running `xtdpipeline` from the command line). This
controls whether the `xtdpipeline` task will calculate new GTIs for each _separate_ dataclass
of a particular observation (if there are multiple dataclasses are present).

Generating individual GTIs for different dataclasses (using the `xtdmodegti` HEASoft task; [see the documentation](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xtdmodegti.html))
allows for a more precise exclusion of detector `dead time`, which may well be different for different dataclasses.

We have disabled this behavior due to the significant increase in processing time required for this step.

```{code-cell} python
calc_dataclass_specific_gti = False
```

```{warning}
Depending on your exact science case, you may wish to set
`calc_dataclass_specific_gti=True` to re-enable the `xtdmodegti` step of `xtdpipeline`.
Doing so will likely increase the run time, but may be necessary if you care about
very precise timings and count-rates.
```

Though we are using the HEASoftPy `xtdpipeline` function, called
as `hsp.xtdpipeline(indir=...)`, we wrap it in a function defined in
the 'Global Setup: Functions' section of this notebook. The `process_xrism_xtend`
wrapper function exists primarily to let us run the processing of different XRISM-Xtend
observations in parallel.

We do not allow for every argument supported by `xtdpipeline` to be passed to the wrapper
function, but you could copy and modify `process_xrism_xtend` to suit your needs, or
create an entirely new wrapper function.

We can use Python's multiprocessing module to call the wrapper function for each
of our XRISM observations, passing the relevant arguments.

The multiprocessing pool will then execute the processing of observations
simultaneously, if there are more cores available than there are observations.

If there are fewer cores than observations, the pool will handle the allocation of
resources to each observation's processing run, and they will be processed in parallel
until all are complete.

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [
        [
            oi,
            os.path.join(OUT_PATH, oi),
            raw_evt_dir_temp.format(oi=oi),
            att_path_temp.format(oi=oi),
            orbit_path_temp.format(oi=oi),
            obs_gti_path_temp.format(oi=oi),
            mkf_path_temp.format(oi=oi),
            file_stem_temp.format(oi=oi),
            ehk_path_temp.format(oi=oi),
            xtd_hk_path_temp.format(oi=oi),
            calc_dataclass_specific_gti,
        ]
        for oi in rel_obsids
    ]

    pipe_result = p.starmap(process_xrism_xtend, arg_combs)

xtd_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in xtd_pipe_problem_ois]
rel_dataclasses = {oi: rel_dataclasses[oi] for oi in rel_obsids}

xtd_pipe_problem_ois
```

```{warning}
Processing XRISM-Xtend data can take a long time, up to several hours for a single observation.
```

We also include a code snippet that will print the output of the `xtdpipeline` run for any
observations that appear to have failed:

```{code-cell} python
if len(xtd_pipe_problem_ois) != 0:
    for all_out in pipe_result:
        if all_out[0] in xtd_pipe_problem_ois:
            print(all_out[1])
            print("\n\n")
```

```{note}
This notebook is configured to acquire XRISM CALDB files from the HEASARC
Amazon Web Services S3 bucket - this can greatly improve the speed of some
steps later in the notebook when running on the Fornax Science Console.

CALDB location configuration can be found in the 'Global Setup: Configuration' section.
```

## 3. Generating new XRISM-Xtend images and exposure maps

The XRISM-Xtend data have now been prepared for scientific use, with the most important
output being the cleaned event list(s); remember that one observation can produce
**two** cleaned event lists if Xtend was operating in a windowed or burst mode.

We will now demonstrate how to generate new XRISM-Xtend data products tailored to your
scientific needs. Images and exposure maps can be generated for the entire
field-of-view (FoV; or at least the entire FoV of a particular observation mode, e.g., full window, 1/8th window, etc.), rather than having to focus on a particular source, so we will
start with them.

### Converting energy bounds to channel bounds

The data products we generate in this section (and the next) can all benefit from selecting events
from within a specific energy range. This might be because your source of interest only
emits in a narrow energy range, and you don't care about the rest, or because different
mechanisms emit at different energies, and you wish to separate them.

Such filtering needs to be performed at the event list level so that the resulting
subset of events can be binned in spatial and temporal dimensions to produce
images and light curves.

The event lists of most high-energy missions (including XRISM) do not directly store
event energies - instead they contain the pulse-height-amplitude (PHA), and/or the
pulse-invariant (PI) channel (calculated from PHA and instrument gain tables) information.

This is because the calibration of detector-channel to energy, the understanding of the
behaviors of the instrument and its electronics, and the performance of the detectors
can all change dramatically over time.

All that said, the tools we will use to generate our energy-bounded images and light
curves do not take _energy_ bounds as an input, but rather _channel_ bounds.

Thus, we have the responsibility of determining equivalent channel bounds for our
hopefully-physics-driven energy-bound choices. For images and light curves, we can
safely assume a perfect linear relationship between energy and channel.

The XRISM ABC guide provides the following mapping
([XRISM GOF & SDC 2024](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Xtend_Data_Analysis.html#SECTION001043000000000000000))
for Xtend:

```{code-cell} python
XTD_EV_PER_CHAN
```

Alternatively, we can figure out this relationship between PI and energy by looking at
a XRISM-Xtend Redistribution Matrix File (RMF), which exists to describe this
mapping.

We will be creating new RMFs as part of the generation of XRISM-Xtend spectra later
in this notebook. For our current purpose, however, it is acceptable to use the RMFs that
were included in the XRISM-Xtend archive we downloaded earlier.

The archived RMFs are generated for the entire Xtend FoV, rather than for the CCDs
our particular target falls on, but practically speaking, that doesn't make a significant
difference.

Using observation 000128000 as an example, we determine the path to the relevant
pre-generated RMF. We only expect a single file and include a validity check to
ensure that this does not change in future versions of the archive:

```{code-cell} python
chosen_demo_obsid = "000128000"

pregen_rmf_wildcard = os.path.join(
    ROOT_DATA_DIR, "{oi}", "xtend", "products", "xa{oi}xtd_p*.rmf*"
)
poss_rmfs = glob.glob(pregen_rmf_wildcard.format(oi=chosen_demo_obsid))
print(poss_rmfs)

# Check how many RMF files we found - there should only be one
if len(poss_rmfs) != 1:
    raise ValueError(f"Expected exactly one RMF file, but found {len(poss_rmfs)}.")
else:
    pregen_rmf_path = poss_rmfs[0]
```

XRISM-Xtend RMFs are written in the FITS file format, and so can be read into
Python using the `astropy.io.fits` module:

```{code-cell} python
# Loading the fits file using astropy
with fits.open(pregen_rmf_path) as rmfo:
    # Iterate through the tables in the RMF, printing their names
    for tab in rmfo:
        print(tab.name)
    print("")

    # Associate the EBOUNDS table with a variable, so it can be used outside
    #  the fits.open context
    e_bounds = rmfo["EBOUNDS"].data

# Convert the read-out energy bound information to an astropy Table, mainly
#  because it will look nicer whe we show it below
e_bounds = Table(e_bounds)
# Display a subset of the table
e_bounds[90:110]
```

We can use this file to visualize the basic linear mapping between energy and
channel - *it will be the most boring figure you've ever seen*:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Set up the figure
plt.figure(figsize=(5.5, 5.5))

# Configuring the axis ticks
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

# Calculate the mid-point of each energy bin
mid_ens = (e_bounds["E_MIN"] + e_bounds["E_MAX"]) / 2

# Plot the relationship between channel and the energy bin mid-points
plt.plot(e_bounds["CHANNEL"], mid_ens, color="navy", alpha=0.9, label="XRISM-Xtend")

plt.xlim(0)
plt.ylim(0)

plt.xlabel("Channel [PI]", fontsize=15)
plt.ylabel("Central Energy [keV]", fontsize=15)

plt.legend(fontsize=14)

plt.tight_layout()
plt.show()
```

Finally, we can validate our assumed relationship between energy and channel by
calculating the mean change in minimum energy between adjacent channels:

```{code-cell} python
# Calculates the energy change from one to channel to the next, then finds the
#  mean value of those energy changes
mean_en_diffs = np.diff(e_bounds["E_MIN"].data).mean()

# Set up the result in an astropy quantity and convert to eV-per-channel for
#  easier comparison to the assumed relationship
rmf_ev_per_chan = Quantity(mean_en_diffs, "keV/chan").to("eV/chan")
rmf_ev_per_chan
```

Clearly, our assumed relationship is valid:

```{code-cell} python
rmf_ev_per_chan / XTD_EV_PER_CHAN
```

### New XRISM-Xtend images

We've established that we understand XRISM-Xtend's relationship between energy and
channel. Now we can use that relationship to choose the energy bounds we generate
data products within and convert them to the channel values required by XRISM HEASoft
tasks.

We recommend that you generate images first, as examining them is a good way to spot
any problems or unusual features of the prepared and cleaned observations.

#### Image energy bounds

We are going to generate images within the following energy bounds:
- 0.6-10.0 keV
- 0.6-2.0 keV
- 2.0-10.0 keV
- ***0.4–2.0 keV*** [not recommended]

A lower bound of ***0.4 keV*** is ***not recommended***, as there
are issues with XRISM-Xtend data below *0.6 keV*. We are generating them to
demonstrate those issues.

```{code-cell} python
# Defining the energy bounds we want images within
xtd_im_en_bounds = Quantity([[0.6, 10.0], [0.6, 2.0], [2.0, 10.0], [0.4, 2.0]], "keV")
```

Converting those energy bounds to channel bounds is straightforward, we simply divide
the energy values by our assumed mapping between energy and channel.

The resulting lower and upper bound channel values are rounded down and up to the
nearest integer channel respectively.

```{code-cell} python
# Convert energy bounds to channel bounds
xtd_im_ch_bounds = (xtd_im_en_bounds / XTD_EV_PER_CHAN).to("chan")
xtd_im_ch_bounds[:, 0] = np.floor(xtd_im_ch_bounds[:, 0])
xtd_im_ch_bounds[:, 1] = np.ceil(xtd_im_ch_bounds[:, 1])
xtd_im_ch_bounds = xtd_im_ch_bounds.astype(int)
xtd_im_ch_bounds
```

```{note}
Though we demonstrate how to convert energy to channel bounds above, the wrapper
function for image generation will repeat this exercise, as it will write
energy bounds into output file names.
```

#### Image binning factor

When generating images, you might wish to bin the event X-Y sky coordinate system so
that one pixel of the output image represents a grouping of 'event pixels'.

This binning could be motivated by increasing the signal-to-noise of each pixel or
reducing the size of the output image file, or your own scientific purpose.

It is worth noting that the Xtend **event pixel** size dramatically subsamples the
point-spread-function (PSF) size induced by the X-ray optics, so an extreme binning
factor would be required to minimize cross-talk between image pixels. As such, this
should not be the primary motivation for your choice of image binning factor.

```{code-cell} python
bin_factors = [1, 4]
```

#### Running image generation

There is no HEASoft tool specifically for generating XRISM-Xtend images, but there is a
generalized HEASoft image (and other data product) generation task that we can use.

If you have previously generated images, light curves, or spectra from HEASARC-hosted
X-ray data on the command line, you may well have come across `XSELECT`; a HEASoft
tool for interactively generating data products from event lists.

When creating data products, `XSELECT` calls the HEASoft `extractor` task, which we
will now use to demonstrate the creation of XRISM-Xtend images.

As with all uses of HEASoft tasks in this notebook, our call to `extractor` will be
through the HEASoftPy Python interface - specifically the `hsp.extractor` function.

We have implemented a wrapper to this function in the 'Global Setup: Functions' section
of this notebook, primarily so that we can easily multiprocess the generation of images
in different energy bands, binning factors, observations, and dataclasses.

Image generation is not a particularly computationally intensive task, but if you are
addressing a large number of observations (or making many images per observation), it
is a good idea to run them in parallel!

```{code-cell} python
arg_combs = [
    [
        EVT_PATH_TEMP.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        *cur_bnds,
        cur_bf,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
    for cur_bnds in xtd_im_en_bounds
    for cur_bf in bin_factors
]

with mp.Pool(NUM_CORES) as p:
    im_result = p.starmap(gen_xrism_xtend_image, arg_combs)
```

### New XRISM-Xtend exposure maps

We also generate exposure maps for the entire FoV of a particular observation mode, rather
than for a particular source. The exposure map gives us the exposure time (unsurprisingly)
at any given pixel of our image (assuming the image and exposure map are
binned the same way). Exposure maps are also a useful way to tell exactly which parts of the sky
are covered by the observation.

The latter capability is of particular importance for the generation/analysis of
spectra, and the creation of Ancillary Response Files (ARFs), which describe the
effective sensitivity of Xtend as a function of energy.

Unlike for image creation, XRISM does have a dedicated HEASoft task for the
generation of exposure maps; `xaexpmap`. We have once again set up a wrapper function
in the 'Global Setup: Functions' section of this notebook to make it easier to run
this task in parallel.

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

Our wrapper function for `xaexpmap` also contains an optional call to the HEASoft FTOOLS `fimgbin` task, which
is used to re-bin the exposure map to a coarser spatial resolution (in this case to match the second
binning factor we generated images for).

The `xaexpmap` task creates exposure maps with 1 image pixel per Sky X-Y coordinate system pixel, so we
only need to specify binning factors **that do not equal 1** here (adding 1 to this
list would be redundant and would waste compute time):

```{code-cell} python
expmap_bin_factors = [4]
```

Finally, we run the exposure map generation:

```{code-cell} python
arg_combs = [
    [
        EVT_PATH_TEMP.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        EVT_PATH_TEMP.format(oi=oi, xdc=dc, sc=0),
        ehk_path_temp.format(oi=oi),
        BADPIX_PATH_TEMP.format(oi=oi, xdc=dc, sc=0),
        "NONE",
        cur_bf,
        expmap_rad_delta,
        expmap_phi_bins,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
    for cur_bf in expmap_bin_factors
]

with mp.Pool(NUM_CORES) as p:
    ex_result = p.starmap(gen_xrism_xtend_expmap, arg_combs)
```

## 4. Handling observations with multiple dataclasses

As we've already mentioned, the XRISM-Xtend detector can operate in two different
data modes simultaneously, with the read-out area of some of the CCDs restricted in
order to increase the read-out speed and improve timing resolution.

If the proposer of the observation has requested an increased-timing-resolution
mode, then odds are the target of their observation will be placed on that
fast-timing-mode portion of the detector.

As such, if that is the target you are also interested in (whether you are the
original proposer or are performing some archival analysis), then the other half
of the XRISM-Xtend detector may not be that interesting to you.

In such cases we will have to decide which dataclass we're going to use from
this point onwards.

The first part of the next section of this demonstration will show you how to set up
the regions from which source and background data products will be extracted.

Event lists from the same observation with different dataclasses are taken from
different halves of the detector and are not co-aligned. That means the source region
we're about to set up for our target will not overlap with the coverage of the other
dataclass.

That would result in empty light curve and spectrum products, and errors
from 'BACKSCAL' calculation, and RMF and ARF generation.


### Visualize separate XRISM-Xtend 000128000 dataclass images

To make our point, and to give an example of the inspection you may want to perform
before choosing the right dataclass for your target, we will visualize
the 0.6–10.0 keV images we generated for 000128000 in the last section:

```{code-cell} python
# Set up the path to the image, and XGA Image class instance, for the '31100010'
#  dataclass, which is the small-window, fast-readout, mode
fast_im_path = IM_PATH_TEMP.format(
    oi="000128000", xdc="31100010", lo="0.6", hi="10.0", ibf=1
)
fast_im = Image(
    fast_im_path,
    "000128000",
    "Xtend",
    "",
    "",
    "",
    Quantity(0.6, "keV"),
    Quantity(10.0, "keV"),
)

# Set up the path to the image, and XGA Image class instance, for the '32000010'
#  dataclass, which is rest of the Xtend detector running as normal
half_im_path = IM_PATH_TEMP.format(
    oi="000128000", xdc="32000010", lo="0.6", hi="10.0", ibf=1
)
half_im = Image(
    half_im_path,
    "000128000",
    "Xtend",
    "",
    "",
    "",
    Quantity(0.6, "keV"),
    Quantity(10.0, "keV"),
)
```

We can quite clearly see that the source of interest, indicated by the
cross-hair, is on the small-window, fast-readout, '31100010' dataclass image:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
fig, ax_arr = plt.subplots(ncols=2, figsize=(6, 6), width_ratios=[3, 1])
plt.subplots_adjust(wspace=0)

half_im.get_view(ax_arr[0], zoom_in=True, custom_title="32000010")
fast_im.get_view(ax_arr[1], src_coord_quant, zoom_in=True, custom_title="31100010")

plt.show()
```

### Choosing our dataclasses

Based on our inspection, we choose the right dataclass for the '000128000' observation:

```{code-cell} python
rel_dataclasses = {"000128000": ["31100010"]}
```

```{hint}
Manual inspection is far from the only way of making this choice - it would be easy to
automate this process by, for instance, retrieving the exposure value corresponding to
the source position from the exposure map; a non-zero exposure for a particular dataclass
would indicate that it contains your source.
```

## 5. Generating new XRISM-Xtend spectra and light curves

In this section we will demonstrate how to generate source-specific data products from
XRISM-Xtend observations; light curves and spectra (along with supporting files like
RMFs and ARFs).

Rather than extracting spectra and light curves for the entire XRISM-Xtend FoV,
*which is how the quick-look spectra and light curves contained in the archive are made*, we
want to control exactly where we are taking events from.

That way we can focus on the particular source(s) of interest present in the
XRISM-Xtend observations we are using.

The size, shape, placement, and number of source extraction regions you need to use for
your work will depend heavily on your science case and the type of astrophysical
source you're analyzing.

You will find that point sources are considerably easier to deal with, as you can
generally learn all you need from a single spectrum encompassing the entire source
emission region.

Indeed, trying to extract spectra from different spatial regions of a point source (even
if the emission *appears* extended in XRISM-Xtend images) is **not valid**, as the
apparently extended emission is caused by the PSF of the telescope optics.

The 'blurring' of the observed emission events by the PSF is one of the reasons that
extended sources are much harder to analyze than point sources. For example, you
might want to extract spectra from a series of annular bins centered on
your extended source to see how a particular spectral property changes in different
parts of the object.

Unfortunately, due to the PSF, each annulus will be contaminated by (and be
*contaminating* in turn) events from other annuli, scattered there by the telescope PSF - this
effect is sometimes referred to as **cross-talk** or **spatial-spectral mixing (SSM)**. Accounting
for this effect is complicated and time-consuming, so our demonstration will focus on a point source, and
extended sources will be discussed in another notebook.

### Setting up source and background extraction regions

To define exactly where we want to extract events from to build our data products, we
will construct 'region files' in the commonly used 'DS9' format.

In this demonstration we will not provide guidance on how to choose particular
source/background regions for your science case or give detailed information
about the DS9 region format and its capabilities.

Instead, we will show you how to construct basic region files using the
astropy-affiliated `regions` module.

Most high-energy missions use three common coordinate systems:
- **Detector (DET) X-Y** - This coordinate system is aligned with the detector; a coordinate in this system will always represent the same physical position on the detector.
- **Sky X-Y** - A transformed version of the DETX-DETY coordinate system, aligned with the roll angle of the telescope. **Within a single observation**, a Sky X-Y coordinate will always represent the same physical position on the sky.
- **RA-DEC** - The familiar right ascension and declination coordinate system.

You need to be careful about which coordinate system you use with which tools, as some
tasks will not accept regions in all coordinate systems.

#### Excluding the XRISM-Xtend calibration sources

The XRISM-Xtend instrument has two calibration sources in its FoV, which present as
bright circles on opposite edges of the detector. While highly useful for the
calibration of Xtend's energy scale, we do not want to accidentally include
calibration events in spectra or light curves we extract.

As such, we have to define regions to exclude these sources from our data products.

This will be quite simple, as HEASoft includes a pre-made region file that defines the
location and extent of the emission from the calibration sources.

A small difficulty arises from the fact that this pre-made region file is defined in
detector coordinates, rather than the RA-Dec coordinates we're going to use for the
source region. That will be pretty easy to deal with, however, as HEASoft includes a
tool to transform region files between different coordinate systems.

```{code-cell} python
# The path to the HEASoft-supplied XRISM-Xtend calibration source region file
detpix_xtend_calib_reg_path = os.path.join(
    os.environ["HEADAS"], "refdata", "calsrc_XTD_det.reg"
)

# Setting up the output file for the RA-Dec calibration source regions
radec_xtend_calib_reg_path = os.path.join(OUT_PATH, "{oi}", "radec_{oi}_calsrc.reg")
```

We only need to generate one RA-Dec calibration source region file for each
observation, as the observations with data from multiple data modes share a common
coordinate system.

The HEASoft tool `coordpnt` is used to perform the transformation from detector to
RA-Dec coordinates - it requires pointing coordinate and telescope roll angle
information which we extract from previously generated image files.

```{code-cell} python
chos_im_en = xtd_im_en_bounds[2]

for oi, dcs in rel_dataclasses.items():
    cur_im_path = IM_PATH_TEMP.format(
        oi=oi, xdc=dcs[0], ibf=1, lo=chos_im_en[0].value, hi=chos_im_en[1].value
    )
    with fits.open(cur_im_path) as imago:
        cur_pnt_ra = imago[0].header["RA_PNT"]
        cur_pnt_dec = imago[0].header["DEC_PNT"]
        cur_roll_ang = imago[0].header["PA_NOM"]

    # Call the HEASoft task that converts from detector coordinates to RA-Dec
    hsp.coordpnt(
        input=detpix_xtend_calib_reg_path,
        outfile=radec_xtend_calib_reg_path.format(oi=oi),
        telescop="XRISM",
        instrume="XTEND",
        ra=cur_pnt_ra,
        dec=cur_pnt_dec,
        roll=cur_roll_ang,
        startsys="DET",
        stopsys="RADEC",
        clobber=True,
    )
```

Finally, we have to pull the RA-Dec calibration regions from the transformed region
file. The `regions` module provides functions to read-in region files in
various formats and coordinate systems; it also understands that the regions in
these files are to be excluded (indicated by a '-' prefix).

Note that we also set each calibration region's color to white and line-style to
'dotted'. This is so that when we plot them on the same image as the source and
background regions (a little later in this notebook) they will be visually distinct.

```{code-cell} python
cal_regs = {}
for oi in rel_obsids:
    cur_cal_regs = Regions.read(radec_xtend_calib_reg_path.format(oi=oi), format="ds9")
    for cur_reg in cur_cal_regs:
        cur_reg.visual["color"] = "white"
        cur_reg.visual["linestyle"] = "dotted"
        # Make sure the frame is consistent with the source/back regions later, as
        #  otherwise HEASoft tools will get confused
        cur_reg.center = cur_reg.center.transform_to("icrs")

    # The '.regions' just retrieves a list of region objects, we don't need to keep
    #  the calibration regions in the regions module 'Regions' class they are read into
    cal_regs[oi] = cur_cal_regs.regions

cal_regs
```

#### Source and background RA-DEC region files

We define a `CircleSkyRegion` instance (a class of the `regions` module) centered on
our target source, with a radius of 2 arcminutes.

```{code-cell} python
# The radius of the source extraction region
src_reg_rad = Quantity(2, "arcmin")

# Setting up a 'regions' module circular sky region instance
src_reg = CircleSkyRegion(src_coord, src_reg_rad, visual={"color": "green"})
```

We do the same to define a region from which to extract a background spectrum, though
this region is of a different size and is not centered on the source. We also set a
line-style for the background region, to distinguish it from the source and
calibration regions:

```{code-cell} python
# The central coordinate of the background region
back_coord = SkyCoord(81.1932474, -69.5073738, unit="deg", frame="icrs")

# The radius of the background region
back_reg_rad = Quantity(3, "arcmin")

# Setting up a 'regions' module circular sky region instance for the background region
back_reg = CircleSkyRegion(
    back_coord, back_reg_rad, visual={"color": "red", "linestyle": "dashed"}
)
```

#### Visualizing the source and background extraction regions on XRISM-Xtend images

We should inspect the regions in-situ to make sure they look sensible - first, our
previously generated images are loaded in as `Image` class (from the `XGA` Python
module) instances.

The regions we created are then assigned to each image's `regions`, and
the `.view()` method is called with the `view_regions=True` argument to display them.

Additionally, we extract the RA-Dec ↔ Sky X-Y WCS from one image per observation so
that we can use it later on to transform our RA-Dec regions into Sky X-Y regions.

```{code-cell} python
chos_im_en = xtd_im_en_bounds[0].to("keV")

oi_skypix_wcs = {}
for oi, cur_dcs in rel_dataclasses.items():
    for dc in cur_dcs:
        cur_im_path = IM_PATH_TEMP.format(
            oi=oi, xdc=dc, ibf=1, lo=chos_im_en[0].value, hi=chos_im_en[1].value
        )
        cur_im = Image(cur_im_path, oi, "Xtend", "", "", "", *chos_im_en)
        cur_im.regions = [src_reg, back_reg] + cal_regs[oi]
        cur_im.view(src_coord_quant, zoom_in=True, view_regions=True)

        oi_skypix_wcs.setdefault(oi, cur_im.radec_wcs)
```

#### Writing observation-specific RA-Dec and sky-pixel coordinate region files

Now we've set up the regions and visualized them, we'll write them to disk as region
files that can be passed to the HEASoft tasks used to generate spectra and light curves.

We set up instances of  the `Regions` class of the astropy-affiliated `regions` module
for the source and background regions plus calibration regions. The `write()` method
is then used to save a DS9-formatted region file to disk.

The calibration source regions are set up to be excluded, and the output files will
reflect that.

We write two versions each of the source and region files, one version in the RA-Dec
coordinate system, and the other in the Sky X-Y system (different tasks have different
requirements for the coordinate system they accept).

Our RA-Dec regions are converted to Sky X-Y using a feature of the `regions`
module, using the WCS information we pulled from the images when we visualized them
with the source, background, and calibration source regions overplotted

```{code-cell} python
# Where to write the new RA-Dec source region file - the double {{}} around 'oi' just
#  means that the f-string will fill in the SRC_NAME and leave '{oi}' to be
#  formatted later
radec_src_reg_path = os.path.join(OUT_PATH, "{oi}", f"radec_{{oi}}_{SRC_NAME}_src.reg")
# Where to write the new RA-Dec background region file
radec_back_reg_path = os.path.join(
    OUT_PATH, "{oi}", f"radec_{{oi}}_{SRC_NAME}_back.reg"
)

# The file path templates for the source and background Sky X-Y system region files
obs_src_reg_path_temp = os.path.join(
    OUT_PATH, "{oi}", f"skypix_{{oi}}_{SRC_NAME}_src.reg"
)
obs_back_reg_path_temp = os.path.join(
    OUT_PATH, "{oi}", f"skypix_{{oi}}_{SRC_NAME}_back.reg"
)

for oi in rel_obsids:
    # We set up a combination of the source region and the calibration source
    #  regions - the calibration regions have been set up to be excluded, which will
    #  be reflected in the final region files we write
    src_comb_regs = Regions([src_reg] + cal_regs[oi])
    # The same but for the background region
    back_comb_regs = Regions([back_reg] + cal_regs[oi])

    # Write the RA-Dec source region file
    src_comb_regs.write(radec_src_reg_path.format(oi=oi), format="ds9", overwrite=True)
    # And the RA-Dec background region file
    back_comb_regs.write(
        radec_back_reg_path.format(oi=oi), format="ds9", overwrite=True
    )

    # Now we repeat the exercise, but use the Sky<->RA-Dec WCS information we pulled
    #  from the images to convert the regions to the Sky X-Y system
    src_comb_skyXY_regs = Regions(
        [r.to_pixel(oi_skypix_wcs[oi]) for r in src_comb_regs]
    )
    back_comb_skyXY_regs = Regions(
        [r.to_pixel(oi_skypix_wcs[oi]) for r in back_comb_regs]
    )

    src_comb_skyXY_regs.write(
        obs_src_reg_path_temp.format(oi=oi), format="ds9", overwrite=True
    )
    back_comb_skyXY_regs.write(
        obs_back_reg_path_temp.format(oi=oi), format="ds9", overwrite=True
    )
```

```{tip}
Events from different data classes of **the same observation** share a common sky-pixel
coordinate system, so sky-pixel region files for one are also valid for the other.
However, different data classes represent different pairs of Xtend CCDs, so there is
no shared sky coverage.
```

### New XRISM-Xtend spectra and supporting files

With the source and background regions defined, we have everything we need to generate
XRISM-Xtend spectra for our target source.

Though there is no XRISM-Xtend-specific HEASoft task for spectrum generation, we can
use the same tool that we used to create our images in Section 3, and that we will use
later in this section to make light curves - `extractor`.

Spectral files are just as simple a data product as images and light curves. All we're
doing to make a spectrum is binning the events in our extraction regions in a 1D
channel space (which will become energy), rather than X-Y, or time. The complexities
of spectral generation come when creating the supporting files (RMFs and ARFs)
required to actually fit models to the spectra.

#### Generating the spectral files

We set up a spectrum-generation-specific region wrapper function for the HEASoftPy
interface to the `extractor` task (see the Global Setup: Functions section near the
top of the notebook).

This once again allows us to parallelize the generation of spectra for different
observations, though it is worth noting we aren't producing multiple variants of the
spectrum like we did with the different-energy-band images.

```{code-cell} python
arg_combs = [
    [
        EVT_PATH_TEMP.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        src_coord,
        src_reg_rad,
        obs_src_reg_path_temp.format(oi=oi),
        obs_back_reg_path_temp.format(oi=oi),
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
]

with mp.Pool(NUM_CORES) as p:
    sp_result = p.starmap(gen_xrism_xtend_spectrum, arg_combs)
```

#### Calculating 'BACKSCAL' for new XRISM-Xtend spectra

Spectral data products generated for high-energy missions typically contain a
measurement of their extraction region area. This is in order to scale source and
background spectra properly when

Our calculation of 'BACKSCAL' doesn't only benefit our spectral analyses, as when we
demonstrate the creation of light curves later in this notebook, we can also use
the values to weight our subtraction of the background.

```{code-cell} python
spec_backscals = {oi: {dc: 0 for dc in rel_dataclasses[oi]} for oi in rel_obsids}
bspec_backscals = {oi: {dc: 0 for dc in rel_dataclasses[oi]} for oi in rel_obsids}

for oi, dcs in rel_dataclasses.items():
    for cur_dc in dcs:
        # Set up the path to input source and background spectra
        cur_spec = SP_PATH_TEMP.format(
            oi=oi,
            xdc=cur_dc,
            ra=src_coord.ra.value.round(6),
            dec=src_coord.dec.value.round(6),
            rad=src_reg_rad.to("deg").value.round(4),
        )
        cur_bspec = BACK_SP_PATH_TEMP.format(
            oi=oi,
            xdc=cur_dc,
            ra=src_coord.ra.value.round(6),
            dec=src_coord.dec.value.round(6),
        )

        # Also need to pass an exposure map, so set up a path to that
        cur_ex = EX_PATH_TEMP.format(
            oi=oi,
            xdc=cur_dc,
            rd=expmap_rad_delta.to("arcmin").value,
            npb=expmap_phi_bins,
            ibf=1,
        )

        # Calculate the BACKSCAL keyword, first for the source spectrum
        hsp.ahbackscal(
            infile=cur_spec,
            regfile=obs_src_reg_path_temp.format(oi=oi),
            expfile=cur_ex,
            logfile="NONE",
        )

        # Then for the background spectrum
        hsp.ahbackscal(
            infile=cur_bspec,
            regfile=obs_back_reg_path_temp.format(oi=oi),
            expfile=cur_ex,
            logfile="NONE",
        )

        # For good measure, and because we're going to need them later for
        #  net light curve calculation, we read the backscal values into Python
        # First, the source spectrum
        with fits.open(cur_spec) as src_specco:
            spec_backscals[oi][cur_dc] = src_specco["SPECTRUM"].header["BACKSCAL"]
        # Now the background
        with fits.open(cur_bspec) as back_specco:
            bspec_backscals[oi][cur_dc] = back_specco["SPECTRUM"].header["BACKSCAL"]
```

Showing the BACKSCAL values:

```{code-cell} python
print(spec_backscals)
print(bspec_backscals)
```

#### Grouping our new spectra

We will group the spectra we just generated. Grouping essentially combines
spectral channels until some minimum quality threshold is reached; in this case a
minimum of one count per grouped channel. We use the HEASoft `ftgrouppha` tool to do
this, once again through HEASoftPy.

First, we set up the grouping criteria and a template variable for the name of the
output grouped spectral files:

```{code-cell} python
spec_group_type = "min"
spec_group_scale = 1
```

Now we run the grouping tool - though this time we do not parallelize the task, as
the grouping process is fast, and we wish to demonstrate how you use a HEASoftPy
function directly. Though remember to look at the Global Setup section of this notebook
to see how we call HEASoftPy tools in the wrapper functions used to parallelize those
tasks.

If you are dealing with significantly more observations than we use for this
demonstration, we do recommend that you parallelize this grouping step as we have
the other processing steps in this notebook.

```{code-cell} python
for oi, dcs in rel_dataclasses.items():
    for cur_dc in dcs:
        # Set up relevant paths to the input and output spectrum
        cur_spec = SP_PATH_TEMP.format(
            oi=oi,
            xdc=cur_dc,
            ra=src_coord.ra.value.round(6),
            dec=src_coord.dec.value.round(6),
            rad=src_reg_rad.to("deg").value.round(4),
        )
        cur_grp_spec = GRP_SP_PATH_TEMP.format(
            oi=oi,
            xdc=cur_dc,
            gt=spec_group_type,
            gs=spec_group_scale,
            ra=src_coord.ra.value.round(6),
            dec=src_coord.dec.value.round(6),
            rad=src_reg_rad.to("deg").value.round(4),
        )

        hsp.ftgrouppha(
            infile=cur_spec,
            outfile=cur_grp_spec,
            grouptype=spec_group_type,
            groupscale=spec_group_scale,
        )
```

#### Generating XRISM-Xtend RMFs

In order for the spectral model fitting software of our choice (XSPEC, for
instance) to be able to map spectrum channels to energies, we need to
generate RMFs.

We have already discussed RMFs and even used them to perform our own conversion between
XRISM-Xtend spectral channels and energy, in Section 3 - there we used the RMF that
was included in the original data download.

Now we wish to generate new RMFs, so we can ensure they are entirely up to date!

We make use of the XRISM-Xtend specific HEASoft task `xtdrmf` - the only input it
requires is the path to the spectral file for which we wish to generate an RMF.

Our `gen_xrism_xtend_rmf` function (defined in the Global Setup: Functions section near
the top of the notebook) wraps the HEASoftPy interface to the `xtdrmf` task. We now use
it to generate RMFs in parallel for all of our new spectra:

```{code-cell} python
arg_combs = [
    [
        SP_PATH_TEMP.format(
            oi=oi,
            xdc=dc,
            ra=src_coord.ra.value.round(6),
            dec=src_coord.dec.value.round(6),
            rad=src_reg_rad.to("deg").value.round(4),
        ),
        os.path.join(OUT_PATH, oi),
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
]

with mp.Pool(NUM_CORES) as p:
    rmf_result = p.starmap(gen_xrism_xtend_rmf, arg_combs)
```

#### Generating XRISM-Xtend ARFs

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
has been altered across its energy range by how good XRISM-Xtend is at detecting
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
that it can be sped up, though at the cost of that accuracy - the most direct way is
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

So now we move onto actually running the ARF generation - using the
`gen_xrism_xtend_arf` function defined in the Global Setup: Functions section (near the top of
the notebook), which wraps the HEASoftPy interface to the `xaarfgen` task. We now use it
to generate ARFs in parallel for all of our new spectra:

```{code-cell} python
arg_combs = [
    [
        os.path.join(OUT_PATH, oi),
        EX_PATH_TEMP.format(
            oi=oi,
            xdc=dc,
            rd=expmap_rad_delta.to("arcmin").value,
            npb=expmap_phi_bins,
            ibf=1,
        ),
        SP_PATH_TEMP.format(
            oi=oi,
            xdc=dc,
            ra=src_coord.ra.value.round(6),
            dec=src_coord.dec.value.round(6),
            rad=src_reg_rad.to("deg").value.round(4),
        ),
        RMF_PATH_TEMP.format(oi=oi, xdc=dc),
        radec_src_reg_path.format(oi=oi),
        arf_rt_num_photons,
        arf_rt_min_photons,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
]

with mp.Pool(NUM_CORES) as p:
    arf_result = p.starmap(gen_xrism_xtend_arf, arg_combs)
```

```{warning}
Due to the high-fidelity ray-tracing method used to calculate XRISM ARFs, the runtime
of this step can be on the order of hours.
```

### New XRISM-Xtend light curves

Now we can quickly demonstrate how to generate XRISM-Xtend light curves - it is
rather simpler than the creation of new spectra.

There is no XRISM-Xtend-specific task for the generation of light curves, so we
once again turn to HEASoft's `extractor` tool (we used it to create XRISM-Xtend
images in Section 3).

By providing a slightly different set of inputs to `extractor`, we can tell it to
bin the cleaned event lists in time, rather than in space, and thus produce a
light curve.

We'll make sure to generate light curves within the source and background regions
that we defined in the previous part of this section, which we can then use to
produce net light curves for our source.

The primary input we need to provide is the time step, or time bin size, which
controls the temporal resolution of the output light curve. This uniform sampling is
the simplest method of dividing an event list into a light curve, but other methods
exist (requiring each time bin to reach a minimum signal-to-noise, for instance).

Your choice of uniform time bin size will depend on your particular science case, as
well as practical considerations based on the length of the overall observation and the
observed count-rate of the source.

```{code-cell} python
lc_time_bin = Quantity(200, "s")
```

It is also very common to want to specify the events included in each by setting
lower and upper energy bounds - this may allow you to focus on light emitted by
a particular process you're interested in, or to exclude energy bands that are
not relevant to your science case.

We define three energy bands that cover much of the useful energy range of the
XRISM-Xtend instrument:
- 0.6-2.0 keV
- 2.0-6.0 keV
- 6.0-10.0 keV

Though again, your choices will depend on what you're trying to learn.

```{code-cell} python
# Defining the various energy bounds we want to make light curves for
xtd_lc_en_bounds = Quantity([[0.6, 2.0], [2.0, 6.0], [6.0, 10.0]], "keV")
```

#### Generating source and background light curves

Using another wrapper function around the HEASoftPy interface to `extractor`, we can
now generate the light curves within the source and background regions, for each of
the specified energy bands.

As with previous steps, our motivation for writing a wrapper function (defined in the
Global Setup section) is to make it easy for us to run generation of different
light curves simultaneously:

```{code-cell} python
arg_combs = [
    [
        EVT_PATH_TEMP.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        src_coord,
        src_reg_rad,
        obs_src_reg_path_temp.format(oi=oi),
        obs_back_reg_path_temp.format(oi=oi),
        *cur_bnds,
        lc_time_bin,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
    for cur_bnds in xtd_lc_en_bounds
]

with mp.Pool(NUM_CORES) as p:
    lc_result = p.starmap(gen_xrism_xtend_lightcurve, arg_combs)
```

#### Calculating net light curves

Unlike with the spectra we generated earlier in this section, we will produce 'net'
light curves, with the background light curve scaled and subtracted from the source.

The applied scaling is to effectively normalize the area within which the background
light curve was extracted to the source light curve.

We have already performed the measurement of the extraction regions for source and
background products - when we used HEASoft to calculate the 'BACKSCAL' keyword after
our spectra were generated.

At the time we made sure to read those 'BACKSCAL' values into Python, specifically
for this purpose.

With the scaling known, we can use the HEASoft `lcmath` tool (through the HEASoftPy
interface) to subtract the background from the source light curve. This operation is
computationally cheap for the number of light curves we are working with, but you
should consider parallelizing this step if you are working with significantly more:

```{code-cell} python
for oi, dcs in rel_dataclasses.items():
    for cur_dc in dcs:
        for cur_bnds in xtd_lc_en_bounds:
            # Constructing the file paths to the source and background light curves
            cur_lc = LC_PATH_TEMP.format(
                oi=oi,
                xdc=dc,
                ra=src_coord.ra.value.round(6),
                dec=src_coord.dec.value.round(6),
                rad=src_reg_rad.to("deg").value.round(4),
                lo=cur_bnds[0].value,
                hi=cur_bnds[1].value,
                lct=0.0,
                tb=lc_time_bin.to("s").value,
            )

            cur_blc = BACK_LC_PATH_TEMP.format(
                oi=oi,
                xdc=dc,
                ra=src_coord.ra.value.round(6),
                dec=src_coord.dec.value.round(6),
                lo=cur_bnds[0].value,
                hi=cur_bnds[1].value,
                lct=0.0,
                tb=lc_time_bin.to("s").value,
            )

            # Now we construct the output file path for the final net light curve
            cur_nlc = NET_LC_PATH_TEMP.format(
                oi=oi,
                xdc=dc,
                ra=src_coord.ra.value.round(6),
                dec=src_coord.dec.value.round(6),
                rad=src_reg_rad.to("deg").value.round(4),
                lo=cur_bnds[0].value,
                hi=cur_bnds[1].value,
                lct=0.0,
                tb=lc_time_bin.to("s").value,
            )

            with contextlib.chdir(
                os.path.join(OUT_PATH, oi)
            ), hsp.utils.local_pfiles_context():
                # The 'lcmath' tool is sensitive to long paths, so we fetch the
                #  relative paths to pass it instead of the absolute paths
                cur_lc = os.path.relpath(cur_lc)
                cur_blc = os.path.relpath(cur_blc)
                cur_nlc = os.path.relpath(cur_nlc)

                # Calculate the scaling that should be applied to the background
                #  light curve before subtraction
                cur_back_multi = (
                    spec_backscals[oi][cur_dc] / bspec_backscals[oi][cur_dc]
                )

                # Run the tool to produce a net light curve
                hsp.lcmath(
                    infile=cur_lc,
                    bgfile=cur_blc,
                    outfile=cur_nlc,
                    multi=1,
                    multb=cur_back_multi,
                )
```

#### Loading and displaying a single light curve

We take a quick look at one of the light curves we just generated to make sure it
looks sensible. First, we specify the ObsID and dataclass of the light curve we will
use as a demonstration, as well as the energy band:

```{code-cell} python
chosen_demo_lc_obsid = "000128000"
chosen_demo_lc_dataclass = "31100010"
chosen_demo_lc_bnds = Quantity([0.6, 2.0], "keV")
```

Now we set up the path and load the light curve into an XGA LightCurve object, as
it has a convenient method to generate visualizations. You could very easily load
the light curve data in directly, using astropy.io.fits, and then plot it yourself:

```{code-cell} python
# Define the path to the demo net light curve
demo_lc_path = NET_LC_PATH_TEMP.format(
    oi=chosen_demo_lc_obsid,
    xdc=chosen_demo_lc_dataclass,
    ra=src_coord.ra.value.round(6),
    dec=src_coord.dec.value.round(6),
    rad=src_reg_rad.to("deg").value.round(4),
    lo=chosen_demo_lc_bnds[0].value,
    hi=chosen_demo_lc_bnds[1].value,
    lct=0.0,
    tb=lc_time_bin.value,
)

# Set up a XGA LightCurve instance
demo_lc = LightCurve(
    demo_lc_path,
    chosen_demo_lc_obsid,
    "Xtend",
    "",
    "",
    "",
    src_coord_quant,
    Quantity(0, "arcmin"),
    Quantity(2.0, "arcmin"),
    *chosen_demo_lc_bnds,
    lc_time_bin,
)

# Show a visualization of the LightCurve
demo_lc.view()
```

## 6. Fitting a spectral model to an XRISM-Xtend spectrum

Finally, to show off the XRISM-Xtend products we just generated, we will perform
a simple model fit to one of our spectra.

Our demonstration of spectral model fitting to an XRISM-Xtend spectrum will be
performed using the [PyXspec](https://heasarc.gsfc.nasa.gov/docs/software/xspec/python/html/index.html) package.

### Configuring PyXspec

Now we configure some behaviors of XSPEC/PyXspec:
- The ```chatter``` parameter is set to zero to reduce printed output during fitting (note that some XSPEC messages are still shown).
- We inform XSPEC of the number of cores we have available, as some XSPEC methods can be paralleled.
- We tell XSPEC to use the Cash statistic for fitting (the reason we grouped our spectra earlier).

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

```{danger}
There is a known issue with the version of PyXspec shipped in HEASoft v6.36 (and
possibly later versions) that can cause the parallelised generation of data products
to hang forever. We avoid this here by importing PyXspec **after** all data product
generation is complete.
```

### Reading a XRISM-Xtend spectrum into PyXspec

Here we define the ObsID and dataclass of the spectrum we want to fit:

```{code-cell} python
chosen_demo_spec_obsid = "000128000"
chosen_demo_spec_dataclass = "31100010"
```

The spectrum, and all of its supporting files, are then read into PyXspec:

```{code-cell} python
# In case this cell is re-run, clear all previously loaded spectra
xs.AllData.clear()

# Set up the paths to grouped source spectrum, ungrouped background
#  spectrum, RMF, and ARF files
cur_spec = GRP_SP_PATH_TEMP.format(
    oi=chosen_demo_spec_obsid,
    xdc=chosen_demo_spec_dataclass,
    gt=spec_group_type,
    gs=spec_group_scale,
    ra=src_coord.ra.value.round(6),
    dec=src_coord.dec.value.round(6),
    rad=src_reg_rad.to("deg").value.round(4),
)

cur_bspec = BACK_SP_PATH_TEMP.format(
    oi=chosen_demo_spec_obsid,
    xdc=chosen_demo_spec_dataclass,
    ra=src_coord.ra.value.round(6),
    dec=src_coord.dec.value.round(6),
)

cur_rmf = RMF_PATH_TEMP.format(
    oi=chosen_demo_spec_obsid,
    xdc=chosen_demo_spec_dataclass,
)

cur_arf = ARF_PATH_TEMP.format(
    oi=chosen_demo_spec_obsid,
    xdc=chosen_demo_spec_dataclass,
    ra=src_coord.ra.value.round(6),
    dec=src_coord.dec.value.round(6),
    rad=src_reg_rad.to("deg").value.round(4),
)

# Load the chosen spectrum (and all its supporting files) into PyXspec
xs_spec = xs.Spectrum(cur_spec, backFile=cur_bspec, respFile=cur_rmf, arfFile=cur_arf)
```

### Restricting the spectral channels used for fitting

When we analyze a spectrum by fitting a model, we often want to apply lower and
upper energy limits to fit the model using only a subset of the data points.

Restricting the spectrum data points by energy allows us to cut out parts of the
spectrum that, for instance, have very low signal-to-noise, aren't relevant to our
science case, or fall outside the optimal energy range of the instrument.

Remember that XRISM-Xtend data are not currently trustworthy around or below 0.4 keV, so
we definitely want to exclude that part of the energy range. If we didn't, then we would be
in danger of biasing our model fitting results, leading to unreliable or unphysical
conclusions about our source of interest.

Here, we only make use of channels within a 0.5–10.0 keV energy range, and we also
ignore any channels that have been marked as 'bad' by any previous processing steps:

```{code-cell} python
xs_spec.ignore("**-0.5 10.0-**")

# Ignore any channels that have been marked as 'bad'
# This CANNOT be done on a spectrum-by-spectrum basis, only after all spectra
#  have been declared
xs.AllData.ignore("bad")
```

### Setting up a spectral model

Now we choose the spectral model we want to fit to our spectrum.

A full list of XSPEC model components can be found in the [XSPEC documentation](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node128.html).

Our choice of model is empirically driven, chosen by someone who is not a specialist in supernova remnants, and should definitely not be considered as scientifically useful!

```{code-cell} python
xs.Model("vapec+vrnei+powerlaw")
```

If we temporarily increase PyXspec's chatter level, we can see the default values
of each model's parameters:

```{code-cell} python
xs.Xset.chatter = 10
xs.AllModels.show()
xs.Xset.chatter = 0
```

### Fitting our PyXspec model to the XRISM-Xtend spectrum

Performing the fit is simple:

```{code-cell} python
xs.Fit.perform()
```

We once again temporarily increase the chatter level and display the fitted parameters:

```{code-cell} python
xs.Xset.chatter = 10
xs.AllModels.show()
xs.Xset.chatter = 0
```

### Visualizing the fitted spectrum

We want to use matplotlib to visualize the spectrum data, and the model we
just fitted to it. PyXspec allows us to extract the data it would have plotted were
we using XSPEC directly:

```{code-cell} python
# This populates plot information attributes for the current
#  spectrum and model. We can extract that information and
#  plot it using matplotlib
xs.Plot()

# These read out the plotting information for the SPECTRUM
spec_en = xs.Plot.x()
spec_en_err = xs.Plot.xErr()
spec_cr = xs.Plot.y()
spec_cr_err = xs.Plot.yErr()

# And the equivalent for the MODEL
spec_mod_cr = xs.Plot.model()
```

Now we can quite easily produce a plot of the spectrum and model:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Visualizing the fitted XRISM-Xtend spectrum
#  First, set up the figure size, configure the axis tick appearance
plt.figure(figsize=(7, 4))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

# Show the spectrum data points, with uncertainties
plt.errorbar(
    spec_en,
    spec_cr,
    xerr=spec_en_err,
    yerr=spec_cr_err,
    fmt="kx",
    capsize=2,
    alpha=0.5,
    label="XRISM-Xtend data",
    zorder=1,
)

# Overplot (zorder=2 will ensure the fit line goes over the data) the model
#  that we fit using PyXpec.
plt.plot(
    spec_en,
    spec_mod_cr,
    color="mediumorchid",
    alpha=1,
    lw=2,
    label="Fitted model",
    zorder=2,
)

# Make sure the energy axis is log scaled
plt.xscale("log")

# Alter the formatters for the energy axis so that (for instance) 10 keV is
#  displayed as '10 keV' rather than '10^1 keV'
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

# Label the X and Y axes
plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Spectrum [ct cm$^{-2}$ s$^{-1}$ keV$^{-1}$]", fontsize=15)

# Add a legend
plt.legend(loc="best", fontsize=14)

plt.tight_layout()
plt.show()
```

## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Kenji Hamaguchi, XRISM GOF Scientist.

Updated On: 2026-06-13

+++

### Additional Resources

**XRISM Help Desk**: https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=xrism

**XRISM Data Reduction (ABC) Guide**: https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide

**HEASoftPy GitHub Repository**: https://github.com/HEASARC/heasoftpy

**HEASoftPy HEASARC Page**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy.html

**HEASoft XRISM `xtdpipeline` help file**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xtdpipeline.html

**HEASoft XRISM `xaexpmap` help file**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xaexpmap.html

**HEASoft XRISM `xtdmodegti` help file**: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xtdmodegti.html

**XSPEC Model Components**: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node128.html

### Acknowledgements


### References

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE XTEND ENERGY-CHANNEL MAPPING_ [ACCESSED 25-NOV-2025]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Xtend_Data_Analysis.html#SECTION001043000000000000000)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE FILE NAMING CONVENTIONS_ [ACCESSED 11-DEC-2025]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html)

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE XTEND DATA MODES_ [ACCESSED 11-DEC-2025]](https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Xtend_SXI.html#SECTION00920000000000000000)
