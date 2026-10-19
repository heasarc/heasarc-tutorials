---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: Antara Basu-Zych
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  orcid: 0000-0001-8525-4920
  website: https://science.gsfc.nasa.gov/sci/bio/antara.r.basu-zych/
- name: Tess Jaffe
  affiliations: ['HEASARC, NASA Goddard']
  orcid: 0000-0003-2645-1339
  website: https://science.gsfc.nasa.gov/sci/bio/tess.jaffe
- name: Mike Corcoran
  affiliations: [The Catholic University of America, 'HEASARC, NASA Goddard']
  orcid: 0000-0002-7762-3172
  website: https://science.gsfc.nasa.gov/sci/bio/michael.f.corcoran
date: '2026-10-19'
execution:
  cal-files:
    xmm-ccf: false
    chandra: false
    xspec-models: false
  dependencies: [workshops/cassiopeiaa_evolution_chandra/1-data-acquisition-full-guide.md]
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
title: "Full Guide 2 \u2013 Watching a supernova remnant evolve!"
---

# Full Guide 2 – Watching a supernova remnant evolve!

## Learning Goals

By the end of this tutorial, you will be able to:

- Make image cutouts of the different data over the 20+ years of the available Chandra data.
- Put the images together to create a movie.

## Introduction

This notebook follows the `1-data-acquisition-full-guide` notebook, which downloads publicly available Chandra observations of **Cas A**.

+++

## Imports

```{code-cell} python
import os
from pprint import pprint
from warnings import filterwarnings

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from astropy.visualization import (
    ImageNormalize,
    LogStretch,
    ManualInterval,
    MinMaxInterval,
)
from astropy.wcs import WCS, FITSFixedWarning
from IPython.display import Image
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from tqdm import tqdm

filterwarnings("ignore", category=FITSFixedWarning)
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# -------------- Set paths and create directories --------------
# Set up the path of the directory into which we previously downloaded Chandra data
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "CasA-Chandra", "")
else:
    ROOT_DATA_DIR = "CasA-Chandra/"

# Whatever the data directory is, make sure it is absolute.
ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# This data path constant is pointing to where the individual images (as opposed to
#  whole observations) are stored.
IMS_ROOT_DATA_DIR = os.path.join(ROOT_DATA_DIR, "individual_files")

# And this one is where we stored the few whole observation directories we
#  downloaded as a demonstration
OBS_ROOT_DATA_DIR = os.path.join(ROOT_DATA_DIR, "full_obs_data_dirs")

# Make sure the download directory exists.
if not os.path.exists(IMS_ROOT_DATA_DIR):
    raise FileNotFoundError(
        f"The {IMS_ROOT_DATA_DIR} directory does not exist - you "
        f"must run the notebooks in order."
    )

# Finally set and create the output directory, into which we will
#  save any files we create
ROOT_OUT_PATH = os.path.abspath("demo_output")
# Make sure the download directory exists.
if not os.path.exists(ROOT_OUT_PATH):
    raise FileNotFoundError(
        f"The {ROOT_OUT_PATH} directory does not exist - you "
        f"must run the notebooks in order."
    )
# --------------------------------------------------------------
```

***

## 1. Defining our target name and position

```{code-cell} python
SRC_NAME = "Cas A"

SRC_COORD = SkyCoord.from_name(SRC_NAME, frame="icrs")
SRC_COORD
```

## 2. Loading our observation information table

+++

If you recall, at the end of the first notebook we saved our table of selected observations as a comma-separated-value (CSV) file. That file contains information that we're going to need to use in this notebook, so our first task is to read it back into an `astropy` `Table` object. As we know the name we gave to the file, and where we saved it, defining the path is easy.

That path can then be passed to the `Table.read(...)` function, which will return an instance of the `Table` class for our observation table (we used the counterpart `.write(...)` function to save the file to disk in the first place).

The only other information we give the `Table.read(...)` function goes to the `converters` argument, and is used to explicitly tell the read-in process that the contents of the ObsID column should be treated as strings, rather than integers - that is because we want to keep the prepended zeros that we added to the ObsIDs in the first notebook, and it will default to reading them as integers (and removing the zeros):

```{code-cell} python
obs_table_path = os.path.join(
    ROOT_OUT_PATH, f"{SRC_NAME.replace(' ', '')}-selected-chandra-observations.csv"
)

obs_table = Table.read(obs_table_path, converters={"obsid": str, "year_best": bool})
obs_table[:5]
```

```{admonition} Extra Task
:class: seealso
You could quickly test what happens if you remove the `converters={"obsid": str}` in the `read(...)` call above - see if you can spot the difference in the ObsID column.
```

+++

### 2.1 Make sure observations are ordered by start time

+++

Seeing as our goal is to make a GIF of the evolution of Cassiopeia A's X-ray emission, it would be quite nice to know that the observations in our `obs_table` variable are ordered from first-taken to last-taken. Now, thanks to the process we went through to give our table a _year_best_ column **at the end of the first notebook** (particularly the bit where we grouped observations by the year they were taken in), the table should be in nearly the right order.

However, seeing as it is critical that the observations be ordered correctly (if they aren't we might see some quite confusing things in our eventual GIF), we should make absolutely sure through the use of the `.sort(...)` function that is built-in to the `Table` class! We simply need to pass the name of the column we want to sort by - _time_.

The default behavior of the `.sort(...)` function is to sort in **ascending order**, but if for some reason you needed a table ordered _the other way_, you could pass an extra argument to the function - e.g. `obs_table.sort('time', reverse=True)`.

Now we run the sort and look at the first ten rows:

```{code-cell} python
# Sorting an astropy Table is an 'in place' operation, so we don't assign the
#  return to a variable (there is no return)
obs_table.sort("time")
obs_table[:10]
```

```{admonition} Extra Task
:class: seealso
You could quickly test what happens if you sort the table by a column other than _time_. **Don't forgot to re-sort by _time_ afterwards though, otherwise your GIF will not be showing the evolution of Cas A correctly!**
```

+++

## 3. Loading the Chandra FITS images into memory

+++

FITS is a very versatile format, used almost ubiquitously in the astronomy community (at least until the last couple of years - conversations about what we're going to replace it with have begun, and some missions are **already** using other file formats). It can be used to store anything from **images**, to spectra, to tables of catalog information, to **event lists** (restricted to high-energy missions, and one of the things that makes archival X-ray and Gamma-ray data so flexible and useful for serendipitous science).

As our goal for this notebook is to examine how Chandra's view of Cas A has changed over ~2 decades worth of observations, we will want to make nice image visualizations for us to look at. To do that we could use a piece of software running with a graphical user interface (GUI), [such as SAOImage DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9), which is probably the most commonly used FITS image viewer - **however**, we are in a Python environment, and if we load the data here we will not only be able to make visualizations of the images that look just as good as DS9's, but we can perform a lot of analysis that is simply not possible in GUI software (it is also **much easier to scale to multiple observations** when working in a coding environment than in a GUI tool).

+++

### 3.1 Opening a FITS file in Python using `astropy`

+++

The `astropy` module (think of it as the core library for astronomy/astrophysics in Python) contains a powerful submodule for interacting with FITS files - "_interacting_", in this case, covers everything from reading data from the various storage structures that FITS files can contain, to making completely new files, to modifying existing ones.

`astropy` also implements the functionality necessary to load FITS files remotely, from URLs and AWS-S3 URIs (or the Google/Microsoft equivalents), either downloading them automatically to local storage and reading them, or streaming them straight into memory. We won't really cover what that means in this workshop, but if any of you are interested to learn please ask one of the team - suffice it to say that it **can be** (though isn't always) a much more efficient way of loading data when working on a cloud platform like NASA Fornax.

Specifically, we will use the `astropy.io.fits(...)` function.

Before we get to loading in all the images we'll use to explore the X-ray evolution of Cassiopeia A, we will read a couple of files from the whole-observation-data-directories we downloaded as a demonstration of the 'easiest' way to acquire Chandra data in the first notebook. First, we will scan the directory we stored full-observations in, and dynamically choose the first ObsID in the list for our demonstration:

```{code-cell} python
demo_obs_id = [
    en.name for en in os.scandir(OBS_ROOT_DATA_DIR) if en.is_dir() and len(en.name) == 5
][0]
demo_obs_id
```

Set up a variable containing the full path to the chosen ObsID directory:

```{code-cell} python
demo_dir = os.path.join(OBS_ROOT_DATA_DIR, demo_obs_id)
```

### 3.2 Lets look at an **event list**

+++

Event lists are the foundation of observational high-energy astrophysics. They contain information like the time, position, and energy of *every* photon that makes it to the detector. Therefore, these datasets contain the most complete information and can be organized and simplifed to produce easy-to-use science data products like images, spectra and lightcurves.

We set up a full path the events list file, dealing with the dynamic nature of Chandra archive file names (discussed in the first notebook) by modifying the image file name we already stored in the observation table:

```{code-cell} python
demo_evt_name = obs_table[obs_table["obsid"] == demo_obs_id]["img_file_name"][
    0
].replace("cntr_img2", "evt2")
demo_evt_path = os.path.join(demo_dir, "primary", demo_evt_name)
```

The first thing to note is that we are opening the FITS file using a `with` statement - this is a **"context manager"**, which in this case is a Pythonic way of making sure that the 'file handler' set up by `fits.open(...)` to read data from the FITS file is closed when we reach the end of the code block indented in the `with`.

Leaving file handler connections open means that you can ask to read more things from the file - say, for instance, that you had loaded the header that describes a FITS image's properties, but not the data array itself, but then later in the code you wanted to read in that data - an open file handler would let you do that, if it had closed you would have to run the `fits.open(...)` call again.

**However**, there are significant caveats to leaving them open (which is why the context manager is considered best practise):
1) If you leave a lot of connections to different files **open** (and it is common to want to read from many 10s of files, sometimes significantly more, during an analysis) then that can negatively impact the performance of your file system (reading and writing files will get slower).
2) Operating systems impose hard limits on how many files can be 'open' at once (though the number is normally quite large, so unlikely to be a problem for a notebook like this).
3) Every connection will use up some of your RAM, which is a finite resource, particularly on smaller Fornax server instances.


Now we will _proceed to be quite inefficient and repeatedly read the same file_, so we can separately demonstrate different aspects of interacting with a FITS file in Python. The first thing to know is that FITS files are made up various _Header Data Units_ - there are different types of HDU that store images and tables, and a 'primary' HDU (an image HDU) which is always the first entry, and contains information about the contents of the rest of the FITS file. We can see below that our demonstration event list has three different `BinTableHDU`s (they store table data).

Additionally, each HDU is in a FITS file assigned a name, which usually informs you of the contents:

```{code-cell} python
with fits.open(demo_evt_path) as evto:
    # Add a newline to separate the two prints
    print(evto, "\n")

    # Iterate through the HDUs
    for cur_hdu in evto:
        print(cur_hdu.name)
```

The most interesting entry in this file is definitely the **events** table, so let's take a quick look at that. The two most important properties of an `astropy` HDU are generally `.header` and `.data` (though it is important to note that `PrimaryHDU`s _don't always_ have data associated, just a header). First, lets look at a small slice of this `BinTableHDU`'s header (sliced because otherwise this header is over 500 rows long).

Exactly what a header contains depends on the type of HDU it belongs to, and on the software that created it, and the mission it was created for (though there is a FITS _standard_ that the structure of each file, header, etc. must follow to be considered a valid FITS file). In our slice we can see the beginnings of table column definitions that define the names, data types, and minimum/maximum values of columns in this HDU's data.

At the bottom of the cell, we also demonstrate that the header keys (e.g. XTENSION, BITPIX, etc.) can be used to access their corresponding value by indexing with them:

```{code-cell} python
with fits.open(demo_evt_path) as evto:
    # Using pprint (pretty print) because it makes the headers easier to read
    pprint(evto["EVENTS"].header[:30])

    # Adding a blank space, for clarity.
    print("\n")

    print("The assigned to the TTYPE5 header key is:", evto["EVENTS"].header["TTYPE5"])
```

```{tip}
The built-in Python function `dir(<put Python object here>)` is a **very** useful way of learning your way around a new-to-you type of object (a `BinTableHDU` instance, for example), as it will list all of the functions, attributes, and properties of the object. Try it out!
```

+++

Now, onto the `.data` property. While the _contents_ of a header will change depending on the type of HDU it is a part of, the _form_ of the header will not. That is not true for the `.data` property, as different types of HDU represent fundamentally different kinds of data - an events list takes the form of a table, with named columns, potentially with different data types (strings, integers, floats, etc.); an image, as we'll see very soon, takes the form of an array, with no named columns or rows, and a uniform data type.

The return from the `.data` property of a `BinTableHDU` comes in the form of a `FITS_rec` object - this is what `astropy` uses to represent and interact with FITS tables (see the second `print` we do below, of a small subset of the events list table). We also include a `print` of the `.data` property of the `PrimaryHDU`, to show that in this case there is no data associated.

These `FITS_rec` objects aren't always nice to work with though, particularly if we want a nice-looking table to examine for our demonstration. We can solve that problem easily, however, by wrapping the return in an `astropy` `Table`, which is the type of object we've been using to interact with our table of observations both in this notebook and the first. You'll note as well, that when we assign the `.data` output to a _variable_ using `evto_table = ...`, we can access that data _outside_ of the `with` statement:

```{code-cell} python
with fits.open(demo_evt_path) as evto:
    print(evto["PRIMARY"].data, "\n")

    print(evto["EVENTS"].data[:3], "\n")

    evto_table = Table(evto["EVENTS"].data)

evto_table
```

Every entry in this table represents an _event_ recorded by the detector, each of which would represent a separate high-energy photon (in a perfect world). The reality is a little different, and some of these events will actually represent high energy particles like cosmic rays, multiple detections of the same photon/particle, or multiple photons combined into one detection. We have various techniques for cleaning event lists so that the vast majority of non-photon-events are removed however, and we are then left with a dataset that tells us:
1) The **arrival time** of each individual; the _time_ column.
2) **Where on the detector each photon landed**; _detx_ and _dety_ columns. For missions that _focus_ the light using X-ray optics (as opposed to just gathering light from a big chunk of the sky and lumping it together - non-imaging missions) we can map the detector positions of photons to locations on the sky; the _x_ and _y_ columns give event positions in a coordinate position that can be transformed to RA-Dec coordinates, as the _detx_ and _dety_ coordinate system has been rotated to align with the RA-Dec coordinate system orientation.
3) The **energy of the incident photon**; the _energy_ column in Chandra-ACIS event lists (though in other mission's event lists it is more common to only see the _pha_ or _pi_ columns, which are detector measurements of the 'strength' of each photon's detection signal that map to energy), and have to do the conversion yourself.

With all that information, you can see why X-ray and Gamma-ray observations are so powerful for archival work - by binning the event list in **2D** using the _x_ and _y_ columns, we can make an **image**! Then, by binning the events in **1D** using the _time_ and _energy_ columns, we can make **lightcurves** (which show how the brightness of a target changes with time, if it does at all) and **spectra** (the brightness of a target at different energies, which tells you a lot about the emission mechanisms in your target) separately. Additionally, using the event position information, we can filter the event list to only select events from a region around our target, to avoid including irrelevant emission in our lightcurves and spectra.

The energy and positional information can also be combined to make images only within specific energy bands (filters), **after the observation has been made**. Most lower-energy (e.g. infrared, optical) observations have to decide which filters to use **before** they make their observations.

Recording all this informaton about individual X-ray/Gamma-ray photons is possible only because they are _so high-energy_ that an individual incident photon can cause a detectable charge to accumulate in the detector, whereas lower-energy (longer-wavelength) observations have to "integrate" for some amount of time for charge to accumulate from multiple photon arrivals - they then can't measure the properties of individual photons, and lose the flexibility of an event list.

All the information we gain from observing our lovely X-ray/Gamma-ray photons does come at a cost though - _because_ they are so high-energy, it is actually very difficult to **focus** these photons, which means X-ray and Gamma-ray observations cannot attain the high level of spatial (also described as "angular") resolution possible for longer-wavelength telescopes. The optics in a focusing X-ray telescope look nothing like those of an optical telescope.

+++

```{admonition} Extra Task
:class: seealso
**As an extension to this notebook** - attempt to make and plot your own images, lightcurves, and spectra from this event list. You should an idea of how to extract columns of data from an `astropy` `Table` from both this and the first notebook of the workshop. The `numpy` module provides both **1D** ([`np.histogram(...)`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html)) and **2D** ([`np.histogram2d(...)`](https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html)) binning functions, and you can use `matplotlib` to plot the resulting data products.
```

+++

### 3.3 The anatomy of a FITS image

+++

You'll recall from our first notebook that we downloaded pre-generated images for all our selected Chandra observations - those are what we'll be using to examine the evolution of **Cas A**.

As such, we'll quickly demonstrate what it looks like when we use `astropy` to read a FITS _image_ file in, as opposed to the event list table file we loaded above.

First of all, we set up the path to the image file stored in the whole-observation directory we're using for these current demonstrations (we could just as easily have made it a path to the individual image files we downloaded for all our selected observations):

```{code-cell} python
demo_img_path = os.path.join(
    demo_dir,
    "primary",
    obs_table[obs_table["obsid"] == demo_obs_id]["img_file_name"][0],
)
```

Now we follow the exact same process as before to read the file in using `astropy` - we also `print` the same information as when we first demonstrated loading an event list. This shows that our image file contains **three** HDUs - the first, _primary_, HDU contains the image data array we want to use, but you'll also notice that **two tables** are stored in this file as well! That is actually fairly common, particularly for high-energy mission data files, as we sometimes wish to keep information about what events, data, or filtering operations were used to generate an image, alongside it, in the same file.

The two tables in this image file contain "good time interval" (GTI) information, which describes the time periods of the observation that the events used to make the image were extracted from. Sometimes we need to filter out large chunks of an observation because they were too noisy, or negatively effected in some way - we do that by defining these GTIs:

```{code-cell} python
with fits.open(demo_img_path) as imgo:
    # Add a newline to separate the two prints
    print(imgo, "\n")

    for cur_hdu in imgo:
        print(cur_hdu.name)
```

```{seealso}
The `fits.open(...)` function has many configurable options that can help to optimize its performance for reading in large images, remote files, etc. - wading in to all those options is outside the scope of this workshop, but see [the `astropy.io.fits` documentation](https://docs.astropy.org/en/stable/io/fits/index.html) for more information.
```

+++

#### 3.3.1 Data

+++

Looking at the `.data` property of the image file's `PrimaryHDU`, you'll note that we get the data returned as a standard numpy array. We've shown a 2D slice of values around the center of the image - each of those numbers is the 'count' of events within that spatial bin (or pixel of the image):

```{code-cell} python
with fits.open(demo_img_path) as imgo:
    demo_img_arr = imgo["PRIMARY"].data

    print(type(demo_img_arr), "\n")

    print(demo_img_arr.shape, "\n")

    print(demo_img_arr[550:650, 550:650])
```

#### 3.3.2 Header

+++

Looking at the image header, we once again show a 30-long slice of entries (as the whole header is much longer). If you compare this to the event list _table_ header, you'll likely notice some differences - for instance, the first entry of the events list table header was `XTENSION` (with the value set to "BINTABLE"), whereas that entry is missing entirely here (as an image HDU is considered the default).

Note that the NAXIS1 and NAXIS2 values (describing the dimensions of the data array stored in the HDU) match the shape of the array we retrieved in the last step.

Image headers also include information to help identify the telescope, instrument, ObsID, etc. without reference to an external source of knowledge (though many table headers also do include this information):

```{code-cell} python
with fits.open(demo_img_path) as imgo:
    pprint(imgo["PRIMARY"].header[:20])

    print("\n------------------------------------------------\n")

    print(f"This image's telescope is {imgo["PRIMARY"].header['TELESCOP']}")
    print(f"This image's instrument is {imgo["PRIMARY"].header['INSTRUME']}")
    print(f"This image's ObsID is {imgo["PRIMARY"].header['OBS_ID']}")
    print(f"This image's target is {imgo["PRIMARY"].header['OBJECT']}")
    print(f"This image's observer is {imgo["PRIMARY"].header['OBSERVER']}")
```

#### 3.3.3 World Coordinate System (WCS) - constructed from header

+++

Think back to the quick look we took at an events list, and you'll remember we highlighted two sets of column pairs that provided spatial information in different coordinate systems (_x_/_y_ and _detx_/_dety_), and mentioned a third coordinate system RA-Dec (generally the coordinate system we ultimately want to end up in, as it describes positions on the actual sky).

When it comes to images, we have at least one more to think about - the _image pixel_ coordinate system, which is likely a binned version of the _x_-_y_ system. With all these definitions of position, and the fact that when we analyse images we tend to define positions, masks, analysis regions, in _pixel_ coordinates, but will eventually need that information in RA-Dec so that it is in the context of the sky, rather than an image that other scientists won't have access to, we **clearly need to be able to convert between these coordinate systems**.

Describing the mapping between different coordinate systems is one of the primary purposes of a "World Coordinate System" (or WCS) - a concept used throughout astronomy, represented in specific ways in FITS files, following the FITS standard, and with a nice interface in `astropy`. WCSes can account for different **projections** - think how the spherical surface of the Earth is 'projected' into just two-dimensions for maps; astronomy has the same problem, but we're _inside_ the sphere, rather than outside - the rotation of spacecraft and instruments relative to the definition of _north_ in different sky coordinate systems, and a bunch of other effects.

We won't be diving deep into the fundamentals of WCSes, as they are far too complicated for this workshop, and even professional astronomers have trouble with them sometimes (including the author of this notebook). Instead, we will quickly demonstrate how the WCS information is stored inside a FITS header. Multiple WCSes can be (and often are) stored in high-energy mission image headers. Our Chandra images contain:
1. A WCS that can convert between the X-Y (what we saw in the event list, similar to detector coordinates but rotated to be aligned with RA-Dec's definition of north) and image pixels.
2. Another WCS that can convert between _RA-Dec_ and image pixels.

Headers defining both of those WCSes are printed separately below. Focusing particularly on the second block of headers (for the RA-Dec <-> image pixel WCS), we can see (for each axis of the coordinate systems):
- **CTYPE\<N\>** - In this case describes the type of projection (mapping) between RA-Dec and the pixel coordinates (which are cartesian) for one of the axes (this is why there are two sets of each of these headers, one for image x pixels and one for image y pixels).
- **CRVAL\<N\>** - Gives us a 'reference' value for the RA or Dec axis of the coordinate system.
- **CRPIX\<N\>** - Tells us the image pixel coordinate that corresponds to the RA-Dec reference value given in the last header.
- **CRDELT\<N\>** - Defines the change in RA or Dec (in degrees) that comes from moving one pixel in the x or y image pixel directions.

```{code-cell} python
with fits.open(demo_img_path) as imgo:
    xy_wcs_start_ind = imgo["PRIMARY"].header.index("MTYPE1")
    xy_wcs_stop_ind = imgo["PRIMARY"].header.index("MTYPE2")

    pprint(imgo["PRIMARY"].header[xy_wcs_start_ind:xy_wcs_stop_ind])

    print("\n------------------------------------------------\n")

    radec_wcs_stop_ind = imgo["PRIMARY"].header.index("CUNIT2") + 1

    pprint(imgo["PRIMARY"].header[xy_wcs_stop_ind:radec_wcs_stop_ind])

    # Assign the header to a variable so we use it in the next demonstration step
    demo_img_hdr = imgo["PRIMARY"].header
```

You may well be happy to hear that, with `astropy`, we don't have to deal with setting up WCSes directly (for images at least). Instead we can use the `WCS` class - feeding it an image header will let it pick out the relevant entries and automatically construct a WCS (**by default, the WCS that maps between RA-Dec and image pixels**):

```{code-cell} python
demo_wcs = WCS(demo_img_hdr)
print(demo_wcs)
```

We can use that `WCS` object with the `astropy` `SkyCoord` class we've already touched upon (the `SRC_COORD` constant we set up at the beginning of a notebook is a `SkyCoord` instance) to convert between image pixel coordinates and RA-Dec (and vice versa):

```{code-cell} python
print(SRC_COORD.to_pixel(demo_wcs), "\n")

print(SkyCoord.from_pixel(600, 600, wcs=demo_wcs))
```

### 3.4 Extracting relevant information from our Chandra image files

+++

Now we've discussed the contents of a FITS file, and how to access them through `astropy`, we will load images for all of the entries in the our table of selected observations (`obs_table`). We've already discussed that we don't want to leave file handler connections open any longer than we have to, so we make sure to store all the information we need from the image headers and data arrays in a Python dictionary `all_loaded_img_info` for easy access later in the notebook:

```{code-cell} python
all_loaded_img_info = {}

with tqdm(desc="Loading Chandra images into memory", total=len(obs_table)) as onwards:

    for cur_obs_row in obs_table:
        cur_obs_id = cur_obs_row["obsid"]

        cur_img_file_name = cur_obs_row["img_file_name"]
        cur_img_full_path = os.path.join(
            IMS_ROOT_DATA_DIR, cur_obs_id, cur_img_file_name
        )

        with fits.open(cur_img_full_path) as cur_imgo:

            cur_wcs = WCS(cur_imgo[0].header)

            cur_exposure = cur_imgo[0].header["EXPOSURE"]

            cur_obs_date = cur_imgo[0].header["DATE-OBS"]

            all_loaded_img_info[cur_obs_id] = {
                "data": cur_imgo[0].data,
                "wcs": cur_wcs,
                "exp": cur_exposure,
                "obs_date": cur_obs_date,
                "count_rate": False,
            }

        onwards.update(1)
```

```{important}
We've already touched on the ways in which leaving a file handler connection open can be detrimental to performance, but you must also understand that reading and writing files ('I/O') is **almost always an 'expensive' operation**. That means that we generally want to read files as few times as possible - so if we have enough RAM to keep the images in memory (which we do, in this case) then we should!
```

+++

## 4. Visualizing the images side-by-side

+++

We've now loaded the image data for all of our selected observations - we want to start looking at it for signs of Cas A's evolution! Eventually we're going to visualize that evolution with an animation showing the changes over the ~2-decades worth of Chandra observations, but before that we just want to look at stationary images. In this section we will demonstrate how you display nice-looking X-ray images using `matplotlib` (the very commonly used Python plotting module) and some `astropy` features, as well as introduce you to the idea of 'stretching' or 'scaling' your data using mathematical functions to visually highlight certain features.

+++

### 4.1 Visualizing image arrays using matplotlib

+++

Using `matplotlib` (or more specifically `pyplot`, usually imported as `from matplotlib import pyplot as plt`), makes it very easy to visualize images stored as `numpy` arrays - we can just set up a figure, use the `plt.imshow(...)` function, add a colorbar with the `plt.colorbar(...)` function, and show the figure.

When calling `plt.imshow(...)` we set `origin='lower'` so that our image is shown the right way up - this argument places the [0, 0] index of the `numpy` array in the _bottom left_ corner of the displayed image, which is how arrays from FITS images are meant to be displayed.

choose a color map called 'gnuplot2' (see [this `matplotlib` documentation](https://matplotlib.org/stable/users/explain/colors/colormaps.html#overview) for an explanation, and full list, of built-in color maps):

```{code-cell} python
plt.figure(figsize=(6, 4.5))
plt.axis("off")

plt.imshow(demo_img_arr, origin="lower", cmap="gnuplot2")

demo_cbar = plt.colorbar()
demo_cbar.set_label("Count", fontsize=15)

plt.tight_layout()
plt.show()
```

You'll probably find that its quite hard to pick out many features looking at the image above, there is very little contrast. That is because, though most of the pixels have relatively low values (the mean pixel value, excluding zero-values, of the demonstration array we're using is ~15), there are a few pixels with much higher values. Those pixels with high values are being assigned the high-contrast colors (pinks, oranges, and yellows in this colormap), and all the pixels that show the interesting detail and spatial structure of the supernova remnant are consigned to the low-contrast blues.

This is a common issue when examining X-ray observations of extended sources, and actually has its root in the physics of some X-ray emission mechanisms (we won't go into that in this workshop). The solution is to change the scaling, or stretch, of the mapping between pixel values and colors. The easiest way to stretch the image in Python is to use a few features of the `astropy.visualization` submodule ([see this documentation for a more detailed explanation of what `astropy` can do with respect to image stretching](https://docs.astropy.org/en/stable/visualization/normalization.html)) - alternatively, you could apply the `numpy` function `np.log(...)` to the data array before passing it to `plt.imshow(...)`.

Here we define an `astropy` `ImageNormalize` object (which was designed to be compatible with `plt.imshow(...)`'s `norm=` argument) - once set up it will take an input value and normalizes it so that it has a value between **0** and **1**. Exactly how it does that is defined by how it is configured; i.e. how the lower and upper _input values_ that map to **0** and **1** are defined, and whether the relationship between input values and **0-1** output values is linear. We configure it by passing the following:
- `data=demo_img_arr` - The image data array (used by the interval).
- `interval=MinMaxInterval()` - Sets up and passes a `MinMaxInterval()` interval object, which sets the _input_ data interval between which values will be normalized to between **0** and **1**. There are different varieties of interval class (and one can specify the values manually), but this one just looks at the input given to `data=` and finds its minimum and maximum values.
- `stretch=LogStretch()` - Tells the normalizer that the relationship between _input_ values within the **interval**, and output values between **0** and **1** should _not_ be linear. Instead it should be logarithmic.

We show the normalized results of two values as a demonstration (**if no stretch were applied the second printed value would be twice as large as the first**):

```{code-cell} python
demo_norm = ImageNormalize(
    data=demo_img_arr, interval=MinMaxInterval(), stretch=LogStretch()
)

print(demo_norm(300), "\n")
print(demo_norm(600))
```

Now we use much the same plotting commands as before, but this time include the `norm=demo_norm` argument in the call to `plt.imshow()`. Clearly, the resulting image looks quite different, and it has become much easier to see the various X-ray emission features of Cas A. If you look at the colorbar on this plot, and compare it to the one above, you'll see the difference that our log stretch has made to the mapping between color and pixel value:

```{code-cell} python
plt.figure(figsize=(6, 4.5))
plt.axis("off")

plt.imshow(demo_img_arr, origin="lower", norm=demo_norm, cmap="gnuplot2")

demo_cbar = plt.colorbar()
demo_cbar.set_label("Count", fontsize=15)

plt.tight_layout()
plt.show()
```

```{admonition} Extra Task
:class: seealso
The `astropy.visualization` submodule includes several other stretch classes ([this documentation](https://docs.astropy.org/en/stable/visualization/ref_api.html#classes) shows a list of them, amongst other classes) that could be imported and used in place of `LogStretch` - this includes a `LinearStretch` class that will apply no stretch at all. **As an extension to this notebook**, import one of the other stretches (e.g. `SqrtStretch`) and use it to normalize an image - how does it look different, and why?
```

+++

### 4.2 How do we center and crop the images?

+++

When dealing with multiple observations of an object, we cannot guarantee that they will all be perfectly aligned - in fact, it is incredibly unlikely that they will be. As such, we shouldn't just plot all the images as they are, because any evidence of Cas A's evolution would likely be obscured by the misalignment that comes from the telescope being oriented or pointed in slightly different directions.

To solve that problem, we're going to choose a single coordinate to center every image on, as well as setting up to cut-out only a section of each image - that way we can zoom-in on the more interesting parts of the observations, if we want. To demonstrate, we'll set the central coordinate of our cutouts to `SRC_COORD` and specify that the side-lengths of the cutout should be _200 pixels_ (**note that we'll also be able to specify the side-length in _angular units_ - degrees, arcminutes, etc.**). We choose the ObsID of the last entry in the `obs_table` to demonstrate with:

```{code-cell} python
demo_central_coord = SRC_COORD
demo_img_side_length = 200

cut_demo_obs_id = obs_table[-1][
    "obsid"
]  # Use the last (index = [-1]) observation from our Observations table as reference
```

We use yet another `astropy` function to easily create our cutouts - `Cutout2D`. It requires both an image data array and the corresponding WCS (which we discussed in an earlier section of the notebook), so it can automatically convert between image pixel and RA-Dec coordinate systems. Some other arguments are also passed to fine tune its behavior:
- `mode='partial'` - This tells the function how to deal with a requested cutout that goes off the edge of actual size of the full image. The 'partial' option means that if _some_ of the requested cutout goes off the edge of the image, `Cutout2D` will make an array of the requested size, and fill any part of it that is off of the original image with the specified `fill_value`.
- `fill_value=0` -  This tells the function to use zero as the fill value mentioned above.

```{code-cell} python
demo_img_data = all_loaded_img_info[demo_obs_id]["data"]
demo_img_wcs = all_loaded_img_info[demo_obs_id]["wcs"]

# Create the demonstrative cutout
demo_cutout = Cutout2D(
    demo_img_data,
    position=demo_central_coord,
    size=demo_img_side_length,
    wcs=demo_img_wcs,
    mode="partial",
    fill_value=0,
)
```

The cutout's data array can be accessed using the `.data` property:

```{code-cell} python
demo_cutout.data
```

```{admonition} Extra Task
:class: seealso
**As an extension to this notebook**, you could take what you learned about visualizing X-ray images earlier in the notebook, and try to make a nice figure out of our demonstration cutout.
```

+++

### 4.3 How do we normalize the images by their exposure time?

+++

These images have different exposures so to show an apples-to-apples comparison we want to have the images scaled to similar flux which is proportional to count rate. Since the pure images are in counts we need to divide the image values by the exposure times.

+++

Just as observations of an object aren't always going to be pointed exactly the same, they are very rarely going to have the exact same exposure time. As such, we can't just compare our _images_ to one another to see how Cas A evolves, as an image represents the number of 'counts' (events) per pixel - if you observed something for half an hour, then the same thing for two days, you would expect the latter observation to have many more counts per pixel (unless of course the _emission from the object is changing_).

So, before any analysis, we tend to turn our _images_ into _count-rate maps_ - it is possible to relate a count-rate to the number of photons emitted by the object per second, if you understand the characteristics of your telescope (and other things), and that gives you a window into the astrophysical processes that are _causing_ the emission of X-ray photons.

The "proper" way to the turn an image into a count-rate array (or rate-map) is to divide by an _exposure map_ (please ask one of the presenters if you want to learn about exposure maps), but we'll make a reasonable approximation here by dividing the image by the exposure time stored in its FITS header.

To avoid setting up a new array per observation, and thus using up more memory than we have to, we're going to modify the existing data arrays we stored in `all_loaded_img_info`, and to make sure that we don't divide by exposure again if we accidentally re-run this cell, we put a boolean variable in `all_loaded_img_info` that is False initially, and set to True when the modification is made:

```{code-cell} python
for cur_obs_id in all_loaded_img_info:
    if not all_loaded_img_info[cur_obs_id]["count_rate"]:
        all_loaded_img_info[cur_obs_id]["data"] = (
            all_loaded_img_info[cur_obs_id]["data"]
            / all_loaded_img_info[cur_obs_id]["exp"]
        )
        all_loaded_img_info[cur_obs_id]["count_rate"] = True
```

```{code-cell} python
all_loaded_img_info["00114"]["data"]
```

### 4.4 Setting up for the side-by-side comparison figure

+++

We're nearly ready to produce our figure that will compare images from all the observations we've selected - we just need to set up some variables to configure it. First of all, how large each image's panel will be (`fig_side_size`), how many columns the comparison figure will have (`fig_num_cols`), the colormap to use (`fig_color_map`), and where the _saved to disk_ version will be stored:

```{code-cell} python
fig_side_size = 4
fig_num_cols = 3

fig_color_map = "gnuplot2"

fig_out_path = os.path.join(
    ROOT_OUT_PATH,
    f"{SRC_NAME.replace(" ", "")}-side-by-side-chandra-image-comparison.png",
)
```

These options control the cutout that will be extracted:

```{code-cell} python
fig_central_coord = SRC_COORD
fig_cutout_side_length = Quantity(7, "arcmin")
```

You may recall that when we first discussed image normalization, earlier in the notebook, we used a `MinMaxInterval()` to define the lower and upper limits on the values that the normalizer should map to the **0** to **1** range.

Setting up the `ImageNormalize` instance for our comparison figure in the same way would be a mistake, as each individual image is going to have different minimum and maximum values, so the exact same color in each image would **not** represent the same number of counts! That would make a direct comparison much harder, so we'll set an interval (using the `ManualInterval` class), based on all of the images, that will be _applied to all of them uniformly_.

We're going to set the minimum value to zero. The maximum value _could_ just be set to the highest pixel value found in any of the images, but instead we could take this opportunity to place the upper limit below the true maximum - that then de-emphasizes the brightest pixels (which are usually from bright, point like sources), and allows the slightly-less-bright _structure_ of the supernova remnant to be shown in higher contrast colors - making them more visually obvious.

Anyway, we do still find the maximum value of each count-rate array, but we then calculate the 95th percentile value to use as the upper limit:

```{code-cell} python
max_rate_vals = [
    cur_im_info["data"].max() for cur_im_info in all_loaded_img_info.values()
]

img_upper_lim = np.percentile(max_rate_vals, 95)

fig_img_interval = ManualInterval(vmin=0, vmax=img_upper_lim)
```

Once again we want to 'stretch' this image logarithmically:

```{code-cell} python
fig_img_stretch = LogStretch()
```

We combine the _interval_ and _stretch_ objects we just created in an `ImageNormalize` object. Note that this time we don't have to pass any data argument, as we have already set the lower and upper value limits of the interval - we can apply this normalizer to any of our count-rate maps.

```{code-cell} python
fig_img_norm = ImageNormalize(interval=fig_img_interval, stretch=fig_img_stretch)
```

### 4.5 Producing the side-by-side comparison figure

+++

Finally, we use `matplotlib` to produce a comparison figure, showing all of observations side-by-side. As `matplotlib` is not the main focus of this workshop, we won't go into details right now, but if you do have questions/would _like_ the details, please ask one of the presenters.

One thing to note though, is that we are producing cutouts as we go - meaning we're not storing them in any variables for later use. This means that we don't use up memory on cutouts that we might not want to use again after this step:

```{code-cell} python
num_ims = len(all_loaded_img_info)
fig_num_rows = int(np.ceil(num_ims / fig_num_cols))

fig, ax_arr = plt.subplots(
    ncols=fig_num_cols,
    nrows=fig_num_rows,
    figsize=(fig_side_size * fig_num_cols, fig_side_size * fig_num_rows),
)
plt.subplots_adjust(wspace=0.02, hspace=0.02)

ax_ind = 0
all_im_arrs = []
all_date_strs = []
for ax_arr_ind, ax in np.ndenumerate(ax_arr):
    if ax_ind >= num_ims:
        ax.set_visible(False)
        continue

    ax.axis("off")

    cur_obs_id = list(all_loaded_img_info.keys())[ax_ind]

    cur_cutout_data_arr = Cutout2D(
        all_loaded_img_info[cur_obs_id]["data"],
        fig_central_coord,
        fig_cutout_side_length,
        all_loaded_img_info[cur_obs_id]["wcs"],
        mode="partial",
        fill_value=0,
    ).data

    ax.imshow(
        cur_cutout_data_arr, origin="lower", norm=fig_img_norm, cmap=fig_color_map
    )

    ax.text(
        0.05,
        0.92,
        all_loaded_img_info[cur_obs_id]["obs_date"],
        transform=ax.transAxes,
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        0.7,
        0.12,
        cur_obs_id,
        transform=ax.transAxes,
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    ax_ind += 1

# Format and save BEFORE showing (doesn't work the other way around)
plt.savefig(fig_out_path, bbox_inches="tight")
plt.show()
```

## 5. Considering our data

+++

It is **always** a good idea to visually inspect your data before using it for science! It allows us to do basic validity checks - looking for any obvious problems with the data, either in the form of errors or unsuitability for the task we want to do.

Thanks to our large side-by-side plot we now have a view of all our observations - we can take this opportunity to consider if we want to utilize all of it.

+++

### 5.1 Does the side-by-side plot show any observations we should exclude?

+++

It is hard to see the evolution of Cas A's X-ray emission from looking at our comparison figure - it isn't made easier by the fact that some of these observations were clearly taken in a different mode to the others, as they show a much thinner strip of Cas A.

While those observations are certainly scientifically useful, they aren't going to help us with our current goal, which is making an animation of our supernova remnant's evolution. Going forward, we should exclude those observations:

```{code-cell} python
to_exclude_obs_ids = [
    "06690",
    "13783",
    "16946",
    "17639",
    "22426",
    "23248",
    "26655",
    "27099",
    "27100",
]
```

Using the `np.isin(...)` function, which produces a boolean array the same length as the `obs_table['obsid']` input by checking every element to see if it is contained in the _second_ input, `to_exclude_obs_ids`, and the Python invert operator `~`, we make a masking array that selects only those rows of the `obs_table` whose ObsID's **aren't** in the `to_exclude_obs_ids` list:

```{code-cell} python
filt_obs_table = obs_table[~np.isin(obs_table["obsid"], to_exclude_obs_ids)]
```

```{admonition} Extra Task
:class: seealso
While looking through ~30 images and recording the ObsIDs you want to exclude is perfectly feasible, what if you had to look at _1000_? **Try to come up with a way of using the information in the cutout image data arrays to find the 'thin strip' images automatically.**
```

+++

### 5.2 Flagging the 'best' (longest exposure) observation made each year

+++

If you think back to the histogram we made in the first notebook, that showed us the distribution of how many Chandra observations were made of Cas A each year, you will recall that, for some years, we selected more than one observation. While in many cases, more observations are better, when we make a movie of the evolution of Cas A's X-ray emission, we might want to restrict ourselves to just **one** observation per year.

A simple proxy for observation quality is the **exposure time** - how long Chandra was looking at the target during a particular observation. Longer exposure times result in what we call 'deeper' data, meaning that we collect more photons and become sensitive to lower and lower flux emission. X-ray astronomy is a photon-starved field, we never really seem to have enough, so _generally speaking_ longer exposure times are better.

**Of course, there is more nuance involved than longest exposure == best observation, but it is a safe enough assumption to start with**.

We're going to modify the `filt_obs_table` table so that it has a new boolean column called _year\_best_ - that column value will be set to `True` for observations that had the longest exposure time of the year. The first step is to add the new column, setting every entry to `False`:

```{code-cell} python
filt_obs_table["year_best"] = False
```

Next, we have to be able to tell which year each observation was taken in - once again we use the `Time` class from `astropy`, passing the whole _time_ column and specifying that the times are currently in the MJD format. We can then use the `.decimalyear` function included in the `Time` class to retrieve the year, which we then convert to an integer, and assign to **another** new column in `selected_obs`:

```{code-cell} python
filt_obs_table["year"] = Time(filt_obs_table["time"], format="mjd").decimalyear.astype(
    int
)
```

At this point we need to find the observation with the maximum value in the _exposure_ column for each separate year represented in the _year_ column. If we wanted to just find the maximum exposure time of **all** the observations, that would be very straightforward.

Running:

```python
filt_obs_table['exposure'].max()
```
would return the maximum value of the exposure column, and running:

```python
filt_obs_table['exposure'].argmax()
```

would return the index of the row with the highest exposure value (try these commands out if you like).

So, we need some way of splitting up the table of selected exposures into a bunch of sub-tables, one per year. Thankfully, we can quite easily do that by "grouping" the table by the values of the _year_ column, using the `.group_by(...)` function of the `Table` class - the return is yet another table, which looks very similar to `filt_obs_table` on the surface (other than that the order of the rows has changed), but now contains new information that defines groups of rows. At the end of the cell we display a representation of the `groups` property of our new table, which prints the start and end index of each group of rows in the `filt_obs_table` table:

```{code-cell} python
filt_obs_table = filt_obs_table.group_by("year")
filt_obs_table.groups
```

That `groups` property of our new grouped table - `filt_obs_table` - has a set of useful functions built-in, designed to make it easy to analyze and interact with the grouped rows.

Here we will use the `aggregate(...)` function, which will apply a passed function (in this case `np.argmax`) to each of the groups individually - when we aggregate-run `argmax` on the exposure column of each of our groups (again, think of them as sub-tables) the return will be the index **within each group/sub-table** of the highest-exposure observation.

If only one observation was taken in the year 2002, for instance, then the group for that year will contain **one** row, so the index of the maximum exposure observation within that group will be **0**. That _doesn't_ tell us where that particular row is in the overall `filt_obs_table` table, however, which we need to know because we want to modify it's _year\_best_ entry to be `True`!

Thankfully, as we saw in the output of the cell above, we know the indexes at which each group starts in the overall `filt_obs_table` table, so we can just add those to the indexes returned from the `aggregate(...)` function:

```{code-cell} python
year_best_obs_inds = (
    filt_obs_table["exposure"].groups.aggregate(np.argmax)
    + filt_obs_table.groups.indices[:-1]
)
year_best_obs_inds
```

Then we use those values to index the relevant rows of the `filt_obs_table` table and set the _year\_best_ column values to `True`:

```{code-cell} python
filt_obs_table["year_best"][year_best_obs_inds] = True
```

Now, if we quickly take a look at the first few entries of the subset of the `year_grouped_obs` table that has _year\_best_ set to True, we should see one entry per year:

```{code-cell} python
filt_obs_table[filt_obs_table["year_best"] is True][:3]
```

## 6. Examining the evolution of Cas A's X-ray emission as a movie

+++

### 6.1 Setting up for the evolution GIF

+++

First, we define the file path where we'll save the completed GIF of Cassiopeia A's X-ray evolution:

```{code-cell} python
gif_out_path = os.path.join(
    ROOT_OUT_PATH, f"{SRC_NAME.replace(" ", "")}-chandra-evolution.gif"
)
```

Now, the GIF's 'frames per second' (FPS) - i.e. the number of observations that are shown per second. Increasing this will make the observations go by faster and the GIF shorter, and vice versa:

```{code-cell} python
gif_fps = 6
```

Finally, we define the radius (in pixels) of a circle that will be overlaid on top of the GIF to help give context to the expansion of Cas A:

```{code-cell} python
illustrative_circle_pix_radius = 270
```

Extracting the ObsIDs to include in the GIF - we're going to limit ourselves to using the best one from each year, but you could use _all_ of the filtered ObsIDs if you wanted (`gif_obs_ids = filt_obs_table['obsid']`):

```{code-cell} python
gif_obs_ids = filt_obs_table[filt_obs_table["year_best"]]["obsid"]
```

As we've chosen a subset of the available observations to use to create a 'data product' (in this case an animation of the expansion of Cas A's X-ray emission), we may want to keep a record of exactly which data went into that product. We could add another boolean column to the "CasA-selected-chandra-observations.csv" file we created in the first notebook (and loaded in earlier in this one), but sometimes its just easier to make a new file with only those relevant entries in.

This is a mildly inefficient use of storage space, seeing as it is purely a subset of an existing sample file, but the file sizes here are so small that, practically speaking, it doesn't matter:

```{code-cell} python
filt_obs_table[filt_obs_table["year_best"]].write(
    obs_table_path.replace("selected", "filtered-selected"),
    format="csv",
    overwrite=True,
)
```

### 6.2 Setting up the frame update function

+++

We're using `matplotlib` to create our animation, and for that to work we need to define a function that will update the visualization for each 'frame' (i.e. observation). The function below will generate the new cutout for the ObsID, based on the input coordinate and side-length information, and then update the `img_frame` (which is the return of a `plt.imshow(...)` call you'll see in the next cell) with the new array.

It also updates the text of the `time_annotation` and `oi_annotation` objects, which are both overlays we add to the GIF - one to show the observation time and one to show the ObsID. Finally, you'll notice that an object called `circle` is returned - that is defined in the code in the next cell and doesn't need to be updated, but overplots a circle to give some visual context to Cas A's expansion:

```{code-cell} python
# Define the update function for the animation loop


def update_frame(frame_obs_id):
    cur_cutout_data_arr = Cutout2D(
        all_loaded_img_info[frame_obs_id]["data"],
        fig_central_coord,
        fig_cutout_side_length,
        all_loaded_img_info[frame_obs_id]["wcs"],
        mode="partial",
        fill_value=0,
    ).data

    # Update the data array of the existing plot object
    img_frame.set_array(cur_cutout_data_arr)

    # Update the text string for the current frame
    time_annotation.set_text(all_loaded_img_info[frame_obs_id]["obs_date"])

    oi_annotation.set_text(frame_obs_id)

    # Return both objects in a list so this updates both layers
    return [img_frame, time_annotation, oi_annotation, circle]
```

### 6.3 Creating and watching our evolution animation!

+++

Finally, we set up a count-rate map visualization (essentially as we've done in the other parts of this notebook we've looked at images), and overlay observation date, ObsID, and a circle. The `animation.FuncAnimation(...)` imported from `matplotlib` then takes a list of our chosen ObsIDs, and iterates through them, passing each to our `update_frame` in turn.

Note that in order to make the GIF pause slightly at the first and last observations, we duplicate the initial and final frame several times.

At the end of the cell, after the GIF has been generated and saved to disk, we use `IPython`'s `Image(...)` object to display the GIF inside the notebook:

```{code-cell} python
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
ax.axis("off")

# We sub-sample the chosen colormap to just 96 colors, to work within the encoding
#  limitations of .gif files
chosen_cmap = plt.get_cmap(fig_color_map, 96)

# Set up the initial frame frame
# We use the first image to establish the display object
first_img_obs_id = gif_obs_ids[0]
first_img_cutout = Cutout2D(
    all_loaded_img_info[first_img_obs_id]["data"],
    fig_central_coord,
    fig_cutout_side_length,
    all_loaded_img_info[first_img_obs_id]["wcs"],
    mode="partial",
    fill_value=0,
).data


img_frame = ax.imshow(
    first_img_cutout,
    animated=True,
    norm=fig_img_norm,
    origin="lower",
    cmap=chosen_cmap,
    interpolation="nearest",
)

cbar = fig.colorbar(ScalarMappable(norm=fig_img_norm, cmap=chosen_cmap), ax=ax)
cbar.set_label(r"Count-rate [ct s$^{-1}$]", fontsize=15, rotation=270, labelpad=20)

# CREATE AND ADD THE CIRCLE PATCH LAYER HERE
# Center is calculated based on half your crop size
circle_center = (first_img_cutout.shape[1] / 2, first_img_cutout.shape[0] / 2)

circle = Circle(
    circle_center,
    radius=illustrative_circle_pix_radius,
    edgecolor="white",
    facecolor="none",
    linewidth=1.5,
    linestyle="dashed",
    animated=True,  # Important: Set to True because we are using blitting
)
ax.add_patch(circle)  # Bake the circle into the axis layout

# --- ADD THE INITIAL ANNOTATION LAYER HERE ---
# transform=ax.transAxes places text relative to the
#  box (0,0 is bottom-left, 1,1 is top-right)
# This prevents the text placement from depending on your data pixel coordinates.

time_annotation = ax.text(
    0.05,
    0.92,  # X, Y position (Top-left corner of the frame)
    all_loaded_img_info[first_img_obs_id]["obs_date"],  # Starting text value
    color="white",  # High contrast color
    fontsize=18,
    fontweight="bold",
    animated=True,
    transform=ax.transAxes,
)

oi_annotation = plt.text(
    0.8,
    0.05,
    first_img_obs_id,
    color="white",
    fontsize=18,
    fontweight="bold",
    animated=True,
    transform=ax.transAxes,
)

frame_ois = ([gif_obs_ids[0]] * 5) + list(gif_obs_ids[1:]) + ([gif_obs_ids[-1]] * 5)

# Construct the animation object. The interval is the time between
#  frames (in milliseconds (1000 / fps))
ani = animation.FuncAnimation(
    fig=fig,
    func=update_frame,
    frames=frame_ois,
    interval=int(1000 / gif_fps),
    blit=True,
)

writer = animation.PillowWriter(
    fps=gif_fps, metadata=dict(artist="HEASARC"), bitrate=1000
)
ani.save(gif_out_path, writer=writer)

plt.close(fig)

# Show the GIF
Image(filename=gif_out_path)
```

```{admonition} Extra Task
:class: seealso
A circle is overlaid to give context to the expansion - **try adding a different shape, or altering the circle's position and radius, to convince yourself of the movement of a specific feature.**
```

+++

## 7. How would you make the changes compared to the first observation more visually obvious?

+++

```{admonition} Extra Task
:class: seealso
The evolution of Cas A's X-ray emission is considerably more visually obvious in our GIF than in our original side-by-side comparison, but can you make it even more obvious? **Make a new _difference imaging_ GIF to better highlight which parts of Cas A change, and which stay more or less the same.**
```

```{code-cell} python
ratio_gif_out_path = os.path.join(
    ROOT_OUT_PATH, f"{SRC_NAME.replace(" ", "")}-chandra-ratio-evolution.gif"
)
ratio_gif_out_path
```

```{code-cell} python
all_rat_ims = {}

first_img_obs_id = gif_obs_ids[0]
first_img_cutout = Cutout2D(
    all_loaded_img_info[first_img_obs_id]["data"],
    fig_central_coord,
    fig_cutout_side_length,
    all_loaded_img_info[first_img_obs_id]["wcs"],
    mode="partial",
    fill_value=0,
).data

for cur_obs_id in gif_obs_ids:
    cur_img_cutout = Cutout2D(
        all_loaded_img_info[cur_obs_id]["data"],
        fig_central_coord,
        fig_cutout_side_length,
        all_loaded_img_info[cur_obs_id]["wcs"],
        mode="partial",
        fill_value=0,
    ).data
    all_rat_ims[cur_obs_id] = np.divide(
        cur_img_cutout,
        first_img_cutout,
        where=first_img_cutout != 0,
        out=np.zeros(first_img_cutout.shape),
    )
```

```{code-cell} python
max_ratio_vals = Quantity(
    [cur_rat_im_arr.max() for cur_rat_im_arr in all_rat_ims.values()]
)

ratio_interval = ManualInterval(vmin=0, vmax=np.percentile(max_ratio_vals, 95))
ratio_norm = ImageNormalize(interval=ratio_interval, stretch=LogStretch())
```

```{code-cell} python
# Define the update function for the animation loop


def update_ratio_frame(frame_obs_id):
    cur_cutout_data_arr = all_rat_ims[frame_obs_id]

    # Update the data array of the existing plot object
    img_frame.set_array(cur_cutout_data_arr)

    # Update the text string for the current frame
    time_annotation.set_text(all_loaded_img_info[frame_obs_id]["obs_date"])

    oi_annotation.set_text(frame_obs_id)

    # Return both objects in a list so this updates both layers
    return [img_frame, time_annotation, oi_annotation, circle]
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
ax.axis("off")

chosen_cmap = plt.get_cmap("gnuplot2", 96)

# Set up the initial frame frame
# We use the first image to establish the display object
first_img_obs_id = gif_obs_ids[0]

img_frame = ax.imshow(
    all_rat_ims[first_img_obs_id],
    animated=True,
    norm=ratio_norm,
    origin="lower",
    cmap=chosen_cmap,
    interpolation="nearest",
)

cbar = fig.colorbar(ScalarMappable(norm=ratio_norm, cmap=chosen_cmap), ax=ax)
cbar.set_label(r"Ratio", fontsize=15, rotation=270, labelpad=20)

# CREATE AND ADD THE CIRCLE PATCH LAYER HERE
# Center is calculated based on half your crop size
circle_center = (first_img_cutout.shape[1] / 2, first_img_cutout.shape[0] / 2)

circle = Circle(
    circle_center,
    radius=illustrative_circle_pix_radius,
    edgecolor="white",
    facecolor="none",
    linewidth=1.5,
    linestyle="dashed",
    animated=True,  # Important: Set to True because we are using blitting
)
ax.add_patch(circle)  # Bake the circle into the axis layout

# --- ADD THE INITIAL ANNOTATION LAYER HERE ---
# transform=ax.transAxes places text relative to the
#  box (0,0 is bottom-left, 1,1 is top-right)
# This prevents the text placement from depending on your data pixel coordinates.

time_annotation = ax.text(
    0.05,
    0.92,  # X, Y position (Top-left corner of the frame)
    all_loaded_img_info[first_img_obs_id]["obs_date"],  # Starting text value
    color="white",  # High contrast color
    fontsize=18,
    fontweight="bold",
    animated=True,
    transform=ax.transAxes,
)

oi_annotation = plt.text(
    0.8,
    0.05,
    first_img_obs_id,
    color="white",
    fontsize=18,
    fontweight="bold",
    animated=True,
    transform=ax.transAxes,
)

frame_ois = ([gif_obs_ids[0]] * 5) + list(gif_obs_ids[1:]) + ([gif_obs_ids[-1]] * 5)

# Construct the animation object. The interval is the time between
#  frames (in milliseconds (1000 / fps))
ani = animation.FuncAnimation(
    fig=fig,
    func=update_ratio_frame,
    frames=frame_ois,
    interval=int(1000 / gif_fps),
    blit=True,
)

writer = animation.PillowWriter(
    fps=gif_fps, metadata=dict(artist="HEASARC"), bitrate=1000
)
ani.save(ratio_gif_out_path, writer=writer)

plt.close(fig)

# Show the GIF
Image(filename=ratio_gif_out_path)
```

```{code-cell} python

```
