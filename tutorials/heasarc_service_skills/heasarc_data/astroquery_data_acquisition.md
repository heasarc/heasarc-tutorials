---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-07-16'
execution:
  cal-files:
    xmm-ccf: false
    chandra: false
    xspec-models: false
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
title: Using Astroquery to download observation data from HEASARC
---

# Using Astroquery to download observation data from HEASARC

## Learning Goals

This notebook will teach you:
- How to retrieve HEASARC 'master' catalogs, which summarize the observations taken by a particular telescope.
- How to filter an observation summary table to find relevant observations for a single source, based on how close the observation was to the source.
- How to download the data files associated with those relevant observations.

## Introduction

This bite-sized tutorial will show you how to find and retrieve observation data files from HEASARC using Astroquery.

HEASARC hosts a large number of catalogs; the vast majority relate to the properties and/or locations
of astrophysical sources and have been produced by scientists during their research.

A small subset of the catalogs served by HEASARC are 'master' (or 'observation summary') catalogs, which
act as the primary index of every observation taken by a particular telescope.

We will filter an 'observation summary' catalog to find 'relevant' observations and then
show you how to acquire those data files.

### Runtime

As of 17th July 2026, this notebook takes ~5-minutes to run to completion
on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) using the 'small' server with 8GB RAM/ 2 cores.

Please note that this runtime is heavily dependent on archive servers, and the speed of
your internet connection, which means runtime will likely vary for users.

## Imports

```{code-cell} python
import glob
import os

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
```

***

## 1. Listing HEASARC's observation summary catalogs

We assume that you already have a basic understanding of HEASARC's Astroquery interface and how you search for
HEASARC catalogs – if not, please see the '{doc}`Find specific HEASARC catalogs using Python <../heasarc_catalogs/finding_relevant_heasarc_catalog>`'
bite-sized tutorial.

To fetch a list of every 'observation summary' catalog (see the [introduction](#introduction)) hosted by HEASARC, we
can simply run:

```{code-cell} python
all_obs_cat = Heasarc.list_catalogs(master=True)
```

To examine the contents of the returned table (see the
'{doc}`Exploring the contents of HEASARC catalogs using Python <../heasarc_catalogs/heasarc_catalog_contents>`'
tutorial, for an explanation of why we use `pprint_all()`) we can run:

```{code-cell} python
# The 'pprint' stands for 'pretty print'
all_obs_cat.pprint_all()
```

## 2. Selecting the observation summary catalog for your mission

We can pick out the name of a particular summary catalog by examining the `name` column, or we can
introduce a keyword search to find a more specific match (see the '{doc}`Find specific HEASARC catalogs using Python <../heasarc_catalogs/finding_relevant_heasarc_catalog>`'
tutorial for an explanation of keyword searches):

```{code-cell} python
filt_obs_cat = Heasarc.list_catalogs(keywords="suzaku", master=True)
filt_obs_cat
```

Then extract the name of the catalog:

```{code-cell} python
obs_cat_name = filt_obs_cat[0]["name"]
obs_cat_name
```

## 3. Filtering observations by distance from a source

One of the most common ways to select observations relevant to your science goal is to require
the nominal central coordinate of the pointing (or 'sky tile', for all-sky surveys) to be
within some radius of the source of interest.

The specific matching radius you choose will depend on:
1. The mission whose observations you're searching (every mission will have a different field of view, or FoV).
2. Which instrument you are most interested in (different instruments on the same mission will oftentimes have different FoVs).
3. Whether the instrument's FoV is circular, square, or rectangular (Chandra's ACIS-S, for instance, is often used in a rectangular configuration that is much longer than it is wide).
4. Your source and science goal – a low-redshift galaxy cluster, for instance, might motivate a larger matching radius as the whole source may not fit within the instrument's FoV.
5. If you only want observations where your source is near the center of the field, where many high-energy telescopes are the most sensitive and have the smallest point spread function (PSF).

Each of HEASARC's observation summary catalogs has a default search radius, which can be found using:

```{code-cell} python
default_search_rad = Heasarc.get_default_radius(obs_cat_name)
default_search_rad
```

We can also define our own search radius – in this instance let's assume we only want to
select observations that have our source in the very center of the FoV:

```{code-cell} python
custom_search_rad = Quantity(3, "arcmin")
custom_search_rad
```

Speaking of our source, we will search for Suzaku observations of PDS 456, a nearby radio-quiet quasar:

```{code-cell} python
source_name = "PDS 456"
```

A string source name variable can be passed directly into the search function we're about to use, which
will then use a name resolver to fetch the coordinate. Alternatively, we could do that ourselves and
pass the coordinate in:

```{code-cell} python
source_coord = SkyCoord.from_name(source_name)
source_coord
```

You should always treat the output coordinates of a name resolver with a little caution. They
are likely to be very reliable for point-like sources, but **for extended sources in particular**
there is often not a single 'correct' position definition, and you will want to check
exactly what coordinate has been returned.

In fact, you might want to define your own coordinate directly:

```{code-cell} python
manual_source_coord = SkyCoord(262.0825, -14.2655, unit="deg")
manual_source_coord
```

Finally, we can run the query that will filter our table of observations:

```{code-cell} python
source_obs_res = Heasarc.query_region(
    position=source_name, catalog=obs_cat_name, radius=custom_search_rad
)
source_obs_res
```

```{note}
We could have passed `position=source_coord` or `position=manual_source_coord` to the
above query, as we defined those coordinates in the discussion above.
```

## 4. Downloading observation data files

Now that we've identified some observations that are relevant to our source of interest (see
the end of [Section 3](#3-filtering-observations-by-distance-from-a-source)), we
can move on to downloading their data files.

The first step is to pass the return from our `Heasarc.query_region(...)` call (in our
case the Astropy `Table` assigned to `source_obs_res`) and pass it to the `locate_data(...)`
method of `Heasarc`. This function will construct a table of 'datalinks', which
describe where the relevant observation data are actually stored, and will provide us an
easy way of accessing them:

```{code-cell} python
source_obs_datalinks = Heasarc.locate_data(source_obs_res)
source_obs_datalinks
```

The table has several columns, including:
- **ID** – A unique International Virtual Observatory (IVO) ID for the data.
- **access_url** – A URL to one of the locations the data are stored, HEASARC's FTP server.
- **sciserver** – The path to the data if you are working on SciServer (see the [HEASARC@SciServer user guide](https://heasarc.gsfc.nasa.gov/docs/sciserver/)).
- **aws** – A URI that points to where the data are stored in the HEASARC Amazon Web Services (AWS) S3 bucket (see the [registry of open data on AWS](https://registry.opendata.aws/nasa-heasarc/)).

This means that when we come to download the data, we have a choice of _where to download it from_. Unless you are
working on SciServer, we generally recommend pulling data from our S3 bucket.

Now we set up a new directory using `os.makedirs(...)` (the `exist_ok=True` argument ensures that no
error is raised if that directory already exists), and start the download.

We pass our datalink table (`source_obs_datalinks`), tell the function to download from the
HEASARC S3 bucket (`host='aws'`), and make sure the downloaded files are placed in the
directory specified by `download_dir` (they would be placed in your current directory if
you didn't pass anything to the `location=` argument):

```{code-cell} python
# Define download path, and create the directories
download_dir = f"heasarc_data/{obs_cat_name}"
os.makedirs(download_dir, exist_ok=True)

# Triggers the download
Heasarc.download_data(links=source_obs_datalinks, host="aws", location=download_dir)
```

```{caution}
If the specified data files already exist in your `download_dir`, then this process will
overwrite them.
```

Finally, we can take a look at the contents of the download directory:

```{code-cell} python
os.listdir(download_dir)
```

As well as the contents of one of the observation directories:

```{code-cell} python
glob.glob(os.path.join(download_dir, "707035020") + "**/*")
```

Then a specific instrument directory:

```{code-cell} python
glob.glob(os.path.join(download_dir, "707035020", "xis") + "**/*")
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-07-16

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[Latest Astroquery Documentation](https://astroquery.readthedocs.io/en/latest/)

We provide several bite-sized tutorials on accessing HEASARC catalogs using Python and Astroquery:
- To learn how to use Python and Astroquery to search for a particular HEASARC catalog, please see the '{doc}`Find specific HEASARC catalogs using Python <../heasarc_catalogs/finding_relevant_heasarc_catalog>`' tutorial.
- To learn how to use Python and Astroquery to retrieve and explore the contents of HEASARC catalogs, please see the '{doc}`Exploring the contents of HEASARC catalogs using Python <../heasarc_catalogs/heasarc_catalog_contents>`' tutorial.
- To learn how to use Python and Astroquery to cross-match a local catalog (either locally on-disk or loaded into local memory) to a catalog hosted by HEASARC, please see the '{doc}`Cross-matching local and HEASARC catalogs using Python <../heasarc_catalogs/uploading_matching_table_heasarc_catalogs>`' tutorial.


### Acknowledgements

### References

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) – _astroquery: An Astronomical Web-querying Package in Python_
