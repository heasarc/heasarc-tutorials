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
title: Using Astroquery to find and acquire HEASARC data
---

# Using Astroquery to find and acquire HEASARC data

## Learning Goals

This notebook will teach you:
- How to retrieve HEASARC 'master' catalogs, which summarize the observations taken by a particular telescope.
- How to filter an observation summary table to find relevant observations.
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

As of 17th July 2026, this notebook takes ~<span style="color:red">**??**</span>-seconds to run to completion on Fornax using the 'small' server with 8GB RAM/ 2 cores.

## Imports

This notebook uses features from an Astroquery pre-release. You will need to install
the latest version using the command below. We will remove this once Astroquery
v0.4.12 is officially released.

```{code-cell} python
---
tags: [hide-output]
jupyter:
  output_hidden: true
---
%pip install --pre astroquery --upgrade
```

```{code-cell} python
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

## ??. Downloading observation data files

```{code-cell} python
Heasarc.locate_data()
```

```{code-cell} python
Heasarc.download_data()
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

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) - _astroquery: An Astronomical Web-querying Package in Python_

[Cavagnolo K. W., Donahue M., Voit G. M., Sun M. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJS..182...12C/abstract) - _Intracluster Medium Entropy Profiles for a Chandra Archival Sample of Galaxy Clusters_

[Evans I. N., Evans J. D., Martínez-Galarza J. R., Miller J. B. et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJS..274...22E/abstract) - _The Chandra Source Catalog Release 2 Series_

[Chandra Source Catalog 2 DOI - doi:10.25574/csc2](https://doi.org/10.25574/csc2)

[Demleitner M. and Heinl H. (2024)](https://dc.g-vo.org/voidoi/q/lp/custom/10.21938/uH0_xl5a6F7tKkXBSPnZxg) - _A Short Course on ADQL; Virtual Observatory Resource_
