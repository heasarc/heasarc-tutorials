---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-03-16'
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
title: Exploring the contents of HEASARC catalogs using Python
---

# Exploring the contents of HEASARC catalogs using Python

## Learning Goals

This notebook will teach you:
- How to retrieve and explore a HEASARC catalog's column names, descriptions, and units.
- How to retrieve the entire contents of a HEASARC catalog.
- How to retrieve a subset of a HEASARC catalog with easy-to-use Astroquery features.

## Introduction

This bite-sized tutorial will show you how to retrieve and explore the contents of HEASARC catalogs in Python.

To learn how to use Python to search for a particular HEASARC catalog, please see the '{doc}`Find specific HEASARC catalogs using Python <finding_relevant_heasarc_catalog>`' tutorial.

### Runtime

As of 16th March 2026, this notebook takes ~30 s to run to completion on Fornax using the 'small' server with 8GB RAM/ 2 cores.

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
from astroquery.heasarc import Heasarc
```

***

## 1. Listing a HEASARC catalog's columns

For this demonstration, we're assuming that you already have a HEASARC-hosted catalog
in mind; if not, you might find the
'{doc}`Find specific HEASARC catalogs using Python <finding_relevant_heasarc_catalog>`'
tutorial useful.

We will use the Archive of Chandra Cluster Entropy Profile Tables (ACCEPT) catalog
([Cavagnolo K. W. et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJS..182...12C/abstract))
as an example.

The best way to get an idea of a catalog's contents is to list the column
names and descriptions. We can do this using the `Heasarc.list_columns(...)`
method, passing the name of the catalog as the first argument.

Each HEASARC catalog has a subset of 'standard' columns that will be returned by
default, which is why the table below contains only a few column names, descriptions, and
units even though this catalog has *79* columns:

```{code-cell} python
# We pass the name of the catalog as the first argument
Heasarc.list_columns("acceptcat")
```

If, as is likely, you want to examine the full set of columns, you can pass the
`full=True` argument:

```{code-cell} python
# Pass an additional argument to ensure all columns are returned
all_accept_cols = Heasarc.list_columns("acceptcat", full=True)
all_accept_cols
```

If you examine the output of the above cell, you'll notice that only part of the
table has been displayed; this is a common behavior when displaying long tables in
Jupyter notebooks, across multiple modules (e.g., Astropy, Pandas, etc.), as 'printing'
many lines can dramatically affect Jupyter's performance.

On the other hand, a table like this isn't going to destroy your computer, so we can
safely sidestep this issue by using the `pprint_all()` method of the `list_columns()`
output (an Astropy `Table` object, more on them in [Section 4](#4-interacting-with-heasarc-catalog-contents)):

```{code-cell} python
# The 'pprint' stands for 'pretty print'
all_accept_cols.pprint_all()
```

## 2. Retrieving the entire contents of a HEASARC catalog

The simplest use case of a HEASARC catalog is that you want to retrieve the
entire table.

We can easily fetch the entire catalog using Astroquery functions, but
before we do, we should check how many rows there are - we want to know what we're
getting into with respect to the size of the table.

Counting the rows in a HEASARC catalog involves writing a very simple 'Astronomical
Data Query Language' (ADQL) query.

ADQL is a cousin of the extremely popular 'Structured Query Language' (SQL) that has
been used for database management in industry for many years; the syntax is similar, but
with additions specific to astronomical searches.

We use the `COUNT(*)` function to return the number of rows in a table:

```{code-cell} python
# Send query designed to count the rows of a catalog
accept_nrow_res = Heasarc.query_tap("SELECT COUNT(*) FROM acceptcat")

# Store the integer number of rows in a variable
accept_nrows = accept_nrow_res["count"][0]

# Visualize the returned table
accept_nrow_res
```

From the output above, we can see that there are 'only' 240 rows in the catalog; combine that information with
the number of columns (which we explored in [Section 1](#1-listing-a-heasarc-catalogs-columns)), and you
get a sense of the table's scale.

As the ACCEPT catalog is quite small (relatively speaking), we can retrieve the whole table without worrying
about download time or memory issues.

```{seealso}
A general tutorial on the many uses and features of ADQL is out of the scope of this
bite-sized demonstration. Various resources for learning ADQL are available online, such
as [this short course](https://docs.g-vo.org/adql/) ([Demleitner M. and Heinl H. 2024](https://dc.g-vo.org/voidoi/q/lp/custom/10.21938/uH0_xl5a6F7tKkXBSPnZxg)),
or the NASA Astronomical Virtual Observatories (NAVO)
[catalog queries tutorial](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html).
```

On the other hand, HEASARC hosts much larger catalogs than ACCEPT. The Chandra Source
Catalog 2 (CSC 2; [Evans I. N. et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJS..274...22E/abstract)),
for instance:

```{code-cell} python
# Same again, but CSC
Heasarc.query_tap("SELECT COUNT(*) FROM csc")
```

```{warning}
For large catalogs like the CSC, we do not recommend retrieving the entire table at once.
```

Finally, now we know that retrieving the entire ACCEPT catalog is reasonable, we can
use the `query_region(...)` method of `Heasarc` to do just that. Few arguments are
required:
- `catalog="acceptcat"` - specifies the name of the catalog table to retrieve.
- `spatial="all-sky"` - overrides `query_region`'s default behavior of searching a catalog around a given coordinate, and instead considers the entire catalog.
- `columns="*"` - specifies that all columns should be returned (otherwise you will get a small subset, as discussed in [Section 1](#1-listing-a-heasarc-catalogs-columns).

```{code-cell} python
accept_cat = Heasarc.query_region(catalog="acceptcat", spatial="all-sky", columns="*")
accept_cat
```

## 3. Retrieving a subset of a HEASARC catalog

If you aren't interested in the _entire_ catalog, then we can also use Astroquery to
impose some restrictions on the rows we retrieve, based on the values of certain columns.

For example, perhaps we're only interested in galaxy clusters with a $z>0.4$. We saw in
[Section 1](#1-listing-a-heasarc-catalogs-columns) that the ACCEPT catalog includes
a column called `redshift`, we can use that to filter the rows we retrieve.

The `query_region(...)` call below will return all columns (`columns="*"`), and all rows where the
value of the `redshift` column is greater than 0.4 (`column_filters={"redshift": (">", "0.4")}`):

```{code-cell} python
accept_cat_higherz = Heasarc.query_region(
    catalog="acceptcat",
    spatial="all-sky",
    column_filters={"redshift": (">", "0.4")},
    columns="*",
)
accept_cat_higherz
```

If we want to further restrict the results, we can use boolean operators to add extra
filtering conditions. Here, for instance, we've decided we only want the
higher-redshift, low-central-entropy, galaxy clusters to be returned:

```{code-cell} python
accept_cat_higherz_lowk = Heasarc.query_region(
    catalog="acceptcat",
    spatial="all-sky",
    column_filters={"redshift": (">", "0.4"), "bf_core_entropy_1": ("<", "15")},
    columns="*",
)
accept_cat_higherz_lowk
```

## 4. Interacting with HEASARC catalog contents

The returns from our calls to the `Heasarc.query_region` method are Astropy
`Table` objects:

```{code-cell} python
type(accept_cat)
```

You can extract information similarly to a Pandas `DataFrame`; e.g., indexing with a
column name string retrieves the entries in that column:

```{code-cell} python
# Extract source names from our subset of the ACCEPT catalog
accept_cat_higherz_lowk["name"]
```

To retrieve the entries for a **row** in the table, you can index with an
integer; e.g., `0` for the first row:

```{code-cell} python
accept_cat_higherz_lowk[0]
```

You can also convert the return to a Pandas `DataFrame` if you prefer working with
one of these data structures:

```{code-cell} python
accept_cat_higherz_lowk_pd = accept_cat_higherz_lowk.to_pandas()
accept_cat_higherz_lowk_pd
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-03-16

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[Latest Astroquery Documentation](https://astroquery.readthedocs.io/en/latest/)

[Short Course on ADQL Website](https://docs.g-vo.org/adql/)

[NAVO catalog queries tutorial](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html#using-the-tap-to-cross-correlate-and-combine)

[Latest PyVO Documentation](https://pyvo.readthedocs.io/en/latest/)

### Acknowledgements

### References

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) - _astroquery: An Astronomical Web-querying Package in Python_

[Cavagnolo K. W., Donahue M., Voit G. M., Sun M. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJS..182...12C/abstract) - _Intracluster Medium Entropy Profiles for a Chandra Archival Sample of Galaxy Clusters_

[Evans I. N., Evans J. D., Martínez-Galarza J. R., Miller J. B. et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJS..274...22E/abstract) - _The Chandra Source Catalog Release 2 Series_

[Chandra Source Catalog 2 DOI - doi:10.25574/csc2](https://doi.org/10.25574/csc2)

[Demleitner M. and Heinl H. (2024)](https://dc.g-vo.org/voidoi/q/lp/custom/10.21938/uH0_xl5a6F7tKkXBSPnZxg) - _A Short Course on ADQL; Virtual Observatory Resource_
