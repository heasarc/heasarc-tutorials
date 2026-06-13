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
title: Cross-matching local and HEASARC catalogs using Python
---

# Cross-matching local and HEASARC catalogs using Python

## Learning Goals

By the end of this tutorial, you will:

- Have used the Astroquery Python package.
- Be able to cross-match a local catalog with a HEASARC-hosted catalog.
- Understand how to upload the local catalog so that matching is performed on HEASARC servers.

## Introduction

In this bite-sized tutorial we take you through the process of cross-matching a local
catalog (either stored locally or loaded into local memory) to a catalog hosted by HEASARC.

This demonstration uploads a catalog table to HEASARC's table access protocol (TAP)
service. That then runs an Astronomical Data Query Language (ADQL) query we set up
to find matches between the local and HEASARC catalogs and returns the results.

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
import pandas as pd
from astropy.table import Table
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
```

***

## 1. Prepare our sample for cross-matching to a HEASARC catalog

We assume that you have a catalog of sources available and that you wish to find
matches in one of HEASARC's catalogs. In this instance we also assume that the
catalog is formatted as a 'comma separated values' (CSV) file.

To demonstrate, we use a relatively small set of galaxy clusters from the
SDSSRM-XCS sample ([Giles P. A. et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.3878G/abstract); [Turner D. J. et al. 2025](https://ui.adsabs.harvard.edu/abs/2025MNRAS.537.1404T/abstract)).

First, we define the path to the CSV file (I know it isn't really a 'local' file, but
you could set this to the path to a CSV on your machine that you wish to cross-match!):

```{code-cell} python
# Set up URL of a sample file
samp_path = (
    "https://github.com/DavidT3/XCS-Mass-II-Analysis/raw/refs/heads/main/"
    "sample_files/SDSSRM-XCS_base_sample.csv"
)
```

You might have noticed that we imported the 'Pandas' module in the
['Imports Section'](#imports) - we're going to use it to read our CSV file into
a Pandas DataFrame.

As we've pointed out, the path to the file we're using in this example is actually a
URL, but the `read_csv()` function can handle both remote and local files.

```{code-cell} python
# Load the CSV file using Pandas
samp = pd.read_csv(samp_path)
samp
```

```{caution}
The query we submit in [Section 4](#4-run-the-cross-match-and-retrieve-the-results) can
be sensitive to certain symbols in column names. Having a column name with a '-'
symbol, for instance, will cause an error.

We also note that queries are **not** case-sensitive, so having columns
named 'e_kT' and 'E_kT' to indicate non-symmetrical uncertainties, for instance, would
trigger an error message about duplicate column names.
```

As per our warning above, some of the column names in our sample would cause errors when
we try to upload them to the HEASARC TAP service, so we'll replace the offending
symbols with something else:

```{code-cell} python
mod_samp_cols = samp.columns.str.replace("-", "minus")
mod_samp_cols = mod_samp_cols.str.replace("+", "plus")

samp.columns = mod_samp_cols
```

Now we will convert our Pandas DataFrame to an Astropy Table, which we'll be able to
pass directly to an Astroquery query function in [Section 4](#4-run-the-cross-match-and-retrieve-the-results):

```{code-cell} python
# Use a useful astropy table method to convert to Pandas dataframe
samp_tab = Table.from_pandas(samp)
samp_tab
```

You might have a catalog stored as a FITS, ASCII, or even HDF5 file - as long as
it is in the form of an Astropy Table by this point, the rest of the steps in this
demonstration should work.

## 2. The HEASARC TAP service

HEASARC, along with many other virtual observatory (VO) services, offers a table
access protocol (TAP) service. That means that we can perform operations on
HEASARC-hosted tables by constructing ADQL queries.

The HEASARC TAP service supports the **upload** of tables for use by queries, which we
will be using for this demonstration; note, however, that not all VO services support
table upload.

We will use the [Astroquery](https://github.com/astropy/astroquery) Python package to
interact with the HEASARC TAP service in this tutorial. Specifically, the
`Heasarc.query_tap(...)` method:

```{code-cell} python
# Show the docstring for the query_tap method
help(Heasarc.query_tap)
```

## 3. Construct a matching query

For this demonstration, we're assuming that you already have a HEASARC-hosted catalog
in mind; if not, you might find the
'{doc}`Find specific HEASARC catalogs using Python <finding_relevant_heasarc_catalog>`'
tutorial useful.

We are going to cross-match our sample to the 'Second ROSAT all-sky survey' source
catalog (2RXS; [Boller T. et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract)). HEASARC's table name for this catalog is:

```{code-cell} python
heasarc_cat_name = "rass2rxs"
```

Now we must decide how close a 2RXS entry has to be to a source in our sample to be
considered a match. As we're demonstrating using a sample of galaxy
clusters (extended objects), we choose a fairly large matching distance - you
should adjust this based on your own use case:

```{code-cell} python
match_dist = Quantity(2, "arcmin")
```

There are two sets of RA-Dec coordinates for each entry in our local catalog. The
difference between them is irrelevant to this demonstration, but it is an important
reminder that you will likely have to adjust the column names for your local
catalog in the ADQL query.

To make that a little easier, we define the RA/Dec column names we want to use here:

```{code-cell} python
local_ra_col = "rm_ra"
local_dec_col = "rm_dec"
```

Now we construct a simple ADQL query that will return all columns (`SELECT *`) and
entries where (`WHERE` - unsurprisingly) the coordinate of a 2RXS
entry (`point('ICRS',cat.ra,cat.dec)`) is within (`contains(...)`) a circle with
radius `match_dist` centered on a source in our
sample (`circle('ICRS',loc.{lra},loc.{ldec},{md})`):

```{code-cell} python
query = (
    "SELECT * "
    "FROM {hcn} as cat, tap_upload.local_samp as loc "
    "WHERE "
    "contains(point('ICRS',cat.ra,cat.dec), "
    "circle('ICRS',loc.{lra},loc.{ldec},{md}))=1".format(
        md=match_dist.to("deg").value.round(4),
        lra=local_ra_col,
        ldec=local_dec_col,
        hcn=heasarc_cat_name,
    )
)

query
```

```{seealso}
A general tutorial on the many uses and features of ADQL is out of the scope of this
bite-sized demonstration. Various resource for learning ADQL are available online, such
as [this short course](https://docs.g-vo.org/adql/) ([Demleitner M. and Heinl H. 2024](https://dc.g-vo.org/voidoi/q/lp/custom/10.21938/uH0_xl5a6F7tKkXBSPnZxg)),
or the NASA Astronomical Virtual Observatories (NAVO)
[catalog queries tutorial](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html).
```

## 4. Run the cross-match and retrieve the results

Finally, we can run our cross-match!

The query we designed will be passed to the HEASARC TAP service through the
Astroquery module.

When we run this query, we also upload our sample table (prepared in
[Section 1](#1-prepare-our-sample-for-cross-matching-to-a-heasarc-catalog)). Notice
that the key we assign our table in the `uploads` dictionary is the same as the table
name in the ADQL query we prepared in [Section 3](#3-construct-a-matching-query).

```{code-cell} python
cat_match = Heasarc.query_tap(query, uploads={"local_samp": samp_tab})
```

Note that the `Heasarc.query_tap(...)` call submits a **synchronous** query to the
HEASARC TAP service, as opposed to an ***asynchronous*** query.

[A discussion of the differences can be found here](https://pyvo.readthedocs.io/en/stable/dal/#synchronous-vs-asynchronous-query), but
the summary is that a synchronous query will stay connected to the HEASARC service until the table
operation is complete and the results are returned, whereas submitting a query asynchronously will send
the job to HEASARC, get a URL reporting the status of the job in return, and then that URL
will have to be polled to find out when the results are ready.

Asynchronous submission is preferable for long-running queries, as it won't be
sensitive to any network issues that might occur while waiting for the results like
a synchronous query would be. Asynchronous HEASARC queries cannot currently be
submitted through Astroquery, though you could instead use the [PyVO module](https://github.com/astropy/pyvo).

Our match results have been returned, and we can convert them into an Astropy Table object, as they
can be a little easier to work with than the `TAPResults` object we received.

By putting the variable name at the bottom of the code cell, we can see a nice rendered
version of the table (only in a Jupyter notebook environment) and see that we do
have some matches!

```{code-cell} python
cat_match_tab = cat_match.to_table()
# Visualize the table
cat_match_tab
```

Seeing as there are a lot of columns in the results table (all the columns from the
uploaded table and the 2RXS catalog), we can check what columns are available by
accessing the `colnames` property of our table:

```{code-cell} python
cat_match_tab.colnames
```

Then we could pull out the values of a particular column as a Numpy array:

```{code-cell} python
cat_match_tab["cat_count_rate"].value.data
```

```{seealso}
An additional resource for learning about the use of virtual observatory services
is the [NASA Astronomical Virtual Observatories (NAVO) workshop notebook set](https://nasa-navo.github.io/navo-workshop/).

[Section 3 of the 'Catalog Queries' notebook](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html#using-the-tap-to-cross-correlate-and-combine) is particularly relevant to this bite-sized tutorial.
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-03-16

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[Short Course on ADQL Website](https://docs.g-vo.org/adql/)

[NAVO Workshop](https://nasa-navo.github.io/navo-workshop/)

[NAVO catalog queries tutorial](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html#using-the-tap-to-cross-correlate-and-combine)

[Astroquery GitHub Repository](https://github.com/astropy/astroquery)

[Latest Astroquery Documentation](https://astroquery.readthedocs.io/en/latest/)

[Description of synchronous and asynchronous queries](https://pyvo.readthedocs.io/en/stable/dal/#synchronous-vs-asynchronous-query)

### Acknowledgements


### References

[Giles P. A., Romer A. K., Wilkinson R., Bermeo A., Turner D. J. et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.3878G/abstract) - _The XMM Cluster Survey analysis of the SDSS DR8 redMaPPer catalogue: implications for scatter, selection bias, and isotropy in cluster scaling relations_

[Turner D. J., Giles P. A., Romer A. K., Pilling J., Lingard T. K. et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025MNRAS.537.1404T/abstract) - _The XMM Cluster Survey: automating the estimation of hydrostatic mass for large samples of galaxy clusters ─ I. Methodology, validation, and application to the SDSSRM-XCS sample_

[Boller T., Freyberg M.J., Trümper J. et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract) - _Second ROSAT all-sky survey (2RXS) source catalogue_
