---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-06-01'
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 1.3
    jupytext_version: 1.17.3
kernelspec:
  display_name: heasoft
  language: python
  name: heasoft
execution:
  cal-files:
    xmm-ccf: True
    chandra: True
    xspec-models: True
title: How to make a tutorial notebook in the HEASARC-tutorials repository
---

# Title: How to make a bite-sized HEASARC-tutorials notebook

One of your first steps in adapting this template should be to fill out the 'front-matter' at the very top of the
Markdown file - the contents of the front-matter are all metadata, to be used in different ways by different
parts of the HEASARC-tutorials infrastructure.

The following information must be added:
- Authors:
  - Anyone who you consider an author of this notebook, following the traditional 'first author contributed the most' ordering.
  - Affiliations must be included, either as a list or a string.
  - Email, ORCID, and personal website are optional, but please consider including them if you have them, as we will produce a DOI for this resource.
- Date:
  - The date this notebook was last updated, following the year-month-day format.
- Execution:
  - 'cal-files' specifies the calibration files that will be needed to run this notebook. This information is both useful to the user and to the HEASARC-tutorials execution infrastructure, where it ensures we don't have to load all possible calibration files for every notebook. Each entry may have a value of True or False. HEASoft's CalDB does not have an entry because these notebooks are set up to use remote CalDB files.


## Learning Goals

By the end of this tutorial, you will:

- Know to only put **1** to **3** goals here; one sentence per goal.
- Understand what has to be present in a **bite-sized** tutorial notebook.

## Introduction

_Please include 'bite-sized', 'bite sized', or 'bite size(d)' in the title or introduction._

Alter this file according to your use case but retain the basic structure and try to
use the same syntax for things like section headings, numbering schemes, and bullet points.

**ONLY MODIFY OR REMOVE SECTION HEADINGS BETWEEN 'IMPORTS' AND 'ABOUT THIS NOTEBOOK'**

All contributed notebooks should be in the [MyST markdown](https://mystmd.org) format; you can visit
the [Fornax documentation](https://docs.fornax.sciencecloud.nasa.gov/markdown-notebooks/) for more information.

The introduction should provide context and motivation:
- Why should a scientist use this notebook?
- What is the background on the science or technical problem addressed?

**A BITE-SIZED NOTEBOOK INTRODUCTION SHOULD BE SHORT AND TO THE POINT, IF IT IS MORE
THAN FIVE SECTIONS LONG, OR CONTAINS TOO MANY SUBSECTIONS, LINES OF COMMENTARY, OR
TEACHES TOO MANY SKILLS THEN YOU WILL NEED TO REWRITE IT AS A FULL NOTEBOOK**

### Runtime

Please report actual numbers and machine details for your notebook if it is expected to run longer or requires specific machines, for example, on Fornax.
Also, if querying archives, please include a statement like:
"This runtime is heavily dependent on archive servers, which means that the runtime may vary for users".

**OR**

"This notebook depends on external services, such as the HEASARC archive, and therefore the runtime may vary for users."

Here is a template runtime statement:
As of {Date}, this notebook takes ~{N}-seconds to run to completion on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) using the '{name: size}' server with NGB RAM/ N cores.

## Imports

_**All** imports should be placed in this cell, unless they absolutely must be somewhere else (you will be asked to justify this in review)._

```{code-cell} python
import numpy as np
```

***

## 1. Data Access

The name of this and all future sections can change.
In general, it probably is a good idea to start with something like "Data Access".
Please note, and stick to, the existing numbering scheme.

**KEEP SUB-SECTIONS TO A MINIMUM FOR BITE-SIZED NOTEBOOKS**

**THERE SHOULD BE NO MORE THAN 5 SECTIONS IN A BITE-SIZED NOTEBOOK**

```{code-cell} python
# Create some example data.
data = np.random.randint(0, 100, size=100)
```

## 2. Data Exploration

Describe what the data look like.
Add summary statistics, initial plots, sanity checks.

For cuts or other data filtering and cleaning steps, explain the scientific reasoning behind them.
This helps people understand both the notebook and the data so that they're more equipped to use the data appropriately in other contexts.

+++

:::{tip}
Please include a narrative for *all* your code cells to help the reader figure out what you are doing and why you chose that path.

Using [MyST admonitions](https://mystmd.org/guide/admonitions) such as this `tip` are encouraged
:::

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

hist, bin_edges = np.histogram(data, bins=10)
hist
```

:::{important}
The HEASARC-tutorials style guide requires that cells using matplotlib (or similar) to produce figures
should be isolated (i.e., only contain plotting code), and must include the following metadata to hide the
code from view (see the source of this cell for the unrendered text:

---
tags: [hide-input]
jupyter:
  source_hidden: true
---
:::

For any Figures, please add a few sentences about what the users should be noticing.

+++

## 3. Section Three

Lorem ipsum... X-rays!

## 4. Section Four

...and more X-rays!

+++

## About this notebook

Author: _Specific author and/or team name. One author per line._

Updated On: {Date}
+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

### Acknowledgements

_Did anyone help you? For instance:_

'We are grateful for assistance from the MAST & IRSA Fornax teams.'

_If generative 'artificial intelligence' was used for any part of the
demonstration, please include something like:_
"AI: This notebook was created with assistance from OpenAI’s ChatGPT 5 model."

### References

_Any citations should be added here, for instance:_

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) - _astroquery: An Astronomical Web-querying Package in Python_

[Cavagnolo K. W., Donahue M., Voit G. M., Sun M. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJS..182...12C/abstract) - _Intracluster Medium Entropy Profiles for a Chandra Archival Sample of Galaxy Clusters_

[Mehrtens N., Romer A. K., Hilton M. et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.1024M/abstract) - _The XMM Cluster Survey: optical analysis methodology and the first data release_
