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

# Title: How to make a tutorial notebook in the HEASARC-tutorials repository

One of your first steps in adapting this template should be to fill out the 'front-matter' at the very top of the
Markdown file – the contents of the front-matter are all metadata, to be used in different ways by different
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

By the end of this tutorial, you will be able to (list 2 - 5 high level goals):

-   Write a python tutorial using [MyST markdown](https://mystmd.org) format.
-   Meet all of the checklist requirements to submit your code for code review.

## Introduction

Alter this file according to your use case but retain the basic structure and try to use the same syntax for things like section headings, numbering schemes, and bullet points.
Specifically the headings in this Intro section should not be edited to maintain consistency between notebooks.

All contributed notebooks should be in [MyST markdown](https://mystmd.org) format.
See the [Fornax documentation](https://docs.fornax.sciencecloud.nasa.gov/markdown-notebooks/) for more info about this.

The Introduction should provide context and motivation.
Why should someone use this notebook?
Give background on the science or technical problem.
Point out the parts that are particularly challenging and what solutions we chose for what reasons.

### Inputs

-   List the data, catalogs, or files needed, and where they come from.
    If there are data that get downloaded to Fornax as part of this notebook, place those in a `data` directory.
    Please do not change the name of this directory for consistency with other notebooks.
    Do not add the contents of `data` to the repo, just the empty directory.

### Outputs

-   List the products the notebook generates (plots, tables, derived data, etc.)
-   If there are intermediate products produced by your notebook, generate an `output` directory for those data.
    Please do not change the name of this directory for consistency with other notebooks.
    Do not add the contents of `output` to the repo, just the empty directory.

### Runtime

Please report actual numbers and machine details for your notebook if it is expected to run longer or requires specific machines, for example, on Fornax.
Also, if querying archives, please include a statement like:
"This runtime is heavily dependent on archive servers, which means that the runtime may vary for users".

**OR**

"This notebook depends on external services, such as the HEASARC archive, and therefore the runtime may vary for users."

Here is a template runtime statement:
As of {Date}, this notebook takes ~{N}-seconds to run to completion on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) using the '{name: size}' server with NGB RAM/ N cores.

## Imports

This should be a list of the modules that are required to run this code.
Importantly, even those that are already installed in Fornax should be listed here so users wanting to run this locally on their own machines have the information they need to do this.

Make sure that you have built a "requirements_notebook_name.txt" file with the modules to be imported.
The name of the notebook should be present in the name of the requirements file, as in our example "requirements_notebook_template.txt"

```{code-cell} python
# This cell should not be edited below this line except for the name of
#  the requirements_notebook_name.txt

# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_notebook_name.txt
```

```{code-cell} python
import numpy as np
```

## Global Setup

### Functions

Please avoid writing functions where possible, but if they are necessary, then place them in the following
code cell - it will be minimized unless the user decides to expand it. **Please replace this text with concise
explanations of your functions or remove it if there are no functions.**

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

```

***

## 1. Data Access

The name of this, and all future sections can change.
In general, it probably is a good idea to start with something like "Data Access".
Please note, and stick to, the existing numbering scheme.

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

## 3. Analysis

The working part of the notebook.
Lay out the step-by-step analysis workflow.
Each subsection should describe what is being done and why.
These can be sections or subsections.

+++

### 3.1 Design Principles

-   Make no assumptions: define terms, common acronyms, link to things you reference.
-   Keep in mind who your audience is.
-   Design for portability - will this notebook work on both Fornax and someone's individual laptop.
-   Cells capture logical units of work.
-   Use markdown before or after cells to describe what is happening in the notebook.

+++

### 3.2 Style Principles

-   Follow suggestions of The Turing Way community [markdown style](https://book.the-turing-way.org/community-handbook/style)
-   Write each sentence in a new line (line breaks) to make changes easier to read in PRs.
-   Avoid latin abbreviation to avoid failing CI.

#### 3.2.1 Best Practice Guidelines
It would be nice if all contributed codes did the following, but these guidelines will not be checked in a code review

-   Section titles should not end with ":".
-   List items should start at the beginning of the line, no spaces first. Exception is nested lists.
-   One empty line between section header and text.
-   One empty line before a list and after.
-   No more than one empty line between any two non-empty lines.

```{code-cell} python

```

## 4. PR Review

Notebooks go through a two step process: first step is getting into the repo, and the second step gets it into the [published tutorials](https://nasa-fornax.github.io/fornax-demo-notebooks/).
Final notebooks are expected to go through both a science and tech review checklist.
Checklists are [here](https://github.com/nasa-fornax/fornax-demo-notebooks/tree/main/template/notebook_review_checklists.md).
Please consider these checklist requirements as you are writing your code.

The first PR of a notebook does not need to have everything from the checklists completed, but should have all the pieces there, and the authors should be aware of the requirements.

To complete the second step of this process and be both rendered and included in users Fornax home directories, both a science and technical reviewer will be looking at [this checklist](https://github.com/nasa-fornax/fornax-demo-notebooks/tree/main/template/notebook_review_checklists.md) to see if the new tutorial notebook meets all of the requirements, or has a reasonable excuse not to.

Any PRs can be opened as drafts, which is in fact preferred, if authors are still working on them.

+++

## About this notebook

-   **Authors:** Specific author and/or team names, plus "and the Fornax team".
-   **Contact:** For help with this notebook, please open a topic in the [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov/) "Support" category.
-   Please edit and keep the above 2 bullet points, and remove this last line.

+++

### Additional Resources


### Acknowledgements

Did anyone help you?
Probably these teams did, so include them: MAST, HEASARC, & IRSA Fornax teams.

Did you use AI for any part of this tutorial, if so please include a statement such as:
"AI: This notebook was created with assistance from OpenAI’s ChatGPT 5 model.", which is a good time to mention that this template notebook was created with assistance from OpenAI’s ChatGPT 5 model.

### References

This work made use of:

-   STScI style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
-   Fornax tech and science review guidelines: https://github.com/nasa-fornax/fornax-demo-notebooks/blob/main/template/notebook_review_checklists.md
-   The Turing Way Style Guide: https://book.the-turing-way.org/community-handbook/style
