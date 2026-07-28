---
authors:
- name: Abdu Zoghbi
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
  orcid: 0000-0002-0572-9613
  website: https://science.gsfc.nasa.gov/sci/bio/abderahmen.zoghbi
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-07-28'
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
mystnb:
  execution_allow_errors: false
title: Getting started with HEASoftPy
---

# Getting started with HEASoftPy
This tutorial provides a quick-start guide to using `heasoftpy`, a Python wrapper for
the high-energy astrophysics software HEASoft.

## Learning Goals
By the end of this tutorial, you will:

- Understand the basic usage of HEASoftPy and the different ways of calling HEASoft tasks.
- Learn about additional options for running pipelines and parallel jobs.

## Introduction
`heasoftpy` is a Python wrapper around the legacy high-energy software suite
HEASoft, which supports analysis for many active and past NASA X-ray and Gamma-ray
missions; it allows HEASoft tools to be called from Python scripts, interactive iPython
sessions, or Jupyter Notebooks.

This tutorial presents a walk through the main features of `heasoftpy`.

### Inputs

- The ObsID of the NuSTAR data used in example 4.
- The ObsIDs of the NICER data used in example 5.

### Outputs

- A filtered pre-processed NICER event list.
- Partially processed NuSTAR data.
- Processed NICER data.

### Runtime
As of 3rd November 2025, this notebook takes ~15-minutes to run to completion on Fornax, using the 'small' server with 8GB RAM/ 2 cores.

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
import multiprocessing as mp
import os

import heasoftpy as hsp
from astroquery.heasarc import Heasarc

%matplotlib inline
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def worker(in_dir: str) -> hsp.core.HSPResult:
    """
    A very simple demonstration of how you can wrap a HEASoftPy task call in order to
    be able to run it on several observations in parallel. In this case the function
    wraps the NICER level-2 processing pipeline, which will prepare data for
    scientific use.

    :param str in_dir: The directory containing the input data.
    :return: The output of the HEASoftPy task.
    :rtype: hsp.core.HSPResult
    """

    with hsp.utils.local_pfiles_context():

        # Call the tasks of interest
        out = hsp.nicerl2(
            indir=in_dir, noprompt=True, clobber=True, geomag_path=GEOMAG_PATH
        )

        # Run any other tasks...

    return out
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# The ObsID of the NuSTAR data we use in example four
NU_OBS_ID = "60001110002"

# The ObsIDs of the NICER data we use in example five
NI_OBS_IDS = [
    "1010010121",
    "1010010122",
    "1012020112",
    "1012020114",
]

# Sets the data host we want to download observation files from.
# The `download_data` method in Astroquery 0.4.12 will attempt to automatically
#  determine where to fetch data from - e.g. if you are running on SciServer then the
#  mounted HEASARC FTP will be used, if you are running on Fornax then AWS will be
#  used - if no specific host is supplied.
# We specify AWS, but you may set DATA_HOST = None to let Astroquery
#  decide automatically.
DATA_HOST = "aws"
```

### Configuration

Here we include code that downloads the data for our examples - we don't include it
in the main body of the notebooks as we do not wish it to be the main focus.

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

# Here we make sure we have all the data this notebook requires
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.abspath("../../../_data")
    nu_data_dir = os.path.join(ROOT_DATA_DIR, "NuSTAR", "")
    ni_data_dir = os.path.join(ROOT_DATA_DIR, "NICER", "")
else:
    ROOT_DATA_DIR = os.getcwd()
    nu_data_dir = "NuSTAR/"
    ni_data_dir = "NICER/"

nu_data_link = Heasarc.locate_data(
    Heasarc.query_tap(f"SELECT * from numaster where obsid='{NU_OBS_ID}'").to_table(),
    "numaster",
)

# We only download the data if a matching ObsID directory does not exist.
#  This is not a perfect way to determine whether the necessary data are fully
#  present, but it is good enough for this tutorial.
# The `download_data` method in Astroquery 0.4.12 will attempt to automatically
#  determine where to fetch data from - e.g. if you are running on SciServer then the
#  mounted HEASARC FTP will be used, if you are running on Fornax then AWS will be used.
if not os.path.exists(nu_data_dir + f"{NU_OBS_ID}/"):
    Heasarc.download_data(nu_data_link, host=DATA_HOST, location=nu_data_dir)

# Construct a string list of NICER ObsIDs to pass to the HEASARC TAP service.
ni_oi_str = "('" + "','".join(NI_OBS_IDS) + "')"
ni_data_links = Heasarc.locate_data(
    Heasarc.query_tap(
        f"SELECT * from nicermastr where obsid IN {ni_oi_str}"
    ).to_table(),
    "nicermastr",
)

# Again, we only download the data if a matching ObsID directory does not exist.
if any([not os.path.exists(os.path.join(ni_data_dir, oi)) for oi in NI_OBS_IDS]):
    Heasarc.download_data(ni_data_links, host=DATA_HOST, location=ni_data_dir)

# -------- Get geomagnetic data ---------
# This ensures that geomagnetic data required for NICER analyses are downloaded
GEOMAG_PATH = os.path.join(ROOT_DATA_DIR, "geomag")
os.makedirs(GEOMAG_PATH, exist_ok=True)
out = hsp.nigeodown(outdir=GEOMAG_PATH, allow_failure=False)
# ---------------------------------------
```

***


## Example 1: Accessing HEASoftPy help files

For general help, you can run `hsp?` (if working in a Jupyter or iPython
notebook) or use `hsp.help()`.

```{code-cell} python
hsp.help()
```

For task-specific help, you can do:

```{code-cell} python
hsp.ftlist?
```

Alternatively, you may use the HEASoft standard `fhelp`:

```{code-cell} python
hsp.fhelp(task="ftlist")
```

```{warning}
Note that the use of '?' is not valid in 'standard' Python, only in Jupyter
notebooks and iPython.
```

## Example 2: Exploring the content of a FITS file with `ftlist`

The simplest way to run a task is call the function directly: `hsp.task_name(...)`.

In this case, it is `hsp.ftlist(...)`

For `ftlist`, there two required inputs: `infile` and `option`, so that both
parameters need to be provided, otherwise, we will get prompted for the missing parameters.

`infile` is the name of the input FITS file. It can be a local or a remote file. In
this case, we use a FITS file from the HEASARC archive. We can specify the header data unit (HDU)
that we want to access from the FITS file in the usual way (e.g., append `[1]` to the file name).

We can also pass other optional parameters (`rows='1-5'` to specify which rows to print).

```{code-cell} python
infile = (
    "https://heasarc.gsfc.nasa.gov/FTP/nicer/data/obs/2017_10/1012010115/"
    "xti/event_cl/ni1012010115_0mpu7_cl.evt.gz[1]"
)
result = hsp.ftlist(infile=infile, option="T", rows="1-5")
```

The return of all task execution calls is an `HSPResult` object. Which is a convenient
object that holds the status of the call and its return. For example:

- `returncode`: a return code: 0 if the task executed without errors (int).
- `stdout`: standard output (str).
- `stderr`: standard error message (str).
- `params`: dict of the parameters used for the task.
- `custom`: dict of any other variables returned by the task.

In this case, we may want to just print the output as:

```{code-cell} python
print("return code:", result.returncode)
print(result.stdout)
```

With this, it may be useful to check that `returncode == 0` after every call if you
are not running the tasks interactively.

With `heasoftpy` version 1.5 and above. You can make the call raise a Python exception
when it fails. This feature is controlled by the config parameter: `allow_failure`.

Setting `hsp.config.Config.allow_failure = False`, means the task will raise
an `HSPTaskException` exception if it fails. Setting the value to `True`, means the
task will not raise an exception, and the return code value will need to be checked by the user.

The value is set to `True` by default for versions `<1.5`. For version `1.5`, not
setting the value prints a warning. In a future version, the default will change
to `False`, so that all failures raise an exception.

We can modify the parameters returned in `result`, and pass them again to the task.

Say we do not want to print the column header:

```{code-cell} python
params = result.params
params["colheader"] = "no"
result_no_col_hdr = hsp.ftlist(params)

print(result_no_col_hdr.stdout)
```

If we forget to pass a required parameter, we will be prompted for it. For example:

```{code-cell} python
---
mystnb:
  raises-exception: true
---
# result = hsp.ftlist(infile="../tests/test.fits")
```

would prompt for the `option` value:

```
Print options: H C K I T  [T] ..
```

In this case, parameter `ftlist:option` was missing, so we are prompted for it, and
the default value is printed between brackets: `[T]`, we can type a value and then
press 'Return' to set our own value, or just press 'Return' to accept the default value.

For tasks that take longer to run, the user may be interested in seeing the output as
the task runs. There is a `verbose` option to print the output of the command similar
to the standard output in command line tasks.

```{code-cell} python
result = hsp.ftlist(infile=infile, option="T", rows="1-5", verbose=True)
```

## Example 3: Using `ftselect`

In this second example, we will work with the same `infile` from above.

We see is the first HDU of the file is an events table. Say, we want to filter the
events that have PHA values between 500 and 600.

We can call `hsp.ftselect` like before, but we can also to the call differently by
using `hsp.HSPTask`, and adding the parameters one at a time

```{code-cell} python
# Create a task object
ftselect = hsp.HSPTask("ftselect")

# Pass the input and output files.
ftselect.infile = infile
ftselect.outfile = "tmp.fits"

# Set the selection expression: PHA between 500-600
ftselect.expression = "PHA>500 && PHA<=600"

# We do not want to copy all the file extensions, just the one that is of interest.
ftselect.copyall = False

# We set clobber so the output file is overwritten if it exits.
ftselect.clobber = True
```

Up to this point, the task has not run yet. We now call `ftselect()` to execute it.

```{code-cell} python
result = ftselect()
```

Now we can check the content of the new file with `ftlist`:

```{code-cell} python
result = hsp.ftlist(infile="tmp.fits", option="T")
print(result.stdout)
```

This filtered file contains only PHA values between 500–600!

We'll also clean up after ourselves by deleting the temporary file:

```{code-cell} python
# Now we remove the temporary file
if os.path.exists("tmp.fits"):
    os.remove("tmp.fits")
```

## Example 4: Parameter query control

For some tasks, particularly pipelines (e.g. `ahpipeline`, `nupipeline`, etc.), the
user may wish to run the task without querying all the parameters. They all have
reasonable defaults.

In that case, we can pass the `noprompt=True` when calling the task, and HEASoftPy
will run the task without checking the parameters. For example, to run the first
stage of processing for the NuSTAR observation `60001110002` (data are downloaded in
the 'configuration' cell near the top of the notebook), we can do:

```{code-cell} python
out = hsp.nupipeline(
    indir=nu_data_dir + f"{NU_OBS_ID}/",
    outdir=f"{NU_OBS_ID}_p",
    steminputs=f"nu{NU_OBS_ID}",
    exitstage=1,
    verbose=True,
    noprompt=True,
    clobber=True,
)
```

## Example 5: Running tasks in parallel

Running HEASoftPy tasks in parallel is straightforward using Python libraries such
as [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). The only subtlety is in the use of parameter files. Many
HEASoft tasks use [parameter file](https://heasarc.gsfc.nasa.gov/ftools/others/pfiles.html) to handle the input parameters.

By defaults, parameters are stored in a `pfiles` folder the user's home
directory. When tasks are run in parallel, care is needed to ensure parallel tasks
don't use the same parameter files (and hence be called with the same parameters).

HEASoftPy provides and a content utility that allows tasks to run using temporary
parameter files, so parallel runs remain independent.

The following is an example, we show how to run a `nicerl2` to process NICER event
files from many observations in parallel (the data themselves are downloaded in
the 'configuration' cell near the top of the notebook).

We do this by creating a helper function `worker` that wraps the task call and add
the temporary parameter files (see the 'Global Setup: Functions' section near the
top of this notebook).

The `NUM_CORES` constant (defined in the 'Global Setup: Configuration' subsection
near the top of this notebook) controls the number of processes to run in parallel.

```{danger}
Running the `nicerl2` tool requires that up-to-date geomagnetic data are available on
your system; [see this for a discussion](https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/geomag/).
The path to the geomagnetic data can either be set in the GEOMAG_PATH environment
variable, or passed to the tool directly through the `geomag_path` parameter.
 ```

We download the geomagnetic data using a HEASoft tool; `nigeodown`. Once again we
use the Python interface provided by HEASoftPy.

In this case, we have wrapped the
`nicerl2` HEASoftPy call in another function, to make parallelization easier. Rather
than adding another argument to the `worker` function (defined near the top of this
notebook), and passing the geomagnetic data path, or setting
an environment variable, we define a constant global variable that we read in the `worker` function,

```{code-cell} python
GEOMAG_PATH = os.path.join(ROOT_DATA_DIR, "geomag")
out = hsp.nigeodown(outdir=GEOMAG_PATH, allow_failure=False, clobber=True)
```

```{code-cell} python
os.listdir(GEOMAG_PATH)
```

This geomagnetic data is going to help us process the following NICER observations:

```{code-cell} python
NI_OBS_IDS
```

Now, we can run the parallelized `nicerl2` tasks:

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    obsids = [os.path.join(ni_data_dir, oi) for oi in NI_OBS_IDS]
    result = p.map(worker, obsids)

# Show the output of the parallel tasks
result
```

In this particular case, we've run `nicerl2` in such a way that the outputs are placed in the
original downloaded data directories, overwriting any existing files with newer versions.

We can quickly examine the cleaned events directory of one of the NICER observations:

```{code-cell} python
os.listdir(os.path.join(ni_data_dir, "1012020112", "xti", "event_cl"))
```

## About this Notebook

Author: Abdu Zoghbi, HEASARC Staff Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-03

+++

### Additional Resources

For more documentation on using HEASoft see :

- [HEASoftPy HEASARC page](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy/)
- [HEASoft HEASARC page](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/)
- [HEASoftPy GitHub](https://github.com/HEASARC/heasoftpy)

### Acknowledgements

### References
