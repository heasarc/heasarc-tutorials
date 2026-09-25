---
authors:
- name: Tess Jaffe
  affiliations: ['HEASARC, NASA Goddard']
  orcid: 0000-0003-2645-1339
  website: https://science.gsfc.nasa.gov/sci/bio/tess.jaffe
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: Antara Basu-Zych
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  orcid: 0000-0001-8525-4920
  website: https://science.gsfc.nasa.gov/sci/bio/antara.r.basu-zych/
- name: Mike Corcoran
  affiliations: [The Catholic University of America, 'HEASARC, NASA Goddard']
  orcid: 0000-0002-7762-3172
  website: https://science.gsfc.nasa.gov/sci/bio/michael.f.corcoran
date: '2026-09-24'
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
title: "Full Guide 0 – Python for These Notebooks"
---

# Full Guide 0 – Python for These Notebooks

## Learning Goals

By the end of this tutorial, you will be able to:

- Run and modify code in Jupyter notebook cells, and use core Python syntax (variables, f-strings, `for` loops) as it appears throughout the Cas A analysis notebooks.
- Work with NumPy arrays — including slicing, vectorized operations, and boolean masking — the core data structure used for images and tables in these notebooks.
- Use Python dictionaries, the `with` statement for file handling, and `os.path` for building file paths safely and portably.
- Filter `astropy` `Table` objects with boolean logic, and recognize the object/attribute/method pattern used by `numpy`, `astropy`, and other libraries.
- Produce Matplotlib figures, fit a model to data with `scipy.optimize.curve_fit`, and attach physical units to quantities with `astropy.units`.

## Introduction

This notebook is your practical introduction to Python — not Python in the abstract, but specifically the Python you will encounter in the Cassiopeia A data analysis notebooks. Every topic here is something you will use. Every code example is the kind of thing you will need to read, modify, or write yourself for our research project.

Work through the sections in order. Each one builds on the last, and there are further advanced materials below that are provided for reference.

### Contents
1. [Running Code in Jupyter](#1-running-code-in-jupyter)
2. [Imports and Library Namespaces](#2-imports-and-library-namespaces)
3. [Variables, Types, and the ALL_CAPS Convention](#3-variables-types-and-the-all_caps-convention)
4. [f-strings: The Modern Way to Build Messages](#4-f-strings-the-modern-way-to-build-messages)
5. [For Loops Over Real Data](#10-objects-attributes-and-methods)
6. [NumPy Arrays: The Core Data Structure](#6-numpy-arrays-the-core-data-structure)

+++

### Contents: Advanced materials:

+++

7. [Dictionaries](#7-dictionaries)
8. [The `with` Statement and File Handles](#8-the-with-statement-and-file-handles)
9. [File Paths with `os.path`](#5-for-loops-over-real-data)
10. [Objects, Attributes, and Methods](#9-file-paths-with-ospath)
11. [List Comprehensions](#11-list-comprehensions)
12. [Boolean Logic and Filtering Tables](#12-boolean-logic-and-filtering-tables)
13. [Matplotlib: Plotting Images and Data](#13-matplotlib-plotting-images-and-data)
14. [Functions as Arguments: `curve_fit`](#14-fitting-models-to-data-curve_fit)
15.  [Astropy Units and Physical Quantities](#15-astropy-units-and-physical-quantities)


### Quick Reference

This notebook describes in detail each of the aspects of Python and its most important libraries that will be used in our example notebooks.  Here's a quick reference, which really only make sense if you read the appropriate section below.  We summarize them here as a quick reference to remind you.

| Topic | Key syntax |
|---|---|
| Imports | `import numpy as np` or `from astropy.io import fits` |
| f-strings | `f"Value is {variable:.2f}"` |
| File paths | `os.path.join(dir, subdir, file)` |
| For loops | `for row in table:` or `for i, val in enumerate(list):` |
| NumPy arrays | `arr[arr > threshold]` or `arr[:, 0]`, `np.argmax(arr)` |
| Dictionaries | `d = {}` and `d['key'] = value` and `for k, v in d.items()` |
| `with` statement | `with fits.open(path) as hdul: data = hdul[0].data` |
| Objects | `object.attribute` or `object.method(args)` |
| List comprehensions | `[expr for item in iterable if condition]` |
| Table filtering | `table[(table['col'] == val) & (table['col2'] > 0)]` |
| imshow | `ax.imshow(array, origin='lower', cmap='gray', norm=norm)` |
| curve_fit | `popt, pcov = curve_fit(model_func, x, y)` |
| Astropy units | `Quantity(3.4, 'kpc')` and `.to('km/s')` |

### Inputs

- No external data files are required to run this notebook.
- One demonstration FITS image (the Horsehead Nebula) is downloaded automatically, on demand, via `astropy.utils.data.download_file` in Section 8; all other example data (tables, arrays) are generated in-place within the notebook's own code cells.

### Outputs

- A small demonstration text file (`Data/demo_output/demo_data.txt`), used to illustrate the `with` statement.
- One annotated PNG figure (`Data/demo_output/horsehead_annotated.png`), used to illustrate Matplotlib image annotation and figure saving.

### Runtime

_[Author to complete: report actual runtime and machine details, e.g. "As of {Date}, this notebook takes ~{N} seconds to run to completion on Fornax using the '{name: size}' server with N GB RAM / N cores."]_

+++

## Imports

This notebook only relies on the standard scientific Python stack used throughout the Cas A tutorials: `numpy`, `matplotlib`, `astropy`, and `scipy`. No specialized HEASoft or mission-specific packages are required.


The import statements themselves, and what each library is for, are walked through as teaching content in Section 2, "Imports and Library Namespaces", below.

***

## 1. Running Code in Jupyter

A Jupyter notebook is made of **cells**. Text cells (like this one) contain explanations. Code cells contain Python. Results of running a cell can appear inline, i.e., just below the code you can see the plot it made.  This makes them like runnable lab notebooks.

To run a code cell, click on it and press **Shift+Enter**. The output appears directly below.

A number in brackets like `[3]` to the left of a cell tells you it has been run, and in what order. An asterisk `[*]` means it is still running. If you ever need to start fresh, use the menu: **Kernel → Restart & Run All**.

Try running the cell below:

```{code-cell} python
# flake8: noqa: E402
print("Notebook is running correctly.")
```

One important Jupyter convenience: if the **last line** of a cell is a variable name by itself, Jupyter will display its value without needing `print()`:

```{code-cell} python
x = 6.28318
x  # Jupyter displays this automatically because it's the last line
```

This works in Jupyter and iPython only — in a standalone Python script you would need `print(x)`.

+++

## 2. Imports and Library Namespaces

Python on its own is a general-purpose language. The power for science comes from **libraries** — collections of pre-written code maintained by the community. Before you can use a library, you have to **import** it.

The import lines at the top of every science notebook look intimidating at first. Let's break them down.  They are large libraries so this import line takes a noticeable amount of time, so give it a minute.

```{code-cell} python
import astropy.units as u
import matplotlib.pyplot as plt

# The two most important libraries for scientific Python.
# 'as np' and 'as plt' create short aliases — this is universal convention.
import numpy as np

# You can also import a specific part of a library:
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file

# Or import a specific function or class from a submodule:
from scipy.optimize import curve_fit

print("All imports succeeded.")
```

In this context, the dot indicates subpackages within the main package.  The part after the "as" is the alias, which also creates a **namespace** — a prefix that tells Python which library a function belongs to. If you don't, the namespace will be the package imported, e.g., "data" for the astropy data handling package.  Clearly that's a bad idea because different libraries sometimes use the same function name:

```python
np.sqrt(16)      # NumPy's square root — works on entire arrays
math.sqrt(16)    # Python's built-in math — works on single numbers only
```

When you see `np.something(...)` in a notebook, you now know it comes from NumPy. When you see `plt.something(...)`, it comes from Matplotlib. You will see this pattern hundreds of times.

```{code-cell} python
# The namespace prefix tells Python where to find the function
print(np.sqrt(16))  # 4.0  — from numpy
print(np.pi)  # a constant stored inside numpy
print(np.log10(1000))  # 3.0
```

## 3. Variables, Types, and the ALL_CAPS Convention

You already know what a variable is. One thing the science notebooks do that `might look odd: they use ALL_CAPS names for values that should never change during the analysis — things like the source name, or the path to the data directory. This is just a convention (Python doesn't enforce it), but it's a widely used signal meaning "treat this like a constant".

```{code-cell} python
# ALL_CAPS: a signal to the reader that this value won't change
SRC_NAME = "Cas A"
ROOT_DATA_DIR = "Chandra/"
SPEED_OF_LIGHT_KMS = 299792.458  # km/s

# Normal variables: expected to change as the code runs
current_obs_id = "00114"
num_observations = 0
```

Python has several basic data types and assigns them implicitly. Knowing which type you have matters because many functions only accept specific types or behave differently for different types of inputs:

```{code-cell} python
an_integer = 42
a_float = 3.14
a_string = "Cassiopeia A"
a_boolean = True

# type() tells you what kind of thing a variable is
print(type(an_integer))
print(type(a_float))
print(type(a_string))
print(type(a_boolean))
```

A common gotcha: integer division vs. float division. In Python 3, `/` always gives a float, `//` gives an integer (floor division):

```{code-cell} python
print(10 / 3)  # 3.3333...  (float division)
print(
    10 // 3
)  # 3          (floor division — used e.g. to compute number of rows in a grid)
print(10 % 3)  # 1          (modulo — the remainder)
```

Another thing to note is that there are options for math functions, but the `numpy` library is more flexible, operates on arrays, and has a lot of useful math functions.

```{raw-cell}

```

## 4. f-strings: The Modern Way to Build Messages

Throughout the notebooks you will see lines like:

```python
print(f"{len(search_result)} observations selected, with {len(search_result.columns)} columns.")
```

The `f` before the opening quote marks this as an **f-string** (formatted string). Anything inside `{}` is executed as Python and its result is inserted into the string. This is the standard modern way to build informative output.  Note that again Python will implicitly determine the type based on what it gets and apply default formatting rules for that type.  But the third example shows you how to control the formatting yourself when you need to.  This is discussed more in the next cell.

+++

## 5. For Loops Over Real Data

The science notebooks use `for` loops in several patterns that go beyond the basic `for i in range(10)`. Here are the ones you'll encounter.

```{code-cell} python
# Pattern 1: Loop over a list of items directly
obs_ids = ["00114", "01952", "04634"]

for obs_id in obs_ids:
    print(f"Processing observation {obs_id}")
```

```{code-cell} python
# Pattern 2: enumerate() — gives you both the index AND the value
# Used when you need to know the position as well as the item
for i, obs_id in enumerate(obs_ids):
    print(f"  [{i}] ObsID: {obs_id}")
```

```{code-cell} python
# Pattern 3: Loop over table rows — each 'row' is like a mini-dictionary
obs_table = Table(
    {
        "obsid": ["00114", "01952", "04634", "09117"],
        "time": [51413.2, 52010.5, 53186.7, 55008.1],  # MJD
        "exposure": [49400, 33000, 98000, 46000],  # seconds
        "detector": ["ACIS-S", "ACIS-S", "ACIS-S", "ACIS-I"],
    }
)
for row in obs_table:
    print(f"  ObsID {row['obsid']}: {row['exposure']/1000:.0f} ks on {row['detector']}")
```

```{code-cell} python
# Pattern 4: Building up a result inside a loop
# This is how the notebooks accumulate results across many observations
total_exposure = 0
processed_ids = []

for row in obs_table:
    total_exposure += row["exposure"]
    processed_ids.append(row["obsid"])

print(
    f"Total exposure across {len(processed_ids)} observations: {total_exposure/1000:.0f} ks"
)
```

```{code-cell} python
source_name = "Cas A"
num_obs = 26
exposure_ks = 49.4

# Old way — hard to read, easy to get wrong:
print("Source: " + source_name + ", observations: " + str(num_obs))

# Modern f-string — the expression inside {} is evaluated inline:
print(f"Source: {source_name}, observations: {num_obs}")

# You can put any Python expression inside the braces:
print(f"Total exposure: {exposure_ks:.1f} ks  ({exposure_ks * 1000:.0f} seconds)")
```

The `:1f` and `:.0f` inside the braces are **format specifiers** — they control how numbers are displayed. `.1f` means 'one decimal place, float format'. You don't need to memorize these; just recognize them when you see them.

```{code-cell} python
# Format specifiers in action:
pi = 3.14159265
print(f"Default:      {pi}")
print(f"2 decimals:   {pi:.2f}")
print(f"Scientific:   {pi:.3e}")
print(f"As integer:   {pi:.0f}")
```

A much more complicated type is objects used in "object-oriented" programming.  Many of the packages we use have this, though we are not using them this way.  So this note is just informational.  Such a variable is an "object" that can contain its own internal variables and functions.  To access these internal components is the other major context in which the dot is used.

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# A simple class definition that takes two inputs, the second of which is optional.
# When an object of this type is initialized, it computes a value that it stores
# internally. It also defines a function that can be called based on that object.
# This can get very complicated, but seeing these examples will let you recognize
# them in future.


#  A class:
class my_object:
    #  Its constructor
    def __init__(self, input1, input2=1):
        # actions to save the inputs and compute a new variable
        self.input1 = input1
        self.input2 = input2
        self.simple_variable = input1 * input2
        print(
            f"The input {self.input1} and {self.input2} multiply into {self.simple_variable}"
        )

    #  A method of the class
    def simple_function(self):
        print(
            f"The sqrt of the input {self.input1}*{self.input2}={self.simple_variable} is {np.sqrt(self.simple_variable):.4f}"
        )


anObject = my_object(3, 4)
print(anObject.input2)
anObject.simple_function()
```

## 6. NumPy Arrays: The Core Data Structure

Every Chandra image is a 2D NumPy array. Every table column of times, exposure values, and fluxes is a 1D NumPy array. NumPy arrays are the single most important data structure in these notebooks, and they behave differently from Python lists in ways that will surprise you if you're not prepared.

### 6a. Creating arrays

```{code-cell} python
import numpy as np

# From a Python list:
a = np.array([1, 4, 9, 16, 25])
print(a)

# np.arange: like Python's range(), but returns an array
b = np.arange(0, 10, 2)  # start, stop, step
print(b)

# np.linspace: N evenly spaced points between start and stop (inclusive)
c = np.linspace(0, 1, 6)
print(c)

# np.zeros, np.ones: useful for pre-allocating the array and filling it with a value
print(np.zeros(4))
print(np.ones((2, 3)))  # 2D: 2 rows, 3 columns
```

### 6b. Element-wise operations: the key difference from Python lists

With a Python list, `my_list * 2` doubles the list (repeats it). With a NumPy array, `my_array * 2` multiplies **every element** by 2. This is called **vectorization**, and it's what makes NumPy fast and concise.

```{code-cell} python
python_list = [1, 2, 3, 4]
numpy_array = np.array([1, 2, 3, 4])

print("List * 2:  ", python_list * 2)  # [1,2,3,4,1,2,3,4]  — repeats the list!
print("Array * 2: ", numpy_array * 2)  # [2,4,6,8]           — multiplies each element

# All arithmetic operations work element-wise:
exposure_times = np.array([49400, 33000, 67000, 28000])  # seconds
exposure_ks = exposure_times / 1000  # convert to kiloseconds
print("Exposures in ks:", exposure_ks)

# Two arrays can be combined element-wise:
counts = np.array([1200, 800, 2100, 560])
rate = counts / exposure_times  # count rate for each observation
print("Count rates:    ", rate)
```

### 6c. Slicing: 1D and 2D

A slice is the term for specifying a subset of an array.  The slice syntax `[start:stop]` works just like on Python lists, but extends naturally to multiple dimensions. This is used constantly when working with image data.

```{code-cell} python
# 1D slicing
times = np.linspace(1999, 2025, 27)  # simulated observation years
print("First 5:  ", times[:5])
print("Last 3:   ", times[-3:])
print("Every 4th:", times[::4])

# 2D slicing — think of it as [rows, columns]
# This is exactly how Chandra image cutouts work:
img_data = np.array([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8]])

print("\nFull image shape:", img_data.shape, "with values:")
print(img_data)
print("Top-left 2x3 corner img_data[:2, :3] (first 2 rows, first 3 columns):")
print(img_data[:2, :3])  #

print("\nCenter pixel region:")
print(img_data[1:3, 1:3])
```

 ---
# Advanced materials

All materials below are hidden, though you can see them in the table of contents (the icon on the far left side bar with three dot-dashes).  They are provided as a reference.

## 6+. NumPy Arrays: The Core Data Structure

### 6d+. Advanced:  Array attributes and common functions

```{code-cell} python
print("Shape:          ", img_data.shape)  # (rows, columns)
print("Total elements: ", img_data.size)
print("Data type:      ", img_data.dtype)
print("Max value:      ", img_data.max())
print("Index of max:   ", np.argmax(img_data))  # index in the flattened array
print("Mean:           ", img_data.mean())
print("95th percentile:", np.percentile(img_data, 95))
```

### 6e+. Advanced:  Boolean masking: filtering data without a for loop

This is one of the most important NumPy skills. Instead of looping through an array and checking each element, you compare the whole array to a condition, which produces a **boolean array** (True/False for each element). You then use that boolean array to index into the original array, keeping only the matching elements.

In the science notebooks this is used to filter observation tables, select pixels above a threshold, and much more.

```{code-cell} python
# Simulated exposure times for a set of observations (in seconds)
exposures = np.array([49400, 8000, 67000, 3200, 158000, 22000, 98000])

# Step 1: A comparison produces a boolean array
long_exposure_mask = exposures > 30000
print("Boolean mask:", long_exposure_mask)

# Step 2: Use that mask to index the original array
long_exposures = exposures[long_exposure_mask]
print("Long exposures:", long_exposures)

# You can also do it in one line:
print("Short exposures:", exposures[exposures < 10000])
```

```{code-cell} python
# Combining conditions with & (and) and | (or)
# Note: use & and |, NOT 'and' and 'or' — and each condition should be in parentheses
# to avoid confusion and unexpected results

medium_exposures = exposures[(exposures > 10000) & (exposures < 100000)]
print("Medium exposures (10-100 ks):", medium_exposures)

# The ~ operator inverts a boolean array
obs_ids = np.array(["00114", "06690", "13783", "01952", "16946"])
to_exclude = ["06690", "13783", "16946"]

keep_mask = ~np.isin(obs_ids, to_exclude)
print("Keeping:", obs_ids[keep_mask])
```

Note:  the second to last line uses np.isin(), a comparison of two arrays, producing the boolean array mask where the first argument's values are somewhere in the second argument somewhere.  In front of that is the tilde '~' symbol, which is the way you specify a NOT, i.e., the mask where True values mark the obs_ids to NOT exclude.

This applies only to operations on numpy arrays. Elsewhere in python the logical operator is NOT itself.

+++

### 6f+. Advanced:  Useful string operations on arrays

The `np.char` submodule provides vectorized string operations — the same idea as element-wise arithmetic, but for strings. This is used in the notebooks to search column values and reformat observation IDs.

```{code-cell} python
detectors = np.array(["ACIS-S", "HRC-I", "ACIS-I", "HRC-S", "ACIS-S"])

# np.char.find returns the position of a substring in each element, or -1 if not found
find_result = np.char.find(detectors, "ACIS")
print("Find result:", find_result)

# Use != -1 to make a boolean mask for 'contains ACIS'
is_acis = np.char.find(detectors, "ACIS") != -1
print("Is ACIS:    ", is_acis)
print("ACIS only:  ", detectors[is_acis])

# np.char.zfill pads strings with leading zeros to a given width
obs_ids_short = np.array(["114", "1952", "13783"]).astype(str)
obs_ids_padded = np.char.zfill(obs_ids_short, 5)
print("Padded IDs: ", obs_ids_padded)
```

## 7. Dictionaries

A Python dictionary stores **key-value pairs**. You look up a value by its key, just like looking up a word in a dictionary. In the notebooks, dictionaries are used heavily to store all loaded image data, indexed by observation ID — a natural fit since each observation ID uniquely identifies one dataset.

```{code-cell} python
# Create a dictionary with curly braces
# Keys are strings (or other immutable types); values can be anything
obs_info = {
    "obsid": "00114",
    "target": "Cas A",
    "exposure": 49400,  # seconds
    "detector": "ACIS-S",
    "archived": True,
}

# Access a value by key
print(obs_info["obsid"])
print(obs_info["exposure"])
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Dictionaries are mutable — you can add and update entries
obs_info["pi_name"] = "Hwang"
obs_info["exposure"] = 49411  # update an existing value
print(obs_info)
```

```{code-cell} python
# Nested dictionaries: this is exactly the pattern used in the notebooks
# to hold all image data, indexed by ObsID
all_loaded_imgs = {}

# Simulating what the notebooks do when loading image files:
for obs_id, exposure in [("00114", 49400), ("01952", 33000), ("04634", 98000)]:
    all_loaded_imgs[obs_id] = {
        "data": np.zeros((100, 100)),  # placeholder for the actual image array
        "exp": exposure,
        "count_rate": False,
    }

# Access nested values:
print(all_loaded_imgs["01952"]["exp"])
print(all_loaded_imgs["04634"]["count_rate"])
```

```{code-cell} python
# Iterating over a dictionary
print("Keys:  ", list(all_loaded_imgs.keys()))
print("---")

# .items() gives you (key, value) pairs — very useful in for loops
for obs_id, info in all_loaded_imgs.items():
    print(f"  ObsID {obs_id}: exposure = {info['exp']} s")
```

## 8. The `with` Statement and File Handles

In the science notebooks, every FITS file is opened using a `with` block:

```python
with fits.open(demo_img_path) as imgo:
    demo_img_arr = imgo['PRIMARY'].data
    demo_img_hdr = imgo['PRIMARY'].header
```

You saw a simpler version of this in the original intro notebook with `open('file.txt')`. The `with` statement is a **context manager** — it guarantees that no matter what happens (even if your code crashes), the file will be properly closed when the indented block exits.

The critical rule: **any data you need from the file must be extracted inside the `with` block and stored in a variable.** Once you leave the block, the file is closed and the handle `imgo` is no longer usable.

```{code-cell} python
# Demonstration with a plain text file — same pattern as FITS
import os

# First, write a small file to read back
with open("Data/demo_output/demo_data.txt", "w") as outfile:
    outfile.write("ObsID  Exposure  Detector\n")
    outfile.write("00114  49400     ACIS-S\n")
    outfile.write("01952  33000     ACIS-S\n")

# Now read it back — the file is automatically closed when the 'with' block ends
with open("Data/demo_output/demo_data.txt", "r") as infile:
    file_contents = infile.read()  # <-- stored in a variable INSIDE the block

# We can use file_contents here because we saved it to a variable
print(file_contents)
```

```{code-cell} python
# The pattern extends naturally to FITS files.
# Here is the actual pattern you will see in the notebooks:

# Download a real FITS image of the Horsehead Nebula (cached after first download)
horsehead_path = download_file(
    "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits", cache=True
)

with fits.open(horsehead_path) as hdul:
    print(hdul.info())  # shows the structure of the FITS file
    img_array = hdul[0].data  # 2D numpy array — extracted inside the block
    img_header = hdul[0].header  # header — also extracted inside the block

# Outside the 'with' block: the file is closed, but our variables are still available
print(f"\nImage shape: {img_array.shape}")
print(f"Data type:   {img_array.dtype}")
```

```{code-cell} python
# Headers behave like dictionaries — access values by keyword name
print(f"Object:    {img_header['OBJECT']}")
print(f"Telescope: {img_header['TELESCOP']}")

# You can look at the first N header cards like a slice:
from pprint import pprint

pprint(list(img_header.items())[:8])  # first 8 header key-value pairs
```

 ---

## 9. File Paths with `os.path`

The science notebooks organize downloaded data into a directory structure, and they build file paths in code rather than typing them out by hand. This is done using Python's built-in `os` module. The key reason: paths look different on Windows (`C:\Users\...`) vs. Mac/Linux (`/home/...`). Using `os.path.join()` makes your code work on any system.

```{code-cell} python
import os

# os.path.join() builds a path from parts — handles slashes automatically
data_dir = "Data/Chandra/individual_files"
obs_id = "00114"
file_name = "acisf00114N006_cntr_img2.fits.gz"

full_path = os.path.join(data_dir, obs_id, file_name)
print(full_path)
```

```{code-cell} python
# os.path.abspath() converts a relative path to a full absolute path
abs_path = os.path.abspath(data_dir)
print(abs_path)

# os.path.exists() checks whether a path actually exists on disk
print(os.path.exists(data_dir))

# os.makedirs() creates a directory (and any missing parent directories)
# exist_ok=True means: don't raise an error if it already exists
os.makedirs("Data/demo_output", exist_ok=True)
print(f"demo_output exists: {os.path.exists('Data/demo_output')}")
```

```{code-cell} python
# os.scandir() lists the contents of a directory as an iterator of entries.
# Each entry has a .name and methods like .is_file() and .is_dir().
# Here we list files in the current directory:
current_dir_files = [entry.name for entry in os.scandir(".") if entry.is_file()]
print(current_dir_files[:10])
```

```{code-cell} python
# A NumPy array is an object.
arr = np.array([5, 3, 8, 1, 9, 2, 7])

# Attributes — accessed with a dot, no parentheses:
print("arr.shape:", arr.shape)
print("arr.dtype:", arr.dtype)
print("arr.size: ", arr.size)

# Methods — accessed with a dot AND parentheses (they are function calls):
print("arr.max():  ", arr.max())
print("arr.mean():  ", arr.mean())
print("arr.sum():   ", arr.sum())
```

```{code-cell} python
# astropy's SkyCoord is a more complex object — a real one from the notebooks.
from astropy.coordinates import SkyCoord

# Instantiate (create) a SkyCoord object by calling the class with arguments
casa_coord = SkyCoord(350.8584, 58.8113, unit="deg")

# Attributes:
print("RA: ", casa_coord.ra)
print("Dec:", casa_coord.dec)

# Methods:
galactic = casa_coord.galactic  # converts to Galactic coordinates
print("Galactic l:", galactic.l)
print("Galactic b:", galactic.b)
```

```{code-cell} python
# astropy Table — used throughout the notebooks to hold observation metadata.
# It behaves like a spreadsheet in Python.
from astropy.table import Table

obs_table = Table(
    {
        "obsid": ["00114", "01952", "04634", "09117"],
        "time": [51413.2, 52010.5, 53186.7, 55008.1],  # MJD
        "exposure": [49400, 33000, 98000, 46000],  # seconds
        "detector": ["ACIS-S", "ACIS-S", "ACIS-S", "ACIS-I"],
    }
)

obs_table
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Attributes and methods of the Table object:
print("Column names:", obs_table.colnames)
print("Number of rows:", len(obs_table))

# .sort() modifies the table in-place (no return value!)
obs_table.sort("exposure")
obs_table
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Accessing columns — like a dictionary, using the column name as key:
print(obs_table["obsid"])
print()

# Accessing rows — using an integer index:
print(obs_table[0])  # first row (after sorting)
print()

# Accessing a specific cell:
print(obs_table["exposure"][2])
```

 ---

## 10. Objects, Attributes, and Methods

Most of the interesting things in the science notebooks are **objects** — bundles of data and functions that belong together. You've already used objects without necessarily thinking about it: a NumPy array is an object, and so is a FITS header.

The two key ideas:
- An **attribute** is data stored inside an object. You access it with a dot: `array.shape`, `result.data`.
- A **method** is a function stored inside an object. You call it with a dot and parentheses: `table.sort('time')`, `header.keys()`.

You do **not** need to write your own objects for these notebooks. But you need to be comfortable **using** them.

+++

## 11. List Comprehensions

A list comprehension is a compact way to write a `for` loop that builds a list. You will see them constantly in the notebooks. They are more readable (once you're used to them) and slightly faster than an equivalent loop.

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# The loop version and the comprehension version do the same thing:

# Long way — a for loop that builds a list
ks_values_loop = []
for row in obs_table:
    ks_values_loop.append(row["exposure"] / 1000)

# Short way — a list comprehension
ks_values_comp = [row["exposure"] / 1000 for row in obs_table]

print(ks_values_loop)
print(ks_values_comp)
```

```{code-cell} python
# With an 'if' filter — only include items that pass a condition
# This is used in the notebooks to list only files (not directories):

# All entries in the demo_output directory:
all_entries = [entry.name for entry in os.scandir("Data/demo_output")]

# Only files:
only_files = [entry.name for entry in os.scandir("Data/demo_output") if entry.is_file()]

print("All entries:", all_entries)
print("Files only: ", only_files)
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# A real example from notebook 2:
# Get the maximum count rate from each image stored in all_loaded_imgs
max_rate_vals = [info["data"].max() for info in all_loaded_imgs.values()]
print("Max rates:", max_rate_vals)

# Then use numpy on the result:
upper_limit = np.percentile(max_rate_vals, 95)
print(f"95th percentile upper limit: {upper_limit}")
```

## 12. Boolean Logic and Filtering Tables

The first notebook is essentially a sequence of table filtering steps. Each one applies a condition to select a subset of rows. This section shows you the full toolkit.

The core pattern is always:
```python
filtered_table = original_table[original_table['column'] == 'value']
```

```{code-cell} python
# Rebuild the table with more rows for this demonstration
obs_table = Table(
    {
        "obsid": ["00114", "01952", "04634", "09117", "06690", "13783"],
        "time": [51413.2, 52010.5, 53186.7, 55008.1, 53900.0, 56200.3],
        "exposure": [49400, 33000, 98000, 46000, 68000, 55000],
        "detector": ["ACIS-S", "ACIS-S", "ACIS-S", "ACIS-I", "ACIS-S", "ACIS-S"],
        "status": [
            "archived",
            "archived",
            "archived",
            "archived",
            "archived",
            "scheduled",
        ],
        "grating": ["NONE", "NONE", "HETG", "NONE", "NONE", "NONE"],
    }
)

print(f"Starting with {len(obs_table)} observations.")
obs_table
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Filter 1: keep only archived observations
archived = obs_table[obs_table["status"] == "archived"]
print(f"After status filter: {len(archived)} observations")

# Filter 2: keep only those with no grating deployed
no_grating = archived[archived["grating"] == "NONE"]
print(f"After grating filter: {len(no_grating)} observations")
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Apply multiple filters at once using & (and)
# This is the preferred pattern shown in the notebook
selected = obs_table[
    (obs_table["status"] == "archived") & (obs_table["grating"] == "NONE")
]
print(f"Combined filter result: {len(selected)} observations")
selected
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Excluding specific observations by ID using np.isin and ~
# ~ is the 'not' operator for boolean arrays
to_exclude = ["06690", "13783"]

cleaned = selected[~np.isin(selected["obsid"], to_exclude)]
print(f"After exclusion: {len(cleaned)} observations")
cleaned
```

## 13. Matplotlib: Plotting Images and Data

The science notebooks use Matplotlib extensively in two ways: plotting data (histograms, scatter plots, line plots) and displaying 2D images. Both use the **object-oriented** `fig, ax` style, which gives you finer control than the simpler `plt.plot()` style.

We'll use the real Horsehead Nebula FITS image we already downloaded.

### 13a. The `fig, ax` pattern

```{code-cell} python
---
jupyter:
  source_hidden: true
---
import matplotlib.pyplot as plt

# Simulated data: observation times (years) and a measured radius (arcsec)
years = np.array([1999, 2000, 2002, 2004, 2007, 2009, 2011, 2013, 2018, 2023])
radii = np.array(
    [153, 158, 163, 170, 179, 185, 191, 197, 212, 229]
) + np.random.default_rng(42).normal(
    0, 2, 10
)  # small scatter

# fig, ax = plt.subplots() is the standard way to create a figure.
# fig is the whole figure canvas; ax is the single plot area (axes) inside it.
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(years, radii, "o-", color="steelblue", label="Measured radius")

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Radius [arcsec]", fontsize=13)
ax.set_title("Cas A Expansion", fontsize=14)
ax.legend(fontsize=12)
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

plt.tight_layout()
plt.show()
```

### 13b. Multiple subplots

The side-by-side Chandra image comparison in notebook 2 uses `plt.subplots(nrows, ncols)` to create a grid of plot panels.

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Create a 1-row, 2-column figure
fig, ax_arr = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# ax_arr is now a list of two Axes objects: ax_arr[0] and ax_arr[1]
ax_arr[0].plot(years, radii, "o", color="steelblue")
ax_arr[0].set_title("Linear scale")
ax_arr[0].set_xlabel("Year")
ax_arr[0].set_ylabel("Radius [arcsec]")

ax_arr[1].plot(years - years[0], radii - radii[0], "s", color="darkorange")
ax_arr[1].set_title("Change since first observation")
ax_arr[1].set_xlabel("Years since 1999")
ax_arr[1].set_ylabel("ΔRadius [arcsec]")

plt.tight_layout()
plt.show()
```

### 13c. Displaying a 2D image with `imshow`

This is the core visualization for X-ray astronomy. `plt.imshow()` treats a 2D NumPy array as an image, mapping each value to a color.

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Display the Horsehead Nebula FITS image we loaded earlier
fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(
    img_array,
    origin="lower",  # (0,0) at bottom-left, matching astronomical convention
    cmap="gray",  # colormap — 'gnuplot2' is used for X-ray in the notebooks
    vmin=3000,  # minimum value mapped to the darkest color
    vmax=15000,
)  # maximum value mapped to the brightest color

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Pixel value [counts]", fontsize=12)

ax.set_title("Horsehead Nebula (photographic plate scan)", fontsize=13)
ax.axis("off")  # hide the pixel-coordinate tick marks

plt.tight_layout()
plt.show()
```

### 13d. Logarithmic image stretch

X-ray images have a huge dynamic range — the brightest features can be thousands of times brighter than the faintest ones. Displaying with a linear color scale washes out the faint structure. A **logarithmic stretch** is standard practice.

```{code-cell} python
---
jupyter:
  source_hidden: true
---
from astropy.visualization import ImageNormalize, LogStretch, ManualInterval

# ImageNormalize combines an 'interval' (which values map to 0-1)
# and a 'stretch' (how values are distributed between 0 and 1)
norm = ImageNormalize(
    interval=ManualInterval(vmin=3000, vmax=15000), stretch=LogStretch()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: linear scale
im0 = axes[0].imshow(img_array, origin="lower", cmap="gray", vmin=3000, vmax=15000)
axes[0].set_title("Linear stretch", fontsize=13)
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0])

# Right: logarithmic stretch via astropy ImageNormalize
im1 = axes[1].imshow(img_array, origin="lower", cmap="gray", norm=norm)
axes[1].set_title("Log stretch (reveals faint structure)", fontsize=13)
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
```

### 13e. Annotating images and saving figures

```{code-cell} python
from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img_array, origin="lower", cmap="gray", norm=norm)
ax.axis("off")

# ax.text() places text on the image.
# transform=ax.transAxes means coordinates are 0-1 relative to the plot box,
# not in data (pixel) units — so the text position won't shift if the data changes.
ax.text(
    0.05,
    0.93,
    "Horsehead Nebula",
    transform=ax.transAxes,
    color="white",
    fontsize=14,
    fontweight="bold",
)

# A circle patch — this is exactly how the expansion circle is drawn on Cas A images
circle = Circle(
    (445, 330),
    radius=120,
    edgecolor="cyan",
    facecolor="none",
    linewidth=2,
    linestyle="dashed",
)
ax.add_patch(circle)

# Save to disk before calling plt.show()
fig.savefig("Data/demo_output/horsehead_annotated.png", bbox_inches="tight", dpi=100)
plt.show()

print("Figure saved.")
```

## 14. Fitting models to data: `curve_fit`

In notebook 3, you will fit a model to data to measure how fast Cas A is expanding. The fitting tool is `scipy.optimize.curve_fit`. Its key feature — and something new Python users find surprising — is that **it takes a function as one of its arguments**.

First, let's understand what that means, and then we'll see `curve_fit` in action.

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Functions are objects in Python — you can assign them to variables
# and pass them as arguments to other functions.


def square(x):
    return x**2


def apply_twice(func, value):
    """Applies func to value, then applies it again to the result."""
    return func(func(value))


# Pass the 'square' function as an argument:
result = apply_twice(square, 3)  # square(square(3)) = square(9) = 81
print(result)
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
from scipy.optimize import curve_fit

# Step 1: Define your model function.
# The first argument MUST be x (the independent variable).
# All subsequent arguments are the free parameters to be fitted.


def straight_line(x, m, c):
    """A straight line: y = m*x + c"""
    return m * x + c


# Step 2: Create some noisy data to fit.
# True values: slope m=3.2 arcsec/year, intercept c=153 arcsec
rng = np.random.default_rng(42)
true_m = 3.2
true_c = 153.0
x_data = np.array(
    [1999, 2000, 2002, 2004, 2007, 2009, 2011, 2013, 2018, 2023], dtype=float
)
x_data_norm = x_data - x_data[0]  # years since first observation
y_data = true_m * x_data_norm + true_c + rng.normal(0, 2, len(x_data))

# Step 3: Call curve_fit.
# Arguments: (model_function, x_data, y_data)
# Returns: (best_fit_parameters, covariance_matrix)
popt, pcov = curve_fit(straight_line, x_data_norm, y_data)

# popt contains the best-fit values for [m, c]
fitted_m, fitted_c = popt
print(f"True slope:   {true_m:.2f} arcsec/year")
print(f"Fitted slope: {fitted_m:.2f} arcsec/year")
print(f"True intercept:   {true_c:.1f} arcsec")
print(f"Fitted intercept: {fitted_c:.1f} arcsec")
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Uncertainties on the fitted parameters come from the covariance matrix.
# The diagonal of pcov contains the variance (sigma^2) for each parameter.
param_errors = np.sqrt(np.diag(pcov))
print(f"Uncertainty on slope:     {param_errors[0]:.3f} arcsec/year")
print(f"Uncertainty on intercept: {param_errors[1]:.2f} arcsec")
```

```{code-cell} python
---
jupyter:
  source_hidden: true
---
# Visualize the fit
x_smooth = np.linspace(x_data_norm.min(), x_data_norm.max(), 100)
y_fit_line = straight_line(x_smooth, fitted_m, fitted_c)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(x_data_norm, y_data, "o", color="steelblue", label="Data", zorder=3)
ax.plot(
    x_smooth,
    y_fit_line,
    "-",
    color="tomato",
    linewidth=2,
    label=f"Fit: slope = {fitted_m:.2f} ± {param_errors[0]:.2f} arcsec/yr",
)

ax.set_xlabel("Years since first observation", fontsize=13)
ax.set_ylabel("Radius [arcsec]", fontsize=13)
ax.set_title("Fitting the expansion of Cas A", fontsize=14)
ax.legend(fontsize=12)
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

plt.tight_layout()
plt.show()
```

 ---

## 15. Astropy Units and Physical Quantities

Notebook 3 asks you to calculate the expansion velocity of Cas A in km/s and compare it to the speed of light. `astropy.units` makes this natural — you attach units to numbers, and `astropy` handles the conversions automatically, warning you if you accidentally combine incompatible units.

```{code-cell} python
import astropy.units as u
from astropy.units import Quantity

# Create a Quantity by multiplying a number by a unit
exposure = Quantity(49400, "s")
print(exposure)

# Convert to different units with .to()
print(exposure.to("hour"))
print(exposure.to("ks"))  # kiloseconds
```

```{code-cell} python
# Arithmetic preserves units
distance = Quantity(3.4, "kpc")  # distance to Cas A
ang_speed = Quantity(0.23, "arcsec / year")

# The small angle approximation: proper_speed = angular_speed * distance
#  (Equivalencies help set the context, here hypothetically for angles in rad or deg.)
proper_speed = (ang_speed * distance).to("km/s", equivalencies=u.dimensionless_angles())
print(f"Expansion speed: {proper_speed:.0f}")
```

```{code-cell} python
# astropy.constants gives you fundamental physical constants with units
from astropy.constants import c as speed_of_light

print(speed_of_light)  # in m/s by default
print(speed_of_light.to("km/s"))  # convert to km/s

# What fraction of the speed of light is Cas A expanding at?
fraction = (proper_speed / speed_of_light.to("km/s")).decompose()
print(f"\nCas A expansion: {fraction.value*100:.2f}% of the speed of light")
```

```{code-cell} python
# Quantities also work naturally with NumPy arrays
from astropy.time import Time

# The Time class handles astronomical time formats
# MJD (Modified Julian Date) is used throughout the notebooks
obs_times_mjd = np.array([51413.2, 52010.5, 53186.7, 55008.1])

time_obj = Time(obs_times_mjd, format="mjd")

# Convert to calendar year:
print(time_obj.decimalyear)

# Time differences:
time_since_first = Quantity((time_obj - time_obj[0]).to_value("day"), "day")
print(time_since_first.to("year"))
```

```{code-cell} python

```

# You're Ready

You've now seen — and run — every major Python pattern used in the Cas A science notebooks.

Open `1-Worksheet-GetStarted.ipynb` to begin.

```{code-cell} python

```

## About this notebook

Author: Tess Jaffe, HEASARC Chief Archive Scientist.

Author: David J Turner, HEASARC Staff Scientist.

Author: Antara Basu-Zych, HEASARC Archive Scientist.

Author: Mike Corcoran, Associate Research Professor.

Updated On: 2026-09-24

+++

### Additional Resources

**HEASARC Help Desk**: [https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

_[Author to add any other resources, including mission-specific helpdesks.]_

### Acknowledgements

_[Author to complete.]_

### References

_[Author to complete.]_
