---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-09-24'
execution:
  cal-files:
    xmm-ccf: false
    chandra: false
    xspec-models: false
  cache_outputs: [data/test_output.txt]
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
---

# Provider Notebook
This notebook simulates downloading or generating data that a downstream notebook needs.

```{code-cell} python
import os

os.makedirs("data", exist_ok=True)
with open("data/test_output.txt", "w") as f:
    f.write("I'm an idiot")
print("Generated data/test_output.txt")
```
