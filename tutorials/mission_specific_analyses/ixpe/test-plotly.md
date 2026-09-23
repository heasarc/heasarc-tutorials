---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-09-23'
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
title: Testing plotly behaviors
---

# Testing plotly behaviors

## Imports

```{code-cell} python
%pip install plotly
```

```{code-cell} python
import numpy as np
import pandas as pd
import plotly.express as px
```

***

## 1. testing

```{code-cell} python
# 1. Line chart
line_data = pd.DataFrame(
    {
        "date": pd.date_range("2026-01-01", periods=20),
        "value": np.cumsum(np.random.default_rng(42).normal(size=20)),
    }
)

fig = px.line(line_data, x="date", y="value", title="Line Chart")
fig.show()
```

```{code-cell} python
# 2. Scatter plot
scatter_data = px.data.iris()

fig = px.scatter(
    scatter_data,
    x="sepal_width",
    y="sepal_length",
    color="species",
    size="petal_length",
    title="Scatter Plot",
)
fig.show()
```

```{code-cell} python
# 3. Bar chart
bar_data = pd.DataFrame(
    {
        "category": ["A", "B", "C", "D"],
        "value": [12, 19, 8, 15],
    }
)

fig = px.bar(
    bar_data,
    x="category",
    y="value",
    color="category",
    title="Bar Chart",
)
fig.show()
```

```{code-cell} python
# 4. Heatmap
heatmap_data = np.random.default_rng(42).integers(0, 100, size=(5, 5))

fig = px.imshow(
    heatmap_data,
    text_auto=True,
    color_continuous_scale="Viridis",
    title="Heatmap",
)
fig.show()
```

## About this notebook

Author: Kavitha Arur, IXPE GOF Scientist

Author: David J Turner, HEASARC Staff Scientist

Updated On: 2026-06-01

+++

### Additional Resources

Support: [IXPE GOF Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=ixpe)

Documents:
- [IXPE Quick Start Guide](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/ixpe_quickstart.pdf)
- [Recommended practices for statistical treatment of IXPE results](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/IXPE_Stats-Advice.pdf)
- [IXPE support documentation website](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/#supportdoc)

### Acknowledgements


### References
