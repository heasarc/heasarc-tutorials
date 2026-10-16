---
execution:
  cal-files:
    xmm-ccf: False
    chandra: False
    xspec-models: False
    cache_outputs:
      - data/test_output.txt
---

# Provider Notebook
This notebook simulates downloading or generating data that a downstream notebook needs.

```{code-cell} python
import os
os.makedirs('data', exist_ok=True)
with open('data/test_output.txt', 'w') as f:
    f.write('State restored successfully!')
print("Generated data/test_output.txt")
```
