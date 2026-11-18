# NDVI–NDMI management zones - mn3392g

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

The included source files support an offline run. `main.py` refreshes the same data preparation and analysis; `eda/01_ndvi_ndmi_clustering.ipynb` reads the prepared gold features and runs the **clustering component** through the same `run_analysis()` function. All plots are saved directly with Matplotlib. Re-running the notebook regenerates the analysis outputs.
