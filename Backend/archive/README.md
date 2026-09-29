# Archived Non-Groundwater NetCDF Files

These files are excluded from the scientific pipeline and are not groundwater salinity predictions.

- `NOT_GROUNDWATER_DATA_regridded_salinity.nc` contains CMIP `so` with `standard_name=sea_water_salinity` and units `0.001`; its global metadata is empty.
- `NOT_GROUNDWATER_DATA_salinity_uncertainty.nc` has scalar `salinity_mean` and `salinity_uncertainty_std` variables carrying the same ocean-water salinity metadata. Its only global attribute is a generic description, so its generation/copy provenance cannot be established from the file.

The available metadata is sufficient to classify these artifacts as ocean salinity data, not groundwater salinity products. Neither is served by the API or used by the projection pipeline. Do not restore them to `data/processed/` without verified provenance and validation.