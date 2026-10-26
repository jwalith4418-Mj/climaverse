# Drop real IMD data here

The pipeline runs on a synthetic stand-in out of the box. For REAL numbers, download
the official products named in the problem statement and place the `.grd` files here:

| File pattern | Product | Source |
|---|---|---|
| `rain*.grd` | Gridded Rainfall 0.25° (129×135) | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html |
| `tmax*.grd` | Max Temperature 1.0° (31×31) | https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html |
| `tmin*.grd` | Min Temperature 1.0° (31×31) | https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html |

INSAT LST/SST/IMR (MOSDAC `3RIMG_L2B_*`) plug in the same way once regridded to the
IMD grid. The loader auto-detects any `<var>*.grd` present and flips `data_is_real`
to true in every report.
