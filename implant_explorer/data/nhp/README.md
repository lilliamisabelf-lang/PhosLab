# Data Folder

This folder contains all input data required by the SPIKE Implant Explorer and pRF Neural Network Prediction tools.

## Required Files

Copy the following files from `D:\GoogleDrive\SPIKE\NIN_storage_SPIKE\SPIKE\` to this folder:

### 1. Brain Atlas (`data/nhp/atlas/`)

| Source | Destination |
|--------|-------------|
| `Spike_Chris/Atlas/D99_in_Spike_iso.nii.gz` | `atlas/D99_in_Spike_iso.nii.gz` |

### 2. pRF Maps - Danny (`data/nhp/prf_maps/danny/`)

| Source | Destination |
|--------|-------------|
| `Spike_Chris/pRF_maps/inSpike/ECC_Danny_inSpike.nii.gz` | `prf_maps/danny/ECC_Danny_inSpike.nii.gz` |
| `Spike_Chris/pRF_maps/inSpike/POL_Danny_inSpike.nii.gz` | `prf_maps/danny/POL_Danny_inSpike.nii.gz` |
| `Spike_Chris/pRF_maps/inSpike/R2_Danny_inSpike.nii.gz` | `prf_maps/danny/R2_Danny_inSpike.nii.gz` |

### 3. pRF Maps - Eddy (`data/nhp/prf_maps/eddy/`)

| Source | Destination |
|--------|-------------|
| `Spike_Chris/pRF_maps/inSpike/ECC_Eddy_inSpike.nii.gz` | `prf_maps/eddy/ECC_Eddy_inSpike.nii.gz` |
| `Spike_Chris/pRF_maps/inSpike/POL_Eddy_inSpike.nii.gz` | `prf_maps/eddy/POL_Eddy_inSpike.nii.gz` |
| `Spike_Chris/pRF_maps/inSpike/R2_Eddy_inSpike.nii.gz` | `prf_maps/eddy/R2_Eddy_inSpike.nii.gz` |

### 4. Fiducials (`data/nhp/fiducials/`)

| Source | Destination |
|--------|-------------|
| `10-2022-SLICER_SPIKE_IMPLANT_DESIGN/1.fcsv` | `fiducials/1.fcsv` |
| `10-2022-SLICER_SPIKE_IMPLANT_DESIGN/2.fcsv` | `fiducials/2.fcsv` |
| ... | ... |
| `10-2022-SLICER_SPIKE_IMPLANT_DESIGN/16.fcsv` | `fiducials/16.fcsv` |

## Quick Copy Commands (PowerShell)

```powershell
$src = "<path-to-SPIKE-source-data>"
$dst = "<repo-root>\data\nhp"

# Atlas
Copy-Item "$src\Spike_Chris\Atlas\D99_in_Spike_iso.nii.gz" "$dst\atlas\"

# pRF Maps - Danny
Copy-Item "$src\Spike_Chris\pRF_maps\inSpike\ECC_Danny_inSpike.nii.gz" "$dst\prf_maps\danny\"
Copy-Item "$src\Spike_Chris\pRF_maps\inSpike\POL_Danny_inSpike.nii.gz" "$dst\prf_maps\danny\"
Copy-Item "$src\Spike_Chris\pRF_maps\inSpike\R2_Danny_inSpike.nii.gz" "$dst\prf_maps\danny\"

# pRF Maps - Eddy
Copy-Item "$src\Spike_Chris\pRF_maps\inSpike\ECC_Eddy_inSpike.nii.gz" "$dst\prf_maps\eddy\"
Copy-Item "$src\Spike_Chris\pRF_maps\inSpike\POL_Eddy_inSpike.nii.gz" "$dst\prf_maps\eddy\"
Copy-Item "$src\Spike_Chris\pRF_maps\inSpike\R2_Eddy_inSpike.nii.gz" "$dst\prf_maps\eddy\"

# Fiducials
Copy-Item "$src\10-2022-SLICER_SPIKE_IMPLANT_DESIGN\*.fcsv" "$dst\fiducials\"
```

## File Descriptions

- **D99_in_Spike_iso.nii.gz**: D99 macaque brain atlas warped to subject (Spike) space, isotropic resolution
- **ECC_*_inSpike.nii.gz**: Eccentricity maps from pRF fitting (degrees of visual angle)
- **POL_*_inSpike.nii.gz**: Polar angle maps from pRF fitting (degrees)
- **R2_*_inSpike.nii.gz**: R² goodness-of-fit maps from pRF model
- **\*.fcsv**: Slicer3D fiducial files containing electrode comb entry/exit coordinates

## Notes

- These files are not tracked in git due to their size
- Total size: ~100-300 MB
- Contact project maintainers if you need access to the original data
