# Wind turbine power system simulations

Simulations of the **electrical network** (power flow, buses, lines, UIC converter, time-domain integration) run in [TOPS](https://arxiv.org/abs/2101.02937). This repository adds the **wind turbine models** and case studies: a baseline wind turbine model fully within TOPS, and a coupled wind turbine model using co-simulation with OpenFAST through an FMU. An impedance identification tool has also been implemented to analyze and compare the two models.

> **Frequency support and electro-mechanical interactions:** the turbine is also
> given droop and virtual-inertia frequency support on the LEOGO islanded
> offshore grid, and a set of studies characterises the resulting
> electro-mechanical interactions (tower side-to-side resonance, drivetrain
> torsion) together with a notch-filter mitigation. See
> **[`README_frequency_support.md`](README_frequency_support.md)** for the full
> documentation, models, driver flags and how to run each study.

| Model | Description | Use case |
|-------|-------------|----------|
| **Baseline** | Two-mass drivetrain, MPT/Cp tables, simplified pitch control — all in TOPS | Faster runs for standard power system studies |
| **Coupled** | OpenFAST (aero-servo-elastic simulation) co-simulated with TOPS | Full wind turbine dynamics for more intricate studies |

> Grid-frequency support (droop / virtual inertia), the reduced tower modes
> (side-to-side and fore-aft), and the electro-mechanical interaction studies are
> documented separately in
> [`README_frequency_support.md`](README_frequency_support.md).

## Setup

1. **Clone the repository** and `cd` into the project root.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Repository layout

- `src/tops_openfast/dyn_models/` — wind turbine, converter, and coupling models (`WindTurbine`, `UIC`, `FMUtoUICdrivetrain`, perturbations, …)
- `casestudies/dyn_sim/` — time-domain runs ([`README.md`](casestudies/dyn_sim/README.md)); `plotting/` regenerates thesis figures from CSV
- `casestudies/impedance_stability/` — LF / MF / HF multitone ID and singletone cross-checks

## Running simulations

### Baseline

Standard power system studies representation of a wind turbine, using a two-mass drivetrain and simplified wind turbine control. Couples to a converter model called UIC (Unified Integral Control).

```bash
python casestudies/dyn_sim/test_WT_sim.py
```

Results: `casestudies/dyn_sim/logs/wt/wt_model.csv`, `casestudies/dyn_sim/logs/wt/plots/thesis/`.

Power-system definition: `casestudies/ps_data/test_WT.py` (network, UIC, and `WindTurbine` parameters).

### Coupled

OpenFAST supplies wind turbine dynamics while TOPS supplies the grid, UIC, and a TOPS-side drivetrain matched to `WindTurbine` for direct comparison between the two models.

1. Place the FMU at `OpenFAST/fast.fmu` (preferred) or `fast.fmu` in the project root.

2. Keep the OpenFAST case at `test1002/` (selected by `testNr=1002`).

3. Run:
   ```bash
   python casestudies/dyn_sim/test_WT_FMU_drivetrain_sim.py
   ```

   Results: `casestudies/dyn_sim/logs/fmu_drivetrain/fmu_drivetrain.csv`, `casestudies/dyn_sim/logs/fmu_drivetrain/plots/thesis/`.

   Replot from CSV (baseline + coupled + comparison, no re-simulation):

   ```bash
   python casestudies/dyn_sim/plotting/regenerate_thesis_plots_from_csv.py
   ```

   Comparison figures: `casestudies/dyn_sim/logs/compare/plots/thesis/`

**Configuration:** TOPS — `casestudies/ps_data/test_WT_FMU_drivetrain_.py` (network, UIC, `FMUtoUICdrivetrain` interface). OpenFAST configuration files lie under `test1002/` (`mainInput.fst`, `IEA-15-240-RWT_*.dat`, `ControlData/ROSCO.IEA15MW.IN`, `WindData/`). OpenFAST's initial conditions are set in the case files, so startup transients are to be expected.

## Impedance stability (band-split multitone FFT)

Impedance analysis of the converter terminal: baseline WT vs coupled FMU, using three band-limited multitone runs (LF / MF / HF) and optional singletone checks.

```bash
python casestudies/impedance_stability/multitone_fft_bands/validate_plan.py
python casestudies/impedance_stability/multitone_fft_bands/run_multitone_id_wt.py
python casestudies/impedance_stability/multitone_fft_bands/run_multitone_id_fmu.py
python casestudies/impedance_stability/multitone_fft_bands/run_singletone.py --plant wt --freq 0.17
```

Results under `casestudies/impedance_stability/logs/` (`wt_uic_multitone_fft_bands/`, `fmu_uic_multitone_fft_bands/`, singletone subdirs).

Band specs and workflow: [`casestudies/impedance_stability/README.md`](casestudies/impedance_stability/README.md), [`multitone_fft_bands/README.md`](casestudies/impedance_stability/multitone_fft_bands/README.md). Paths: `casestudies/impedance_stability/paths.py`.

## TOPS

Power flow and time-domain simulation use **TOPS** (Tiny Open Power System Simulator) by Hallvard Haugdal. Thanks for making the underlying tool available.

**Citing:** [this paper](https://arxiv.org/abs/2101.02937). **Contact:** [Hallvard Haugdal](mailto:hallvhau@gmail.com)
