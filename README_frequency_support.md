# Frequency support and electro-mechanical interactions

This extends the coupled OpenFAST↔TOPS model (see the main
[`README.md`](README.md)) with **grid-frequency support from the wind turbine**
and a set of **electro-mechanical interaction** studies on the LEOGO islanded
offshore platform grid.

A grid-forming wind turbine is given **droop and virtual-inertia frequency
support**: the controller reads the live grid frequency, filters it, and turns
it into a **generator-torque offset** that is fed to ROSCO over a ZeroMQ link.
This closes a two-way loop between the electrical grid (TOPS) and the mechanical
turbine (OpenFAST), and the case studies characterise the interactions that
arise, together with a simple filtering-based mitigation.

Two complementary wind-turbine models are used. A fast **reduced-order model**
built entirely in TOPS (with droop support and reduced tower modes, calibrated
against OpenFAST) was developed first, and the **high-fidelity OpenFAST FMU** is
used for the detailed co-simulation studies below.

| Study | What it shows |
| --- | --- |
| **GT-trip frequency support** | Droop / virtual inertia help the weak grid ride through a gas-turbine trip, within the turbine rating (de-load reserve + burst cap). |
| **Tower side-to-side resonance** | The same torque path can excite the tower side-to-side mode (~0.233 Hz); a notch filter removes the resonance while keeping the grid benefit. |
| **Drivetrain torsion vs LPF** | Whether the support loop reaches the ~3.1 Hz drivetrain torsional mode, and how the support low-pass filter shapes it. |

## Simplified (reduced) wind-turbine model

Before and alongside the OpenFAST coupling, a fast **reduced-order wind-turbine
model** was built entirely in TOPS. It runs far faster than the co-simulation
and was used to add the first frequency support and reduced tower modes,
calibrated against the OpenFAST FMU.

- `src/tops_openfast/dyn_models/windturbine.py` — `WindTurbine`: two-mass
  drivetrain (rotor `J_m`, generator `J_e`, flexible shaft `K` / `D`), MPT / Cp
  aerodynamic tables, PI pitch control and speed low-pass filters (10 states).
- `src/tops_openfast/dyn_models/windturbine_tower.py` — `WindTurbineTower`:
  extends `WindTurbine` with droop frequency support and reduced tower modes.

**Droop frequency support** (`_droop_command`) is a simple, stateless linear
droop with a de-loading reserve:

```
ΔP = K_droop · (f_nom − f_grid),   P_ref = clip(P_base + ΔP, 0, P_available)
```

Parameters: `droop_enable`, `K_droop_pu_per_hz`, `headroom_pu`, `f_nom_hz`. This
was the first, simpler frequency support; the OpenFAST / ZeroMQ controller above
adds the low-pass filter, virtual inertia and notch.

**Reduced tower modes** are two second-order modal oscillators driven by the
turbine forcing. They are forward-coupled by default (the tower does not feed
back onto the electrical states unless `ss_feedback_enable=1`):

- **Side-to-side (SS)** — states `q_ss`, `q_ss_dot`, driven by generator torque
  `Te`. Params `ss_enable`, `f_ss_hz` (≈0.234 Hz), `zeta_ss` (≈0.34 %), `g_ss`.
  `ss_acceleration()` mirrors OpenFAST's `YawBrTAyp`.
- **Fore-aft (FA)** — states `q_fa`, `q_fa_dot`, driven by rotor thrust. Params
  `fa_enable`, `f_fa_hz` (≈0.235 Hz), `zeta_fa` (≈1.25 %), `g_fa`.
  `fa_acceleration()` mirrors `YawBrTAxp`.

The modal gains and damping were **calibrated against the OpenFAST FMU** (the SS
gain matched to the FMU resonance peak, the damping taken from an FMU ring-down).
The reduced model was needed because ROSCO (VSContrl=5) rejects external
generator-torque / power commands in the FMU, so the electrical → mechanical
excitation path is otherwise blocked; the reduced model keeps that path open by
construction and runs fast. The later ZeroMQ work reopened the path through
ROSCO's own interface.

### Reduced-model case studies

- `casestudies/dyn_sim/test_WT_sim.py` — baseline reduced WT run (see main README).
- `casestudies/dyn_sim/test_WT_LEOGO_frequency_sim.py` — droop on vs off under a
  LEOGO load event (`droop_enable`, `K_droop_pu_per_hz`, `headroom_pu`).
- `casestudies/dyn_sim/test_WT_LEOGO_torsional_resonance_sim.py` — drivetrain
  torsion frequency sweep near ~3.49 Hz (`--forcing-freq-hz`, `--sweep ...`).
- `casestudies/dyn_sim/plot_ss_fa_excitation.py` — grid-driven excitation of the
  SS (≈0.234 Hz) and FA (≈0.235 Hz) modes.
- Calibration sweeps under `casestudies/dyn_sim/_archive/sweeps/`
  (`sweep_ss_reduced.py`, `sweep_resonance.py`).

### Background notes (`docs/`)

- `frequency_analysis.tex` — reduced two-mass drivetrain torsional mode ≈3.49 Hz.
- `forced_response.tex`, `network_excitation.tex` — electrically driven drivetrain
  torsion (from the turbine terminal and from the LEOGO PCC).
- `tower_ss_resonance.tex` — tower SS mode ≈0.234 Hz vs the LEOGO grid mode ≈0.226 Hz.
- `openfast_excitation.tex` — why external torque commands failed in the FMU
  (this motivated the reduced model).
- `drivetrain_torsion_resonance.tex` — drivetrain torsion in the high-fidelity FMU (≈3.1 Hz).

Key frequencies: tower SS ≈0.234 Hz, tower FA ≈0.235 Hz, drivetrain torsion
≈3.49 Hz (reduced) / ≈3.1 Hz (FMU), LEOGO grid mode ≈0.226 Hz.

## Setup

Same environment as the main README:

```
pip install -r requirements.txt
```

Additional requirements for these studies:

1. Place the OpenFAST FMU at the project root: `fast.fmu` (fast) and, for
   studies that need the tower side-to-side acceleration output
   (`fmu_YawBrTAyp`), `fast_debug.fmu` (slower).
2. Keep the OpenFAST case at `test1002/` (`testNr=1002`). The frequency-support
   runs use ElastoDyn in a **side-to-side-only** configuration
   (`TwSSDOF1=True`, `TwFADOF1=False`, `GenDOF=True`, `DrTrDOF=False`) in
   `test1002/IEA-15-240-RWT-Monopile_ElastoDyn.dat`.
3. ROSCO must have the ZeroMQ interface enabled
   (`ZMQ_Mode=1` in `test1002/ControlData/ROSCO.IEA15MW.IN`).

## Repository layout and configuration

Key folders:

- `src/tops_openfast/dyn_models/` — models: `WindTurbine`, `WindTurbineTower`
  (reduced model), `FMUtoUICdrivetrain`, `UIC`, `speed_lpf`, perturbations.
- `casestudies/dyn_sim/` — time-domain drivers, runner scripts and plotting.
- `casestudies/ps_data/` — power-system / model **parameter definitions**
  (network, buses, lines, UIC, and the reduced wind-turbine parameters).
- `casestudies/impedance_stability/` — impedance-identification tools.
- `test1002/` — OpenFAST case files (ElastoDyn, InflowWind, ServoDyn, ROSCO, wind).
- `results/em_interaction_sweep/` — output CSVs and figures for these studies.
- `docs/` — background notes (`.tex`).

### Where the parameters are set

There are three places to configure a run, depending on the model:

1. **Reduced-model studies** — edit the power-system definition file
   `casestudies/ps_data/test_WT_LEOGO.py`. It defines the network, the UIC
   converter and the `WindTurbine` / `WindTurbineTower` parameters, including the
   droop (`droop_enable`, `K_droop_pu_per_hz`, `headroom_pu`) and the reduced
   tower modes (`ss_enable`, `f_ss_hz`, `zeta_ss`, `g_ss`, `fa_enable`,
   `f_fa_hz`, `zeta_fa`, `g_fa`). `test_WT.py` holds the baseline (no-LEOGO) case.

2. **OpenFAST / ZeroMQ FMU studies** — pass **command-line flags** to
   `test_WT_LEOGO_FMU_sim.py` (see *Key driver flags* below). The FMU-side
   coupling parameters (`J_m`, `J_e`, `K`, `D`, `ElecPwrCom_kW`, …) are defined
   inline in that driver, and the OpenFAST case is configured under `test1002/`
   — the ElastoDyn DOFs in `IEA-15-240-RWT-Monopile_ElastoDyn.dat` and `ZMQ_Mode`
   in `ControlData/ROSCO.IEA15MW.IN`.

3. **Sweep / batch runner scripts** (e.g. `freq_support_3way.py`,
   `ss_notch_resonance_curve.py`, `drivetrain_torsion_lpf.py`) — edit the
   constants near the top of each script (frequencies, gains, `RUN_LEGS` /
   `CASES`, output paths). They build the driver command lines for you and are
   resumable.

## Core components

- `casestudies/dyn_sim/test_WT_LEOGO_FMU_sim.py` — main driver. Couples the
  OpenFAST/ROSCO FMU to the LEOGO grid (TOPS + `FMUtoUICdrivetrain` + UIC),
  runs the frequency-support controller each control step, applies the chosen
  disturbance, and logs the results to CSV.
- `casestudies/dyn_sim/rosco_zmq_grid_coupling.py` — the frequency-support
  controller and ZeroMQ **REP** server. Pipeline on the measured grid
  frequency:

  ```
  f_grid → low-pass (LPF) → notch → K_droop·Δf + K_inertia·dΔf/dt → clip → torque offset
  ```

  The offset is returned to ROSCO (ZeroMQ **REQ** client inside the FMU) as
  setpoint 0 at every control step. The notch is a second-order RBJ band-stop
  biquad tuned to the tower frequency.
- `src/tops_openfast/dyn_models/speed_lpf.py` — ROSCO-style speed low-pass
  filter helpers used by the coupling.
- `src/tops_openfast/dyn_models/FMUtoUICdrivetrain.py` — the FMU↔UIC coupling
  with a two-mass drivetrain (extended for these studies).

### Key driver flags

| Flag | Meaning |
| --- | --- |
| `--fmu {fast,debug}` | Select `fast.fmu` or `fast_debug.fmu` (debug exposes `fmu_YawBrTAyp`). |
| `--zmq-grid` | Run the ZeroMQ grid-coupling controller (frequency support). |
| `--fix-leogo-xqt` | Damp the ~5.3 Hz LEOGO q-axis artifact mode. |
| `--droop-nm-per-hz`, `--inertia-nm-s-per-hz` | Droop and virtual-inertia gains. |
| `--freq-lpf-hz`, `--freq-lpf-order` | Support low-pass corner and order. |
| `--support-notch-hz`, `--support-notch-q` | Notch centre frequency and quality factor (0 = off). |
| `--support-max-nm`, `--support-max-over-nm` | Symmetric clip and asymmetric over-rating cap on the offset. |
| `--support-start`, `--deload-nm` | Support activation time and standing de-load (reserve). |
| `--load-step-mw`, `--event-time`, `--event-duration`, `--load-ramp-on-s` | Load-step / gas-turbine-trip disturbance. |
| `--load-sine-mean`, `--load-sine-amplitude`, `--load-sine-freq-hz` | Sustained sinusoidal process-load (slug-flow) disturbance. |
| `--t-end`, `--out`, `--zmq-log` | Run length and output CSV paths. |

## Running the studies

All studies write under `results/em_interaction_sweep/`. Most runner scripts are
resumable (they skip CSVs that already exist) and accept `--plot-only` to
regenerate figures without re-simulating.

### Gas-turbine-trip frequency support

Droop / virtual inertia during the LEOGO N-1 gas-turbine trip.

```
python casestudies/dyn_sim/freq_support_3way.py --di-inertia 8e7 --di-lpf 2.0 --max-over-nm -3e6 --suffix _cap
python casestudies/dyn_sim/plot_freq_support_5way.py --cap
python casestudies/dyn_sim/plot_freq_support_5way_mechanics.py
```

Results: `results/em_interaction_sweep/freq_support_3way/`.

### Droop / inertia tuning sweep

```
python casestudies/dyn_sim/sweep_droop_inertia_opt.py --block all
python casestudies/dyn_sim/plot_droop_inertia_opt.py
```

Results: `results/em_interaction_sweep/droop_inertia_opt/` (nadir / RoCoF /
peak-power heatmaps, Pareto and LPF trade-off).

### Tower side-to-side resonance and the notch

Sustained slug-flow load near the tower side-to-side mode; shows the resonance
and its removal by the support-path notch (needs `fast_debug.fmu`).

```
python casestudies/dyn_sim/ss_resonance_cases.py            # 5-case bar chart
python casestudies/dyn_sim/ss_notch_resonance_curve.py      # frequency sweep
python casestudies/dyn_sim/plot_notch_payoff.py             # "notch is (almost) free" figure
python casestudies/dyn_sim/plot_ss_notch_combined_5e6.py    # sweep + time series
```

Results: `results/em_interaction_sweep/ss_resonance/` and
`results/em_interaction_sweep/full_matrix/`.

### Drivetrain torsion vs support LPF

```
python casestudies/dyn_sim/drivetrain_torsion_lpf.py
```

Results: `results/em_interaction_sweep/drivetrain_torsion_lpf/`.

### Full electro-mechanical interaction matrix

```
python casestudies/dyn_sim/sweep_em_full_matrix.py
python casestudies/dyn_sim/plot_em_full_matrix.py
```

Results: `results/em_interaction_sweep/full_matrix/`.

## Results layout

```
results/em_interaction_sweep/
  freq_support_3way/          gas-turbine-trip cases (grid frequency, power, mechanics)
  droop_inertia_opt/          droop / inertia tuning sweep
  ss_resonance/               sustained-load tower SS resonance cases
  full_matrix/                SS resonance frequency sweep + combined figures
  drivetrain_torsion_lpf/     drivetrain torsion vs support LPF
  tower_ss_resonance/         tower SS studies
  two_turbine/                multi-turbine runs
```

## Notes and gotchas

- **One run at a time.** ROSCO's ZeroMQ interface binds `tcp://*:5555`, and the
  OpenFAST case files under `test1002/` are shared, so only a single FMU run can
  execute at a time.
- **Blocking ZeroMQ.** ROSCO's request–reply is synchronous, so it blocks until
  the server replies. If the machine sleeps mid-exchange the run can deadlock;
  keep the machine awake during long runs.
- **Debug FMU is slow.** `fast_debug.fmu` (needed for `fmu_YawBrTAyp`) runs at
  roughly real-time-order speed, so a `t_end 400` run takes tens of minutes.
- **Torque vs power.** Frequency support is injected as a **generator-torque**
  offset (ROSCO's native control variable). Near rated speed torque is close to
  proportional to power; a power command (`--perfect-tracking`, `ElecPwrCom`)
  is available for comparison.

## TOPS

**Citing:** [this paper](https://arxiv.org/abs/2101.02937). **Contact:** [Hallvard Haugdal](mailto:hallvhau@gmail.com)
