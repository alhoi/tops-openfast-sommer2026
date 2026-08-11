# `_archive/` — arkiverte filer (ikke slettet)

Disse filene ble flyttet hit under opprydding for å holde `casestudies/dyn_sim/`
fokusert på den aktive OpenFAST-FMU + LEOGO frekvensstøtte-pipelinen.
**Ingenting er slettet** — flytt tilbake til `casestudies/dyn_sim/` ved behov.

## Aktiv pipeline (ligger IKKE her, står i `dyn_sim/`)
`test_WT_LEOGO_FMU_sim.py`, `rosco_zmq_grid_coupling.py`,
`plot_r3_resonance.py`, `plot_freq_support.py`, `build_rosco_zmq.ps1`,
`_analyze_gt_trip_fmu.py`, `_compare_zmqgrid_onoff.py`,
`_check_zmq_live.py`, `_check_oppoint.py`.

## Undermapper
- **`scratch_diagnostics/`** — engangs sjekke-/diagnose-skript (`_check_*`,
  `_diag*`, `_measure_*`, `_find_*`, `_verify_*`, gamle `_analyze_*`).
- **`old_drivers/`** — tidligere simuleringsdrivere (reduserte WT-modeller,
  torsjon, 2WT, tidligere FMU-varianter, `test_WT_LEOGO_sim_BACKUP.py`).
- **`old_plots/`** — tidligere plotteskript (`plot_em_interaction_*`,
  `plot_fmu_genuine_*`, `plot_slugflow_*`, m.fl.).
- **`sweeps/`** — parameter-/frekvenssveip-skript (`sweep_*`).
- **`openloop_and_tuning/`** — åpen-sløyfe-generatorer (`make_ol_*`),
  droop-tuning og den gamle åpen-sløyfe ZMQ-serveren (`rosco_zmq_server.py`).
- **`misc_sims/`** — ikke-LEOGO eksempler (`k2a_sim*`, `assignment_sim`,
  `line_outage`, `short_circuit`, `oscillator*`).
