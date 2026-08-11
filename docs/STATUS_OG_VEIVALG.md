# Status og veivalg — elektrisk-drevet mekanisk resonans (IEA 15 MW + LEOGO)

_Overleveringsnotat før ferie. Skrevet 2026-07-10. Alt her er verifisert mot koden/konfigurasjonen i repoet._

---

## 0. TL;DR (les dette først)

- **Det som ALLEREDE fungerer og er ferdig dokumentert:** elektrisk → mekanisk
  torsjonsresonans (drivverket, ~3.49 Hz) i BÅDE den forenklede TOPS-modellen og
  i FMU-en. En nett-forstyrrelse (shunt-last / prosesslast på LEOGO-bussen) driver
  drivverket til resonans. Dette er en komplett avhandlings-demonstrasjon.
- **Det som IKKE fungerer:** å eksitere OpenFAST sine **strukturelle tårnmoder**
  (side-to-side ~0.234 Hz, fore-aft) via nett/elektrisk vei. Årsak: ROSCO
  (`VSContrl=5`) eier generatormomentet internt og ignorerer alt vi sender inn
  (`GenSpdOrTrq`, `ElecPwrCom`). `VSContrl=4` (ekstern Simulink-moment) krasjer —
  binæren mangler pekeroppkoblingen, krever FMU-rebuild.
- **NYTT OG VIKTIGSTE FUNN (start her etter ferie):** ROSCO har **Open-Loop
  Control** (`OL_Mode` + `Ind_GenTq`). Da leser ROSCO generatormomentet fra en
  tidsserie-fil (`OLInput_ROSCO.dat`) ved **runtime** — akkurat som alle andre
  test1002-filer. Det betyr at vi kan injisere en momentsvingning rett inn i
  OpenFAST sitt faktiske generatormoment **UTEN å bygge om FMU-en og uten
  VSContrl=4**. Dette er den reelle veien til å manipulere momentet og eksitere
  side-to-side / fore-aft. Detaljert oppskrift i §3.
- **Reserveplan (vei 2):** to vindturbiner og studere en shaft-/inter-turbin-mode.
  Krever endringer for å kjøre to OpenFAST-instanser (Fortran global state,
  separate arbeidskataloger). Går i dybden i §4. Det finnes også en **pragmatisk
  snarvei** (turbin 2 = forenklet 2-masse-modell) som unngår hele
  multi-FMU-problemet — se §4.5.
- **Arkitektur (§5):** droop finnes to steder. Den ytre WT-sløyfa
  (`_droop_command`, global COI-frekvens) er overflødig — **UIC-en har allerede en
  innebygd droop** (Unified Integral Control; `Ki`/`Kv` = `k_i`/`k_v`), og den er
  **aktiv nå** siden `perfect_tracking=0`. Anbefaling: fjern den ytre WT-droopen
  (`droop_enable=0`) og styr frekvensresponsen via UIC-en. Ikke-blokkerende.

---

## 1. Systemet i korte trekk

- **Co-simulering:** TOPS (elektrisk nett + regulering) ↔ OpenFAST-FMU (aerodynamikk,
  struktur, ROSCO) via FMPy, fast kommunikasjonssteg `dt = 0.01 s`.
- **FMU-en leser OpenFAST-inputfilene i `test1002/` ved RUNTIME.** Bevist gjentatte
  ganger (endringer i vindhastighet, RotSpeed, BlPitch, VSContrl slår inn uten
  rebuild). Dette er nøkkelen til at OL_Mode-veien i §3 kan gjøres uten rebuild.
- **To modeller finnes:**
  1. **Forenklet TOPS-`WindTurbine`** (2-masse drivverk). E→M-veien er ÅPEN:
     `Te = Pe / (omega_e_filt·eta)`, `Pe` = levert nett-effekt. Torsjonsmoden
     resonerer. **Dette er arbeidshesten for E→M-demonstrasjonen.**
  2. **FMU-`FMUtoUICdrivetrain`** (OpenFAST + ROSCO). To viktige moder:
     - **3.49 Hz torsjon** lever i WRAPPEREN (`FMUtoUICdrivetrain`), ikke i
       OpenFAST. Generatormassen `omega_e` (treghet `H_e`) på en myk aksel
       (`K = K_original/100`), rotorhastighet `omega_m` foreskrevet fra OpenFAST
       `RotSpeed`. Denne føler nett-momentet direkte og resonerer.
     - **Side-to-side tårnmode ~0.234 Hz** (og fore-aft) lever INNE i OpenFAST.
       Denne er blokkert fra nettet (§2).

---

## 2. Hva fungerer og hva fungerer ikke (validert)

### 2.1 FUNGERER ✔

| Fenomen | Modell | Resultat |
|---|---|---|
| Torsjonsmode E→M-resonans | Forenklet 2-masse | f = 3.491 Hz, ζ = 4.37 %. Shunt-last ved WT-buss: T_shaft topp-topp peaker skarpt ved 3.49 Hz, ~**36×** forsterkning vs 1 Hz. |
| Torsjonsmode E→M-resonans | FMU (wrapper) | f_n = 3.485 Hz, ζ = 4.44 %. 0.5 MW forcing → T_shaft topp-topp 8.57 MNm ved 3.49 Hz, ~5.2× vs 1 Hz. |
| LEOGO-nett → turbin-torsjon | Forenklet | Samme mode nås fra LEOGO-bussen, dempet ~**13×** av nett-impedansen, men frekvens/form bevart. |
| Realistisk prosesslast → torsjon | Forenklet | 0.5 MW pulsasjon @3.49 Hz → 74 kNm akselmoment. 9.7 MW lastkobling → ring-down ζ = 4.29 % (matcher egenverdi 4.37 %). |
| Aerodynamisk eksitasjon av SS | FMU | Oscillerende vindretning (yaw-misalignment) @0.234 Hz → SS bygger seg opp til ±0.42 m/s². Men dette er AERO, ikke elektrisk. |

Alt over har sim-skript, plott og (for torsjon) LaTeX-seksjon
(`docs/forced_response.tex`, `\label{sec:forcedresponse}`).

### 2.2 FUNGERER IKKE �’ (og hvorfor)

| Forsøk | Resultat | Rotårsak |
|---|---|---|
| Injisere moment via `GenSpdOrTrq` | Ignorert (bit-identisk GenTq med/uten) | `VSContrl=5`: ROSCO regner momentet selv fra målt generatorhastighet (Kω²/WSE-TSR). |
| Injisere effekt via `ElecPwrCom` (±20 %) | Ignorert i både Region 2 og 3 | Samme — ROSCO eier momentet. |
| `VSContrl=4` (ekstern Simulink-moment) | **KRASJ** ved første `doStep` (`OSError: access violation writing 0x0…0`) | Binæren mangler Simulink/Labview-pekeroppkoblingen. Ikke fiksbart fra config — krever FMU-rebuild. |
| Filtre skjuler SS? | Nei | ROSCO-filtrene (`F_LPFCornerFreq` m.fl.) demper bare hvordan ROSCO RESPONDERER på hastighetssignalet; de er ikke grunnen til at nettet ikke når SS. |

**Kjernen:** i denne FMU-en (VSContrl=5) er OpenFAST-generatormomentet fullstendig
eid av ROSCO. Nett-forstyrrelsen når kun wrapper-generatormassen (→ 3.49 Hz
torsjon), aldri OpenFAST sin egen struktur (→ tårnmoder). Det er en full-converter
frakobling uten returvei.

---

## 3. VEI 1 — manipulere momentet i OpenFAST for å eksitere SS / fore-aft

Målet: få en momentsvingning inn i OpenFAST sitt **faktiske** generatormoment, slik
at reaksjonsmomentet på nacellen eksiterer tårnmodene.

### 3.1 Fysikken (hvorfor moment → side-to-side)

- Generatormomentet virker om den (horisontale) akselaksen. Reaksjonsmomentet på
  nacellen ruller nacellen sideveis → kobler til **side-to-side** tårnbøying.
- Rotor-**trykket** (thrust, langs akselen) driver **fore-aft**.
- Så: en momentsvingning ved SS-frekvensen (0.234 Hz) treffer SPESIFIKT
  side-to-side-moden. (At ROSCO har en egen `TRA_Mode` = "tower resonance
  avoidance" som bruker MOMENT for å unngå en tårnfrekvens, bekrefter at
  moment→tårn-koblingen finnes i modellen.)

### 3.2 Løsning A (ANBEFALT, INGEN REBUILD): ROSCO Open-Loop Control

ROSCO v2.10.0 (den vi kjører) støtter open-loop-styring. Da leser ROSCO
generatormomentet fra en tidsserie-fil i stedet for å regne det selv. Fila leses
ved runtime, så **ingen FMU-rebuild trengs**.

Relevante linjer i `test1002/ControlData/ROSCO.IEA15MW.IN`:
- `OL_Mode` (linje 28) — sett `0 → 1` (open loop vs. tid).
- Open-Loop-blokken (linje ~197–205):
  - `OL_Filename = "OLInput_ROSCO.dat"` (finnes IKKE ennå — må lages).
  - `Ind_Breakpoint = 1` (tid i kolonne 1).
  - `Ind_GenTq` — sett `0 → 2` (generatormoment i kolonne 2, enhet **Nm**).

**Oppskrift etter ferie:**

1. Ta backup: `ROSCO.IEA15MW.IN` → `.bak_OLmode`.
2. Sett `OL_Mode = 1` og `Ind_GenTq = 2` (behold `Ind_Breakpoint = 1`).
3. Lag `test1002/ControlData/OLInput_ROSCO.dat` med to kolonner (tid, GenTq[Nm]):
   - Middelmoment ≈ **1.1179e7 Nm** (= 11179 kNm, målt drift ved 8 m/s / Region 2).
     (Rated er 1.9624e7 Nm — ikke bruk rated i Region 2.)
   - GenTq(t) = middel + A·sin(2π·0.234·(t − t_onset)), t_onset ≈ 30 s (etter oppstart).
   - Start med A ≈ 5 % av middel ≈ **±0.56e6 Nm**. Tett tidsoppløsning (≤0.05 s).
   - Filformat: ROSCO forventer en tabell med header-linjer + numeriske kolonner
     (se ROSCO-dokumentasjonen for "Open Loop Input"; typisk `! <n_lines> <n_cols>`
     header + kommentarrad). Sjekk `rosco/toolbox` eksempelfil `Example_OL.dat`.
4. Kjør med `fast_debug.fmu` (den eksponerer `YawBrTAyp` = SS-akselerasjon):
   ```powershell
   Set-Location "tops-openfast-sommer2026"
   .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_FMU_sim.py `
     --t-end 200 --load-step-mw 0
   ```
   (Steady vind, ingen last-event — da er momentsvingningen ENESTE eksitasjon.)
5. Analyser `YawBrTAyp` (SS) og `YawBrTAxp` (fore-aft): forventer at SS BYGGER seg
   opp mot en stasjonær tvunget amplitude (i motsetning til fri-henfall som vi så i
   alle ignorerte moment-/effekt-kjøringene). Bruk `_ss_analysis.py`.

**Viktige forbehold med OL_Mode:**
- `OL_Mode=1` med `Ind_GenTq` **overstyrer** ROSCO sin momentregulering helt. Da
  forsvinner hastighetsreguleringen → rotoren kan drive. Hold kjøringene korte
  (60–200 s), overvåk `RotSpeed`/`GenSpeed` for drift. Middelmoment må matche
  aerodynamisk moment ved 8 m/s rimelig godt, ellers akselererer/bremser rotoren.
- Alternativ hvis drift blir et problem: sett middelverdien lik det ROSCO selv
  ville brukt (les av `GenTq` fra en baseline-kjøring først), eller bruk en veldig
  liten amplitude så nettodrift over vinduet blir neglisjerbar.
- Blade pitch beholdes av ROSCO (`Ind_BldPitch = 0 0 0`), som er greit i Region 2
  (pitch = 0).

### 3.3 Løsning B (tyngre): FMU-rebuild med VSContrl=4

Hvis OL_Mode ikke gir nok/ren eksitasjon: bygg om FMU-en slik at `VSContrl=4`
(ekstern moment) faktisk kobler `GenSpdOrTrq` til OpenFAST-generatormomentet. Da blir
wrapperens `_te_pu` (nett-effekt-avledet moment) det FAKTISKE momentet → ekte
nett→moment→struktur-vei. Krever tilgang til FMU-byggekjeden (kildekode +
Simulink/Labview-portoppkobling). Dette er den "riktige" veien for
avhandlings-historien "nett-hendelse eksiterer tårnmode", men er mer arbeid.

### 3.4 Hvis vei 1 lykkes

Da har du den sterke historien: en elektrisk/moment-forstyrrelse ved 0.234 Hz
eksiterer side-to-side tårnmoden i en høy-fidelitet OpenFAST-modell. Sammenlign
on-/off-resonans, gjør en frekvenssveip, og skriv en LaTeX-seksjon parallelt med
`forced_response.tex`.

---

## 4. VEI 2 (reserve) — to vindturbiner og shaft-/inter-turbin-mode

Hvis verken OL_Mode eller rebuild gir SS/fore-aft-eksitasjon, er dette planen.
Idé: to turbiner koblet til samme nett kan utveksle energi gjennom nettet og danne
en lett dempet **inter-turbin shaft-mode** (to drivverk som svinger mot hverandre
via omformerne/nettet). En nett-hendelse kan eksitere denne, og den er
elektrisk-mekanisk av natur.

### 4.1 Hva "endre FMU-en for 2 turbiner" faktisk betyr

Utfordringen er ikke FMU-*formatet*, men å kjøre **to OpenFAST-instanser i samme
Python-prosess**. Delproblemene:

**(a) Fortran global/modul-state (den store).**
OpenFAST bruker mye `SAVE`/modul-nivå global tilstand. Hvis samme DLL lastes to
ganger i samme prosess, deler instans 2 instans 1 sine globaler → korrupsjon. Dette
er den klassiske grunnen til at man ikke kan kjøre to OpenFAST i én prosess.
- **Godt nytt (hypotese å teste FØRST):** i koden gjør hver modell
  `unzipdir = extract(fmu_file)`, og FMPy sin `extract()` lager en **unik temp-katalog
  per kall**. To instanser får dermed to FYSISK forskjellige kopier av DLL-en på
  ulike stier. Windows `LoadLibrary` nøkler på full sti → to forskjellige stier gir
  to separate modul-instanser → **separat global state automatisk**. Så
  DLL-kopi-trikset skjer allerede av seg selv. Dette KAN bety at to instanser
  fungerer uten rebuild — men det MÅ testes; noen builds cacher på DLL-basenavn.
- Se `FMUtoUICdrivetrain.__init__` (extract + instantiate) i
  [FMUtoUICdrivetrain.py](../src/tops_openfast/dyn_models/FMUtoUICdrivetrain.py).

**(b) Separate arbeidskataloger.**
FMU-en leser arbeidsmappa fra `resources/wd.txt` inne i den utpakkede FMU-en, og
OpenFAST skriver output (`mainInput.outb`, `fort.30`, `.dbg` …) DIT. To instanser
som peker på samme `test1002/` vil overskrive hverandres filer.
- Fiks: gi instans 2 sin egen mappe, f.eks. kopier `test1002/` → `test1002b/`, og
  sett `openfast_test_dir` for instans 2 til `test1002b`. Parameteren finnes
  allerede (`par['openfast_test_dir']` skrives til `wd.txt` per instans).

**(c) Unikt `instanceName`.**
`instanceName='instance1'` er hardkodet i begge wrapper-klassene. Gjør det unikt per
instans (kosmetisk, men ryddig, og noen FMI-implementasjoner bryr seg).

**(d) `testNr`.**
Begge sender `testNr` (1002). Sjekk om FMU-en bruker den til å velge katalog eller
bare som ID — gi evt. instans 2 en egen `testNr`.

### 4.2 TOPS-siden: to UIC + to drivverk

- I `build_model()` (i
  [test_WT_LEOGO_FMU_sim.py](../casestudies/dyn_sim/test_WT_LEOGO_FMU_sim.py)):
  legg til en **andre UIC** og en **andre `FMUtoUICdrivetrain`**, koblet til nettet
  ved en annen WT-buss (eller samme park-buss via `Trafo WindPark`).
- Hver `FMUtoUICdrivetrain` peker på sin egen `openfast_test_dir` og sin egen
  `.fmu`-fil (evt. samme fil — `extract()` gir uansett ulike temp-kataloger).
- Kjøre-løkka (`step_fmu` per modell) håndterer allerede flere FMU-modeller så lenge
  hver har egen instans; ingen prinsipiell blokker på TOPS-siden.

### 4.3 Hva "shaft-moden" mellom to turbiner er

- Med ÉN turbin er alle turbinmoder **WT-isolerte** (interaksjonsanalysen:
  torsjon 3.49 Hz har WT-side-deltakelse = 1.00, nett-kobling neglisjerbar).
- Med TO turbiner på samme svake nett får du en ny frihetsgrad: de to drivverkene
  kan svinge i **motfase** gjennom nettet (omformer-medierte). Dette er en
  inter-turbin elektromekanisk mode — analog til en inter-area-mode i klassiske
  kraftsystemer, men mellom to aksel-/drivverk. En nett-hendelse eksiterer typisk
  denne bedre enn den WT-isolerte enkelt-torsjonen.
- Analyseverktøyene finnes allerede: gjenbruk
  `casestudies/modal_analysis/interaction_WT_LEOGO.py` og
  `participation_WT_LEOGO_torsional.py` på den 2-turbin-modellen for å finne
  frekvens/demping/deltakelse for den nye moden, deretter tvungen respons som i
  `test_WT_LEOGO_torsional_resonance_sim.py`.

### 4.4 Konkrete steg (vei 2)

1. **Test multi-instans-hypotesen billig FØRST:** instansier to
   `FMUtoUICdrivetrain` med hver sin `openfast_test_dir` (`test1002` og `test1002b`)
   og kjør noen få steg. Krasjer det (global-state-kollisjon) eller kjører det?
   Dette avgjør om rebuild trengs.
2. Hvis OK: bygg 2-turbin `build_model()`, kjør småsignal-analyse for å finne
   inter-turbin-moden.
3. Tvungen respons: shunt-last på LEOGO-bussen ved den nye modefrekvensen, vis
   oppbygging + FFT + ring-down-ζ.
4. Hvis multi-instans krasjer og du ikke vil bygge om FMU-en → §4.5.

### 4.5 Pragmatisk snarvei (unngår hele multi-FMU-problemet)

La **turbin 1 = FMU** og **turbin 2 = den forenklede 2-masse `WindTurbine`-modellen**
(som allerede har åpen E→M-vei og er billig). Da får du fortsatt to drivverk koblet
gjennom nettet og kan studere inter-turbin shaft-interaksjonen — uten å kjøre to
OpenFAST-instanser i samme prosess. Dette er den raskeste veien til et 2-turbin-
resultat hvis §4.1(a) viser seg vanskelig.

---

## 5. Arkitektur — fjerne ytre-sløyfe-droop og bruke UIC-ens innebygde droop

> **RETTELSE:** en tidligere versjon av dette notatet påsto at UIC-en ikke har
> droop. Det er **feil**. UIC-en (Unified Integral Control) ER en droop-regulator
> ved design, og droopen er **aktiv nå**. Under er det korrigert.

### 5.1 To droop-mekanismer finnes — én er overflødig

**(1) Ytre WT-sløyfe-droop** (`WindTurbine._droop_command` i
[windturbine.py](../src/tops_openfast/dyn_models/windturbine.py)):
- Lov: `ΔP = K_droop · (f_nom − f_grid)`, lagt oppe på av-lastet basis
  `P_base = P_available − headroom`, klippet til tilgjengelig effekt.
- `f_grid` settes utenfra via `set_grid_frequency_hz(...)`; i sim-skriptene er dette
  **COI-frekvensen** til LEOGO-generatorene, ikke en lokal måling.
- Parametre: `f_nom_hz`, `droop_enable`, `K_droop_pu_per_hz`, `headroom_pu`.
- Finnes KUN i den forenklede `WindTurbine`; FMU-turbinen har den ikke.

**(2) UIC-ens innebygde droop** (Unified Integral Control, `UIC_sig` i
[UIC.py](../src/tops_openfast/dyn_models/UIC.py)). Kontrolloven i koden er

```
dvi = j·ω_n·Ki·i_error  +  ω_n·Kv·v_error  +  j·vi·x_filter·perfect_tracking
```

som er nøyaktig avhandlingens likning (2.10),
`v̇_i^{ω0} = (j·k_i·ε_s + k_v·ε_v)·e^{jΔθ}`, med kobling til koden:
- `Ki` ↔ `k_i` (= `m_q/T`), `Kv` ↔ `k_v` (= `1/T`) — likn. (2.14).
- `i_error` bærer effekt-referansen (`i_ref = conj(s_ref/vi)`), altså rollen til
  `ε_s = Δs̄ = Δp − jΔq`. `j`-rotasjonen gjør at et **effektavvik → frekvens/vinkel**
  (P-f droop, likn. 2.11) og et **spenningsavvik → magnitude** (Q-V droop, likn. 2.12).
- `perfect_tracking`-leddet er nettopp likn. (2.15): det mater
  frekvensavviket (`x_filter ≈ Δω̃`) tilbake i integralet og **kansellerer droopen**
  → PLL-fri "perfect tracking".

**Nøkkel:** droopen er PÅ når `perfect_tracking = 0`, og AV når `perfect_tracking = 1`.

### 5.2 Hva modellene faktisk bruker nå (verifisert)

I både `test_WT_LEOGO_sim.py` og
[test_WT_LEOGO_FMU_sim.py](../casestudies/dyn_sim/test_WT_LEOGO_FMU_sim.py):
`Ki = 0.03`, `Kv = 0.0`, `xf = 0.1`, **`perfect_tracking = 0`**, `T_filter = 0.01`.
Kommentaren i FMU-skriptet sier eksplisitt: `perfect_tracking (0 = grid disturbance
reaches the WT)`.

Altså: **UIC-ens droop er allerede aktiv** (via `Ki=0.03`, P-f-kobling gjennom
`j`-leddet). `Kv=0` betyr at den separate Q-V-magnitude-sløyfa er avslått, men
P-f-droopen virker. Dette er også grunnen til at nett-forstyrrelser i det hele tatt
når WT-en i FMU-modellen — omformeren er droop-koblet, ikke stiv.

### 5.3 Anbefaling: fjern den ytre WT-droopen, behold UIC-droopen

Siden UIC-en allerede gir en lokal, fysisk droop, er den ytre WT-sløyfa (som bruker
en global COI-frekvens regnet ut i skriptet) overflødig og litt kunstig. Ryddejobb:

1. **Slå av den ytre droopen:** `droop_enable = 0` i WT-parametrene (allerede tilfellet
   i torsjons-forcing-skriptene). WT/FMU leverer da bare av-lastet MPT-effekt som
   `p_ref`; ingen frekvens-modulasjon i den mekaniske sløyfa.
2. **Behold `perfect_tracking = 0`** så UIC-droopen er aktiv (sett den til 1 KUN hvis
   du vil ha PLL-fri stiv sporing UTEN droop).
3. **Still droop-styrken via `Ki` (= `k_i`)** — og `Kv` (= `k_v`) hvis du også vil ha
   Q-V-droop. Tun mot ønsket P-f-helling (likn. 2.11–2.14). Da bestemmes
   frekvensresponsen ett sted (omformeren), lokalt målt, i stedet for to steder.
4. **Fordel:** dette gir frekvensstøtte også for **FMU-turbinen** (som mangler ytre
   droop) uten å røre FMU-en, siden droopen sitter i UIC-en.

### 5.4 Forbehold / ting å sjekke

- **Egenverdi-sjekk etterpå:** `Ki`/`Kv` er allerede en aktiv tilbakekoblingssløyfe;
  å endre dem (eller fjerne den ytre droopen) flytter demping på den nett-koblede
  ~1.445 Hz-moden og potensielt på torsjonen. Kjør
  `casestudies/modal_analysis/interaction_WT_LEOGO.py` /
  `param_sweep_uic_coupling_WT_LEOGO.py` (den sveiper allerede `Ki`) etter endring.
  (Fra tidligere sveip: `Ki<0.02` kollapser dempingen på vi_x-moden til ~15–18 %.)
- **Relevans for resonans-studien:** torsjons-forcing-skriptene kjører allerede med
  `droop_enable=0`, så å fjerne den ytre droopen **endrer ikke** torsjonsresultatene
  (§2.1). UIC-droopen (`perfect_tracking=0`) var dessuten aktiv i alle de kjøringene.
  Dette er altså en arkitektur-opprydding, ikke en blokker.
- **Av-lasting/headroom:** ligger i WT/FMU (`P_available − headroom`). Behold det der;
  UIC-droopen jobber oppå den leverte `p_ref`, begrenset av strømgrensen
  (`i_max_pu = 1.9`, finnes allerede).

---

## 6. Prioritert rekkefølge etter ferie

1. **§3.2 — ROSCO OL_Mode + `OLInput_ROSCO.dat`.** Størst sjanse for å manipulere
   OpenFAST-momentet og eksitere side-to-side/fore-aft, uten rebuild. Start her.
2. Hvis OL_Mode gir svak/uren eksitasjon: vurder **§3.3 FMU-rebuild (VSContrl=4)**.
3. Hvis strukturell tårn-eksitasjon fortsatt ikke går: **§4 to turbiner**, og test
   **§4.1(a) multi-instans-hypotesen billig først**; fall tilbake på **§4.5**
   (turbin 2 = forenklet modell) om nødvendig.
4. **§5 (ikke-blokkerende):** fjern den overflødige ytre WT-droopen
   (`droop_enable=0`) og bruk UIC-ens innebygde droop (`perfect_tracking=0`, still
   `Ki`/`Kv`). Kjør egenverdi-sjekk etterpå.
5. Uansett utfall: torsjons-E→M-historien (§2.1) står allerede ferdig som et solid
   avhandlings-resultat.

---

## 7. Referanse — filer og kommandoer

**Konfigurasjon (leses ved runtime):**
- `test1002/ControlData/ROSCO.IEA15MW.IN` — ROSCO-parametre. OL_Mode linje 28;
  Open-Loop-blokk linje ~197–205. Nåværende: `VS_ControlMode=2`, `VSContrl=5`
  (i ServoDyn), `OL_Mode=0`.
- `test1002/IEA-15-240-RWT-Monopile_ServoDyn_wROSCO.dat` — `VSContrl=5` (linje 19).
  Backup: `.bak_VSContrl5`.
- `test1002/IEA-15-240-RWT-Monopile_InflowFile.dat` — vind (`WindType=1` steady 8 m/s).

**Sim-skript:**
- `casestudies/dyn_sim/test_WT_LEOGO_FMU_sim.py` — hoved-FMU-sim (SS via `YawBrTAyp`,
  kun i `fast_debug.fmu`). CLI: `--t-end`, `--load-step-mw`, `--elecpwr-mod-*`, `--torque-mod-*`.
- `casestudies/dyn_sim/test_WT_LEOGO_FMU_torsional_resonance_sim.py` — FMU torsjon (3.49 Hz).
- `casestudies/dyn_sim/test_WT_LEOGO_torsional_resonance_sim.py` — forenklet torsjon (E→M, virker).
- `casestudies/dyn_sim/test_WT_LEOGO_process_load_excitation_sim.py` — realistisk prosesslast.
- `casestudies/modal_analysis/interaction_WT_LEOGO.py`, `participation_WT_LEOGO_torsional.py` — modal.

**Analyse/plott:**
- `_ss_analysis.py` (repo-rot) — SS-envelope/FFT.
- `casestudies/dyn_sim/plotting/plot_WT_LEOGO_FMU_torsional_resonance.py` m.fl.

**LaTeX:**
- `docs/forced_response.tex` (`\label{sec:forcedresponse}`), `docs/openfast_excitation.tex`,
  `docs/frequency_analysis.tex`, `docs/network_excitation.tex`.

**FMU-filer:** `fast.fmu` (rask, release, ~100 steg/s, har `YawBrTAxp` men IKKE
`YawBrTAyp`), `fast_debug.fmu` (treg debug, har `YawBrTAyp` = SS — bruk denne for SS).

**Terminal-huskeregler (Windows PowerShell):**
- Kjed kommandoer med `;`, ALDRI `&&`. Terminaler starter i FORELDER-mappa →
  `Set-Location "tops-openfast-sommer2026"` først.
- IKKE bruk `Tee-Object` med piped python (buffrer til prosessen dør).
- Nøkkelfakta: nesten alle OpenFAST-inputfiler i `test1002/` leses ved runtime →
  kan endres uten rebuild.
