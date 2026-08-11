<#
Build a ZeroMQ-enabled ROSCO controller (libDISCON.dll) for the OpenFAST FMU.

WHY
  The FMU's bundled ROSCO (test1002/ControlData/libDISCON.dll, ROSCO 2.10.1) was
  compiled WITHOUT the ZMQ client. ROSCO's CMake only enables ZMQ when libzmq is
  discoverable via pkg-config:
      pkg_check_modules(PC_ZeroMQ libzmq)
      if(PC_ZeroMQ_FOUND) -> add_definitions(-DZMQ_CLIENT="TRUE") + zmq_client.c
  With libzmq absent at build time, ZMQ_Mode>0 fails at runtime with
  "ZeroMQ client has not been properly installed" (and crashes the co-sim).

  Rebuilding ROSCO with libzmq present auto-enables the ZMQ client, giving a
  LIVE grid->generator-torque path via the rosco_zmq_server.py bridge -- WITHOUT
  rebuilding the OpenFAST FMU wrapper (only the controller DLL is swapped).

TOOLCHAIN (installed here via winget + conda-forge)
  Miniforge (conda), then a build env with:
    fortran-compiler, c-compiler, cmake, make, pkg-config, zeromq, git

USAGE (run phases in order; each phase is re-runnable)
  powershell -ExecutionPolicy Bypass -File build_rosco_zmq.ps1 -Phase deps
  powershell -ExecutionPolicy Bypass -File build_rosco_zmq.ps1 -Phase env
  powershell -ExecutionPolicy Bypass -File build_rosco_zmq.ps1 -Phase clone
  powershell -ExecutionPolicy Bypass -File build_rosco_zmq.ps1 -Phase build
  powershell -ExecutionPolicy Bypass -File build_rosco_zmq.ps1 -Phase install
  powershell -ExecutionPolicy Bypass -File build_rosco_zmq.ps1 -Phase verify
#>
param(
    [ValidateSet("deps", "env", "clone", "build", "install", "verify")]
    [string]$Phase = "deps",
    [string]$RoscoTag = "v2.10.1",
    [string]$BuildRoot = "$env:USERPROFILE\rosco_build",
    [string]$EnvName = "roscobuild",
    [string]$TargetDll = "$PSScriptRoot\..\..\test1002\ControlData\libDISCON.dll"
)

$ErrorActionPreference = "Stop"
$RoscoSrc = Join-Path $BuildRoot "ROSCO"
$CtrlDir = Join-Path $RoscoSrc "rosco\controller"
$CtrlBuild = Join-Path $CtrlDir "build"

function Find-Conda {
    $cands = @(
        "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\miniforge3\Scripts\conda.exe"
    )
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    $g = Get-Command conda -ErrorAction SilentlyContinue
    if ($g) { return $g.Source }
    return $null
}

switch ($Phase) {

    "deps" {
        Write-Host "== Phase deps: ensure Miniforge (conda) ==" -ForegroundColor Cyan
        $conda = Find-Conda
        if ($conda) {
            Write-Host "conda already present: $conda"
        }
        else {
            Write-Host "Installing Miniforge3 via winget..."
            winget install --id CondaForge.Miniforge3 -e `
                --accept-source-agreements --accept-package-agreements
            $conda = Find-Conda
        }
        if (-not $conda) { throw "conda not found after install; open a new shell and re-run." }
        Write-Host "OK. conda = $conda"
    }

    "env" {
        Write-Host "== Phase env: create build env '$EnvName' ==" -ForegroundColor Cyan
        $conda = Find-Conda; if (-not $conda) { throw "Run -Phase deps first." }
        & $conda create -y -n $EnvName -c conda-forge `
            fortran-compiler c-compiler cmake make pkg-config zeromq git
        Write-Host "OK. Env '$EnvName' created."
    }

    "clone" {
        Write-Host "== Phase clone: ROSCO $RoscoTag ==" -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
        if (Test-Path $RoscoSrc) {
            Write-Host "ROSCO already cloned at $RoscoSrc"
        }
        else {
            git clone --depth 1 --branch $RoscoTag https://github.com/NREL/ROSCO.git $RoscoSrc
        }
        Write-Host "OK. Source at $RoscoSrc"
    }

    "build" {
        Write-Host "== Phase build: cmake + make (ZMQ auto-detected) ==" -ForegroundColor Cyan
        $conda = Find-Conda; if (-not $conda) { throw "Run -Phase deps first." }
        if (-not (Test-Path $CtrlDir)) { throw "Run -Phase clone first." }
        New-Item -ItemType Directory -Force -Path $CtrlBuild | Out-Null
        # Run cmake/make INSIDE the conda env so compilers + libzmq .pc are on PATH.
        & $conda run -n $EnvName cmake -S $CtrlDir -B $CtrlBuild -G "Unix Makefiles" `
            -DCMAKE_BUILD_TYPE=Release
        & $conda run -n $EnvName cmake --build $CtrlBuild --config Release -j
        Write-Host "OK. Build complete. Look for 'Found ZeroMQ' / -DZMQ_CLIENT above."
    }

    "install" {
        Write-Host "== Phase install: copy new libdiscon into test1002 ==" -ForegroundColor Cyan
        $built = Get-ChildItem -Path $RoscoSrc -Recurse -Include "libdiscon.dll", "libDISCON.dll" `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $built) { throw "Built libdiscon.dll not found; check the build phase output." }
        $target = (Resolve-Path $TargetDll -ErrorAction SilentlyContinue)
        if (-not $target) { $target = $TargetDll }
        if (Test-Path $target) {
            Copy-Item $target "$target.bak_nozmq" -Force
            Write-Host "Backed up existing -> $target.bak_nozmq"
        }
        Copy-Item $built.FullName $target -Force
        Write-Host "OK. Installed $($built.FullName) -> $target"
    }

    "verify" {
        Write-Host "== Phase verify: ZMQ client present in the DLL? ==" -ForegroundColor Cyan
        $target = (Resolve-Path $TargetDll).Path
        $bytes = [System.IO.File]::ReadAllBytes($target)
        $txt = [System.Text.Encoding]::ASCII.GetString($bytes)
        # The C zmq_client string only exists when zmq_client.c was compiled in.
        if ($txt -match "Connecting to ZeroMQ server at" -or $txt -match "zmq_ctx_new") {
            Write-Host "OK: ZMQ client IS compiled in (found zmq_client runtime strings)." -ForegroundColor Green
        }
        else {
            Write-Host "WARNING: no zmq_client runtime strings found; ZMQ may not be enabled." -ForegroundColor Yellow
            Write-Host "Check the build log for 'PC_ZeroMQ_FOUND' / that libzmq was located by pkg-config."
        }
    }
}
