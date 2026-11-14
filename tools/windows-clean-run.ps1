<#
.SYNOPSIS
  The Windows clean run (docs/internals/WINDOWS-RUNBOOK.md, sections 1-3) on a pinned commit, unattended.

.DESCRIPTION
  1. Makes a detached worktree of -Commit next to this checkout (..\bvh-clean-run), installs and builds it,
     so edits in this checkout during the run change nothing.
  2. Waits until the machine is quiet: no process whose command line names any of -WaitFor (default: the
     downsampler), and the NVIDIA GPU under -IdleUtil % for -IdleMinutes minutes in a row.
  3. Correctness pages per GPU (runbook section 1 plus query-oracle, trace-cpu-agreement and any page
     named in -ExtraPages). A failing trace-oracle or trace-inline on a GPU skips that GPU's benches.
  4. scene-trees-sponza memory on the RTX (section 3).
  5. The paired surface-ray benches on the RTX (section 2), -BenchRuns runs each, waiting for quiet again
     before every run. Rows go to <out>\trace-rows.csv (not appended to bench/results/trace.csv: check
     the device column, then append by hand with surface-to-csv.ts --append).
  6. Example smoke pages on the RTX.
  Everything lands in -OutDir (default bench\runs\<date>-<commit>), with summary.md listing each step.

.EXAMPLE
  pwsh tools\windows-clean-run.ps1 -Commit main
  pwsh tools\windows-clean-run.ps1 -Commit main -SkipIntel -SkipBenches   # correctness on the RTX only
#>
param(
    [string]$Commit = 'main',
    [string[]]$WaitFor = @('downsampler'),
    [int]$IdleUtil = 10,
    [int]$IdleMinutes = 5,
    [int]$BenchRuns = 3,
    [string[]]$ExtraPages = @(),
    [switch]$SkipIntel,
    [switch]$SkipBenches,
    [switch]$SkipSmoke,
    [switch]$NoWait,
    [string]$AssetsDir = 'C:\Users\denni\Documents\dev\webgpu-ddgi\assets',
    [string]$ManifestsDir = 'C:\Users\denni\Documents\dev\webgpu-ddgi\scenes\manifests',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'
$repo = (git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
$sha = (git -C $repo rev-parse --short $Commit).Trim()
$work = Join-Path (Split-Path $repo -Parent) 'bvh-clean-run'
if (-not $OutDir) { $OutDir = Join-Path $repo ("bench\runs\{0}-{1}" -f (Get-Date -Format 'yyyy-MM-dd'), $sha) }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$summary = Join-Path $OutDir 'summary.md'
"# Windows clean run, $sha`n`nStarted $(Get-Date -Format s). Worktree $work.`n" | Set-Content $summary

function Log([string]$line) { Write-Host $line; Add-Content $summary $line }

function Wait-Quiet([string]$why) {
    if ($NoWait) { return }
    Log "- waiting for quiet ($why) from $(Get-Date -Format T)"
    $quietSince = $null
    while ($true) {
        $busy = Get-CimInstance Win32_Process | Where-Object {
            $cmd = $_.CommandLine
            $cmd -and $_.ProcessId -ne $PID -and ($WaitFor | Where-Object { $cmd -match [regex]::Escape($_) })
        }
        $util = 0
        try { $util = [int]((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) | Select-Object -First 1) } catch {}
        if (-not $busy -and $util -lt $IdleUtil) {
            if (-not $quietSince) { $quietSince = Get-Date }
            if (((Get-Date) - $quietSince).TotalMinutes -ge $IdleMinutes) { break }
        } else { $quietSince = $null }
        Start-Sleep -Seconds 30
    }
    Log "  quiet at $(Get-Date -Format T)"
}

# Runs one runner invocation; returns $true on exit code 0. The log and JSON go to $OutDir.
function Run-Pages([string]$name, [string]$filter, [string]$gpu, [string]$query = '') {
    $log = Join-Path $OutDir "$name.log"
    $json = Join-Path $OutDir "$name.json"
    $env:BVH_PAGE_QUERY = $query
    $start = Get-Date
    Push-Location $work
    try {
        & node tests/gpu/run.ts --filter $filter --gpu $gpu --json $json *>&1 | Tee-Object -FilePath $log | Out-Host
        $ok = ($LASTEXITCODE -eq 0)
    } finally { Pop-Location; $env:BVH_PAGE_QUERY = '' }
    $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    Log ("- {0} {1} ({2}, {3} min)" -f ($(if ($ok) { 'PASS' } else { 'FAIL' })), $name, $gpu, $mins)
    return $ok
}

# 1. Pinned worktree, installed and built.
if (Test-Path $work) { git -C $repo worktree remove --force $work }
git -C $repo worktree add --detach $work $sha | Out-Null
Push-Location $work
try {
    pnpm install --frozen-lockfile *>&1 | Out-Null
    pnpm build *>&1 | Tee-Object -FilePath (Join-Path $OutDir 'build.log') | Out-Null
    if ($LASTEXITCODE -ne 0) { Log '- FAIL pnpm build; stopping.'; exit 1 }
    pnpm test *>&1 | Tee-Object -FilePath (Join-Path $OutDir 'unit.log') | Out-Null
    Log ("- {0} pnpm test" -f ($(if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' })))
} finally { Pop-Location }

$env:BVH_ASSETS_DIR = $AssetsDir
$env:BVH_MANIFESTS_DIR = $ManifestsDir

$correctness = @(
    'trace-oracle', 'trace-inline', 'query-oracle', 'trace-cpu-agreement',
    'bvh-lbvh,bvh-hploc,bvh-blas-refit,bvh-top-level-gpu,bvh-shared',
    'scene-trees-runtime,scene-trees-deforming,scene-trees-mid-swap',
    'three-trace,three-skinned'
) + $ExtraPages

$gpus = @('nvidia') + $(if ($SkipIntel) { @() } else { @('intel') })
$gates = @{}
foreach ($gpu in $gpus) {
    Wait-Quiet "correctness on $gpu"
    Log "`n## Correctness, $gpu`n"
    $gates[$gpu] = $true
    foreach ($filter in $correctness) {
        $name = "$gpu-" + ($filter -replace ',', '+')
        $ok = Run-Pages $name $filter $gpu
        if (-not $ok -and ($filter -eq 'trace-oracle' -or $filter -eq 'trace-inline')) { $gates[$gpu] = $false }
    }
}

if (-not $gates['nvidia']) {
    Log "`ntrace-oracle or trace-inline failed on the RTX: no memory page or benches (runbook section 1)."
} else {
    Wait-Quiet 'scene-trees-sponza'
    Log "`n## Memory (section 3)`n"
    Run-Pages 'nvidia-scene-trees-sponza' 'scene-trees-sponza' 'nvidia' | Out-Null

    if (-not $SkipBenches) {
        Log "`n## Paired benches (section 2), $BenchRuns runs each`n"
        $jsons = @()
        for ($i = 1; $i -le $BenchRuns; $i++) {
            foreach ($scene in @('sponza', 'bistro-interior')) {
                Wait-Quiet "bench $scene run $i"
                $name = "nvidia-surface-$scene-$i"
                Run-Pages $name "trace-bench-surface-$scene" 'nvidia' | Out-Null
                $jsons += (Join-Path $OutDir "$name.json")
            }
        }
        $existing = $jsons | Where-Object { Test-Path $_ }
        if ($existing) {
            Push-Location $work
            try {
                & node tests/gpu/pages/trace-bench/surface-to-csv.ts @existing --commit $sha --out (Join-Path $OutDir 'trace-rows.csv') *>&1 | Out-Host
            } finally { Pop-Location }
            Log '- rows: trace-rows.csv'
        }
    }
}

if (-not $SkipSmoke) {
    Wait-Quiet 'example smoke'
    Log "`n## Examples`n"
    Push-Location $work
    try {
        & node tests/gpu/run.ts --examples --gpu nvidia --json (Join-Path $OutDir 'examples.json') *>&1 | Tee-Object -FilePath (Join-Path $OutDir 'examples.log') | Out-Host
        Log ("- {0} example smoke (nvidia)" -f ($(if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' })))
    } finally { Pop-Location }
}

Log "`nFinished $(Get-Date -Format s). Report per runbook section 6; Dennis decides the defaults from the ratios."
