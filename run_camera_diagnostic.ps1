param(
    [ValidateSet("quick", "evidence")]
    [string]$Preset = "quick",
    [ValidateRange(8, 2048)]
    [int]$Resolution = 0,
    [ValidateRange(1, 1000000)]
    [int]$RequestedPackages = 1,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

if ($Preset -eq "evidence") {
    $renderResolution = 64
    $packages = 4
    $pairBudget = 20000000
    $outDir = "exposures/camera_diagnostic_evidence_64"
} else {
    $renderResolution = 16
    $packages = 1
    $pairBudget = 2000000
    $outDir = "exposures/camera_diagnostic_quick_16"
}

if ($Resolution -gt 0) {
    $renderResolution = $Resolution
    $packages = $RequestedPackages
    $outDir = "exposures/camera_diagnostic_${renderResolution}"
}

Write-Host "Camera diagnostic configuration: ${renderResolution}x${renderResolution}, output=$outDir, requested-packages=$packages"

if ($PlanOnly) {
    return
}

python exposure_render_demo.py `
    --integrator bdpt `
    --scene-mode thick-lens-lab `
    --width $renderResolution `
    --height $renderResolution `
    --total-rays 10000 `
    --rays-per-batch 10000 `
    --frames 1 `
    --no-window `
    --gpu-resident `
    --save-files `
    --out-dir $outDir `
    --bdpt-native-packages $packages `
    --t5-pair-budget $pairBudget

if ($LASTEXITCODE -ne 0) {
    throw "Camera diagnostic failed with exit code $LASTEXITCODE"
}

Write-Host "Camera diagnostic complete: $outDir/0000_cpp.png"
Write-Host "Evidence report: $outDir/0000_cpp_summary.json"
