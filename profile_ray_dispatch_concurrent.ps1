param([int]$JobCount=4,[int]$Batches=2000,[switch]$Split,
      [int]$GpuRays=10000,[int]$GpuWindow=30000,[int]$GpuBatches=0,
      [int]$Grid=1,[int]$Bands=4,[string]$OutputDirectory='')
$ErrorActionPreference = 'Stop'
if ($JobCount -lt 1 -or $JobCount -gt 8 -or $Batches -lt 1) {
    throw 'JobCount must be 1..8 and Batches must be positive'
}
$pythonExe = (Get-Command python).Source
$root = if ($OutputDirectory) { [IO.Path]::GetFullPath($OutputDirectory) } else { Join-Path $env:TEMP 'ray-dispatch-profile' }
$work = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Path $root -Force | Out-Null
$jobs = @()
$names = @('a','b','c','d','e','f','g','h') | Select-Object -First $JobCount
foreach ($name in $names) {
    $mode = if ($Split) { if ($name -in @('a','b','c','d')) {'gpu'} else {'cpu'} } else {'mixed'}
    $rays = if ($mode -eq 'gpu') {$GpuRays} else {10000}
    $window = if ($mode -eq 'gpu') {$GpuWindow} else {30000}
    $runs = if ($mode -eq 'gpu' -and $GpuBatches -gt 0) {$GpuBatches} else {$Batches}
    $arguments = @("$work\profile_ray_dispatch_saturation.py", '--rays', "$rays", '--batches',
                   "$runs", '--window', "$window", '--mode', $mode, '--job', $name,
                   '--record-budget', '100000000', '--grid', "$Grid", '--bands', "$Bands")
    $jobs += Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $work `
        -RedirectStandardOutput "$root\ray_concurrent_$name.out" `
        -RedirectStandardError "$root\ray_concurrent_$name.err" -WindowStyle Hidden -PassThru
}
$samples = [System.Collections.Generic.List[object]]::new()
$began = [System.Diagnostics.Stopwatch]::StartNew()
Start-Sleep -Seconds 1
$pidPattern = 'pid_(' + (($jobs | ForEach-Object { $_.Id }) -join '|') + ')_'
$enginePaths = @((Get-Counter -ListSet 'GPU Engine').PathsWithInstances |
    Where-Object { $_ -match $pidPattern -and $_ -match 'engtype_(3D|Compute)' })
Write-Output "gpu_engine_paths=$($enginePaths.Count)"
while ($true) {
    foreach ($job in $jobs) { $job.Refresh() }
    $alive = @($jobs | Where-Object { -not $_.HasExited })
    $gpuLines = & nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    foreach ($line in $gpuLines) {
        $parts = $line -split ','
        if ($parts.Count -ge 3) {
            $samples.Add([pscustomobject]@{seconds=$began.Elapsed.TotalSeconds;
                gpu=[int]$parts[0].Trim();util=[int]$parts[1].Trim();
                memory_mb=[int]$parts[2].Trim();alive=$alive.Count})
        }
    }
    if ($enginePaths.Count) {
        try {
            $engineSample = @(Get-Counter -Counter $enginePaths -SampleInterval 1 -MaxSamples 2 -ErrorAction Stop)
            foreach ($value in $engineSample[-1].CounterSamples) {
                if ($value.InstanceName -match $pidPattern) {
                    $samples.Add([pscustomobject]@{seconds=$began.Elapsed.TotalSeconds;
                        gpu=$value.InstanceName;util=[math]::Round($value.CookedValue,2);
                        memory_mb=0;alive=$alive.Count;raw=$value.RawValue;type=$value.CounterType})
                }
            }
        } catch { Write-Output "gpu_engine_sample_error=$($_.Exception.Message)" }
    }
    if ($alive.Count -eq 0) { break }
    if ($began.Elapsed.TotalSeconds -gt 90) {
        foreach ($job in $alive) { Stop-Process -Id $job.Id -Force }
        throw 'concurrent ray probe exceeded 90 seconds'
    }
    Start-Sleep -Milliseconds 250
}
$samples | Export-Csv "$root\ray_concurrent_gpu_samples.csv" -NoTypeInformation
foreach ($job in $jobs) { Write-Output "pid=$($job.Id) exit=$($job.ExitCode)" }
$samples | Group-Object gpu | ForEach-Object {
    $values = @($_.Group | Where-Object { $_.alive -eq $JobCount -and [double]$_.util -ge 0 -and [double]$_.util -le 100 } | ForEach-Object { $_.util })
    if ($values.Count) {
        Write-Output "gpu=$($_.Name) simultaneous_samples=$($values.Count) mean=$([math]::Round(($values | Measure-Object -Average).Average,1)) peak=$(($values | Measure-Object -Maximum).Maximum)"
    }
}
if (@($jobs | Where-Object { $_.ExitCode -ne 0 }).Count) {
    throw 'one or more concurrent ray jobs failed; inspect their .err and .out files'
}
