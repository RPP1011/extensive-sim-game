param(
    [ValidateSet("single", "sequence", "safe-sequence", "hub-stages")]
    [string]$Mode = "single",
    [string]$OutDir = "generated/screenshots/windows",
    [int]$Steps = 30,
    [int]$Every = 1,
    [int]$WarmupFrames = 3,
    [switch]$Hub,
    [switch]$Persist
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "cargo was not found in PATH. Run from a Windows shell with Rust installed."
}

if ($Every -lt 1) {
    Write-Error "-Every must be >= 1"
}
if ($Steps -lt 1) {
    Write-Error "-Steps must be >= 1"
}
if ($WarmupFrames -lt 0) {
    Write-Error "-WarmupFrames must be >= 0"
}
if ($Hub -and $Mode -eq "sequence") {
    Write-Error "Hub+sequence is currently unbounded with --steps. Use -Mode single -Hub, or run sequence without -Hub."
}
if ($Hub -and $Mode -eq "safe-sequence") {
    Write-Error "Hub+safe-sequence is not supported. Use -Mode single -Hub."
}
if ($Mode -eq "hub-stages" -and -not $Hub) {
    $Hub = $true
}

$captureDir = $OutDir
if (-not $Persist) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $captureDir = Join-Path "generated/screenshots" "_tmp_windows_capture_${stamp}_$PID"
    Write-Host "Ephemeral capture (auto-cleanup): $captureDir" -ForegroundColor Yellow
}

$exitCode = 0

try {
$commonArgs = @("run", "--")
if ($Hub) {
    $commonArgs += "--hub"
}

if ($Mode -eq "single") {
    $args = $commonArgs + @(
        "--screenshot", $captureDir,
        "--screenshot-warmup-frames", "$WarmupFrames"
    )
    Write-Host "Running: cargo $($args -join ' ')" -ForegroundColor Cyan
    & cargo @args
    $exitCode = $LASTEXITCODE
} elseif ($Mode -eq "sequence") {
    $args = $commonArgs + @(
        "--steps", "$Steps",
        "--screenshot-sequence", $captureDir,
        "--screenshot-every", "$Every",
        "--screenshot-warmup-frames", "$WarmupFrames"
    )
    Write-Host "Running: cargo $($args -join ' ')" -ForegroundColor Cyan
    & cargo @args
    $exitCode = $LASTEXITCODE
} elseif ($Mode -eq "safe-sequence") {
    New-Item -ItemType Directory -Force -Path $captureDir | Out-Null
    for ($i = 1; $i -le $Steps; $i++) {
        $captureIndex = $i - 1
        $tmpDir = Join-Path $captureDir "_tmp_step_$i"
        if (Test-Path $tmpDir) {
            Remove-Item -Recurse -Force $tmpDir
        }
        $args = $commonArgs + @(
            "--steps", "$i",
            "--screenshot", $tmpDir,
            "--screenshot-warmup-frames", "$WarmupFrames"
        )
        Write-Host "Running: cargo $($args -join ' ')" -ForegroundColor Cyan
        & cargo @args
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "capture step $i failed (exit $LASTEXITCODE), continuing"
            continue
        }

        $srcPng = Join-Path $tmpDir "frame_00000.png"
        $srcJson = Join-Path $tmpDir "frame_00000.json"
        $dstPng = Join-Path $captureDir ("frame_{0:D5}.png" -f $captureIndex)
        $dstJson = Join-Path $captureDir ("frame_{0:D5}.json" -f $captureIndex)

        if (Test-Path $srcPng) {
            Move-Item -Force $srcPng $dstPng
        } else {
            Write-Warning "PNG missing for step $i ($srcPng)"
        }
        if (Test-Path $srcJson) {
            Move-Item -Force $srcJson $dstJson
        } else {
            Write-Warning "JSON missing for step $i ($srcJson)"
        }
        if (Test-Path $tmpDir) {
            Remove-Item -Recurse -Force $tmpDir
        }
    }
    $exitCode = 0
} else {
    $args = @(
        "run", "--",
        "--hub",
        "--screenshot-hub-stages", $captureDir,
        "--screenshot-warmup-frames", "$WarmupFrames"
    )
    Write-Host "Running: cargo $($args -join ' ')" -ForegroundColor Cyan
    & cargo @args
    $exitCode = $LASTEXITCODE
}
}
finally {
    if ($Persist) {
        Write-Host "Screenshots kept at: $captureDir" -ForegroundColor Green
    } elseif (Test-Path $captureDir) {
        Remove-Item -Recurse -Force $captureDir
    }
}

exit $exitCode
