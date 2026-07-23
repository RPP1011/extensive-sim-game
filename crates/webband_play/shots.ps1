# S12 — bounded launch + OS-level window capture for `webband_play`.
#
# Follows S8's harness shape exactly: launch the binary, poll for its window,
# screenshot the window rect with GetWindowRect + CopyFromScreen at a list of
# elapsed times, and ALWAYS close it in a `finally` (CloseMainWindow, then
# Kill). Nothing is ever left running: the binary also closes itself via
# `--exit-after-secs`, so there are two independent bounds.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File crates/webband_play/shots.ps1 `
#       -OutDir target/s12_shots -Seconds 60 -Shots "6,20,34,48" -PlayArgs "--speed 3 --force-raid-day 2"

param(
  [string]$OutDir = "target/s12_shots",
  [double]$Seconds = 60,
  [string]$Shots = "8,20,32,44",
  # If > 0, ignore -Shots and capture every $Interval seconds until -Seconds.
  # A raid's whole visible life (muster at the entry arc -> charge -> fight ->
  # pool reset) is ~15 s of wall clock, so a regular sweep is the reliable way
  # to have frames of it rather than betting on one timestamp.
  [double]$Interval = 0,
  [string]$PlayArgs = "",
  [string]$Exe = "target/debug/webband_play.exe"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  public struct RECT { public int L, T, R, B; }
}
"@

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if ($Interval -gt 0) {
  $times = @()
  for ($t = $Interval; $t -lt $Seconds; $t += $Interval) { $times += $t }
} else {
  $times = $Shots.Split(",") | ForEach-Object { [double]$_ }
}

$argList = @("--exit-after-secs", "$Seconds")
if ($PlayArgs -ne "") { $argList += $PlayArgs.Split(" ") }
Write-Host "[shots] launching $Exe $($argList -join ' ')"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $Exe).Path
$psi.Arguments = ($argList -join " ")
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.WorkingDirectory = (Get-Location).Path

$p = [System.Diagnostics.Process]::Start($psi)
$outFile = Join-Path $OutDir "stdout.txt"
$errFile = Join-Path $OutDir "stderr.txt"
$outTask = $p.StandardOutput.ReadToEndAsync()
$errTask = $p.StandardError.ReadToEndAsync()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

try {
  foreach ($t in $times) {
    while ($sw.Elapsed.TotalSeconds -lt $t -and -not $p.HasExited) { Start-Sleep -Milliseconds 150 }
    if ($p.HasExited) { Write-Host "[shots] process exited before t=$t"; break }
    $p.Refresh()
    $h = $p.MainWindowHandle
    if ($h -eq [IntPtr]::Zero) { Write-Host "[shots] no window yet at t=$t"; continue }
    [void][W]::SetForegroundWindow($h)
    Start-Sleep -Milliseconds 250
    $r = New-Object W+RECT
    [void][W]::GetWindowRect($h, [ref]$r)
    $w = $r.R - $r.L; $hgt = $r.B - $r.T
    if ($w -le 0 -or $hgt -le 0) { Write-Host "[shots] bad rect at t=$t"; continue }
    $bmp = New-Object System.Drawing.Bitmap $w, $hgt
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
    $name = "t{0:d3}s.png" -f [int]$t
    $path = Join-Path $OutDir $name
    $bmp.Save((Join-Path (Get-Location) $path), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    Write-Host "[shots] wrote $path ($w x $hgt)"
  }
  $deadline = $Seconds + 20
  while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $deadline) { Start-Sleep -Milliseconds 200 }
}
finally {
  if (-not $p.HasExited) {
    Write-Host "[shots] closing the window"
    [void]$p.CloseMainWindow()
    if (-not $p.WaitForExit(5000)) { Write-Host "[shots] killing"; $p.Kill() }
  }
  $p.WaitForExit()
  Set-Content -Path $outFile -Value $outTask.Result -Encoding utf8
  Set-Content -Path $errFile -Value $errTask.Result -Encoding utf8
  Write-Host "[shots] exit code $($p.ExitCode); logs in $OutDir"
}
