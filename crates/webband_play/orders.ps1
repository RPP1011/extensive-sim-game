# S13 — evidence that THE PLAYER'S HANDS reach the sim, and that the guild
# layer is on screen.
#
# Same harness shape as S12's controls.ps1: launch the binary bounded, drive
# the REAL Windows message queue with SendKeys (the path a human's keyboard
# takes into winit), screenshot the window around each press, and ALWAYS close
# it in a `finally`. The binary also closes itself on --exit-after-secs, so
# there are two independent bounds and nothing is ever left running.
#
#   a_start      the colony working, politics on the HUD (ASK / POWERS / ARC)
#   b_selected   [N] cycled the selection — SELECTED names a colonist
#   c_trade      [V] gave them a standing work order
#   d_hold       [H] anchored them — SELECTED reads "HOLD"
#   e_raid       the warband at the fences
#   f_focus      [F] named a raider for the line
#   g_answer     [9] refused the open ask (or printed why it could not)
#
# NOTE ON KEYS: selection is bound to BOTH `Tab` and `n`. The harness presses
# `n`, because egui owns Tab for focus navigation and a consumed key would make
# this script measure egui rather than the game.

param(
  [string]$OutDir = "target/s13_shots/orders",
  [double]$Seconds = 105,
  [string]$PlayArgs = "--speed 2 --force-raid-day 2",
  [string]$Exe = "target/debug/webband_play.exe"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W3 {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  public struct RECT { public int L, T, R, B; }
}
"@

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $Exe).Path
$psi.Arguments = "--exit-after-secs $Seconds $PlayArgs"
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.WorkingDirectory = (Get-Location).Path
$p = [System.Diagnostics.Process]::Start($psi)
$outTask = $p.StandardOutput.ReadToEndAsync()
$errTask = $p.StandardError.ReadToEndAsync()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

function Shoot([string]$name) {
  $p.Refresh()
  $h = $p.MainWindowHandle
  if ($h -eq [IntPtr]::Zero) { Write-Host "[orders] no window for $name"; return }
  [void][W3]::SetForegroundWindow($h)
  Start-Sleep -Milliseconds 300
  $r = New-Object W3+RECT
  [void][W3]::GetWindowRect($h, [ref]$r)
  $bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
  $bmp.Save((Join-Path (Get-Location) (Join-Path $OutDir "$name.png")), [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bmp.Dispose()
  Write-Host "[orders] wrote $OutDir/$name.png at t=$([int]$sw.Elapsed.TotalSeconds)s"
}

function Press([string]$keys) {
  if ($p.HasExited) { return }
  $p.Refresh()
  [void][W3]::SetForegroundWindow($p.MainWindowHandle)
  Start-Sleep -Milliseconds 250
  [System.Windows.Forms.SendKeys]::SendWait($keys)
  Start-Sleep -Milliseconds 500
}

function WaitUntil([double]$t) {
  while ($sw.Elapsed.TotalSeconds -lt $t -and -not $p.HasExited) { Start-Sleep -Milliseconds 150 }
}

try {
  WaitUntil 16
  Shoot "a_start"
  Press "n"; Press "n"
  Shoot "b_selected"
  Press "v"
  Shoot "c_trade"
  Press "h"
  Shoot "d_hold"
  Press "0"          # the guild report -> stdout
  WaitUntil 46
  Shoot "e_raid"
  Press "f"          # focus a raider
  Shoot "f_focus"
  Press "9"          # refuse the open ask (or print why not)
  Shoot "g_answer"
  Press "0"
  WaitUntil ($Seconds + 15)
}
finally {
  if (-not $p.HasExited) {
    Write-Host "[orders] closing the window"
    [void]$p.CloseMainWindow()
    if (-not $p.WaitForExit(5000)) { $p.Kill() }
  }
  $p.WaitForExit()
  Set-Content -Path (Join-Path $OutDir "stdout.txt") -Value $outTask.Result -Encoding utf8
  Set-Content -Path (Join-Path $OutDir "stderr.txt") -Value $errTask.Result -Encoding utf8
  Write-Host "[orders] exit code $($p.ExitCode); logs in $OutDir"
}
