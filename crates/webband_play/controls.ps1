# S12 — evidence that the player's CONTROLS actually reach the sim.
#
# Launches `webband_play` bounded, then drives the real Windows message queue
# with SendKeys (the same path a human's keyboard takes into winit) and
# screenshots the HUD around each press:
#   t0  baseline           -> "speed x4    paused 0"
#   press SPACE            -> "paused 1" and the tick STOPS advancing
#   press SPACE again      -> "paused 0" and the tick moves again
#   press 3                -> "speed x16"
# The tick number is read out of the HUD text in the PNGs by eye; the process
# also prints every transition to stdout, which this script captures.

param(
  [string]$OutDir = "target/s12_shots/controls",
  [double]$Seconds = 46,
  [string]$Exe = "target/debug/webband_play.exe"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W2 {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  public struct RECT { public int L, T, R, B; }
}
"@

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $Exe).Path
$psi.Arguments = "--exit-after-secs $Seconds --speed 1"
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
  if ($h -eq [IntPtr]::Zero) { Write-Host "[controls] no window for $name"; return }
  [void][W2]::SetForegroundWindow($h)
  Start-Sleep -Milliseconds 300
  $r = New-Object W2+RECT
  [void][W2]::GetWindowRect($h, [ref]$r)
  $bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
  $bmp.Save((Join-Path (Get-Location) (Join-Path $OutDir "$name.png")), [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bmp.Dispose()
  Write-Host "[controls] wrote $OutDir/$name.png"
}

function Press([string]$keys) {
  $p.Refresh()
  [void][W2]::SetForegroundWindow($p.MainWindowHandle)
  Start-Sleep -Milliseconds 300
  [System.Windows.Forms.SendKeys]::SendWait($keys)
  Start-Sleep -Milliseconds 600
}

try {
  while ($sw.Elapsed.TotalSeconds -lt 14 -and -not $p.HasExited) { Start-Sleep -Milliseconds 200 }
  Shoot "a_running"
  Press " "            # space -> pause
  Shoot "b_paused"
  Start-Sleep -Seconds 4
  Shoot "c_paused_still"   # 4 s later the tick MUST be unchanged
  Press " "            # space -> resume
  Start-Sleep -Seconds 3
  Shoot "d_resumed"
  Press "3"            # speed x16
  Start-Sleep -Seconds 3
  Shoot "e_speed_x16"
  Press "c"            # dump the chronicle to stdout
  Start-Sleep -Seconds 1
  while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt ($Seconds + 20)) { Start-Sleep -Milliseconds 200 }
}
finally {
  if (-not $p.HasExited) {
    [void]$p.CloseMainWindow()
    if (-not $p.WaitForExit(5000)) { $p.Kill() }
  }
  $p.WaitForExit()
  Set-Content -Path (Join-Path $OutDir "stdout.txt") -Value $outTask.Result -Encoding utf8
  Set-Content -Path (Join-Path $OutDir "stderr.txt") -Value $errTask.Result -Encoding utf8
  Write-Host "[controls] exit code $($p.ExitCode)"
}
