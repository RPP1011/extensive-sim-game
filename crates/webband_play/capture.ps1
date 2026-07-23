# S13 — window capture that also works when the desktop compositor will not
# hand GDI the client area of a Vulkan window.
#
# S12's harness used GetWindowRect + CopyFromScreen (a screen-region grab).
# That returned a blank white client area on this machine at S13 time, with
# every frame byte-identical, while the process was demonstrably alive (keys
# landed, the campaign advanced, stdout moved). PrintWindow with
# PW_RENDERFULLCONTENT (0x2) asks the window itself to draw into a DC and
# captures unredirected surfaces the screen grab misses. This script tries
# PrintWindow first and falls back to the screen grab, and says which it used.
#
# Usage:
#   powershell -File crates/webband_play/capture.ps1 -OutDir target/s13_shots/x `
#       -Seconds 120 -PlayArgs "--speed 2" -Script "16:shoot:a; 18:key:n; ..."
#
# -Script is a ';'-separated list of "<t-seconds>:<verb>:<arg>" steps, verbs
# `shoot` (screenshot named <arg>) and `key` (SendKeys <arg>).

param(
  [string]$OutDir = "target/s13_shots/cap",
  [double]$Seconds = 120,
  [string]$PlayArgs = "",
  [string]$Script = "20:shoot:a",
  [string]$Exe = "target/debug/webband_play.exe"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Drawing;
using System.Runtime.InteropServices;
public class WCap {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  public struct RECT { public int L, T, R, B; }
  // PrintWindow with PW_RENDERFULLCONTENT into a Bitmap. Returns null on failure.
  public static Bitmap Print(IntPtr h, int w, int hgt) {
    Bitmap bmp = new Bitmap(w, hgt);
    using (Graphics g = Graphics.FromImage(bmp)) {
      IntPtr hdc = g.GetHdc();
      bool ok = PrintWindow(h, hdc, 0x2);
      g.ReleaseHdc(hdc);
      if (!ok) { bmp.Dispose(); return null; }
    }
    return bmp;
  }
}
"@ -ReferencedAssemblies System.Drawing, System.Windows.Forms

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
  if ($h -eq [IntPtr]::Zero) { Write-Host "[cap] no window for $name"; return }
  [void][WCap]::SetForegroundWindow($h)
  Start-Sleep -Milliseconds 350
  $r = New-Object WCap+RECT
  [void][WCap]::GetWindowRect($h, [ref]$r)
  $w = $r.R - $r.L; $hgt = $r.B - $r.T
  if ($w -le 0 -or $hgt -le 0) { Write-Host "[cap] bad rect for $name"; return }
  $how = "PrintWindow"
  $bmp = [WCap]::Print($h, $w, $hgt)
  if ($bmp -eq $null) {
    $how = "CopyFromScreen"
    $bmp = New-Object System.Drawing.Bitmap $w, $hgt
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
    $g.Dispose()
  }
  $bmp.Save((Join-Path (Get-Location) (Join-Path $OutDir "$name.png")), [System.Drawing.Imaging.ImageFormat]::Png)
  $bmp.Dispose()
  Write-Host "[cap] $name via $how at t=$([int]$sw.Elapsed.TotalSeconds)s ($w x $hgt)"
}

function Press([string]$keys) {
  if ($p.HasExited) { return }
  $p.Refresh()
  [void][WCap]::SetForegroundWindow($p.MainWindowHandle)
  Start-Sleep -Milliseconds 250
  [System.Windows.Forms.SendKeys]::SendWait($keys)
  Start-Sleep -Milliseconds 400
}

try {
  foreach ($step in $Script.Split(";")) {
    $step = $step.Trim()
    if ($step -eq "") { continue }
    $parts = $step.Split(":")
    $t = [double]$parts[0]
    while ($sw.Elapsed.TotalSeconds -lt $t -and -not $p.HasExited) { Start-Sleep -Milliseconds 120 }
    if ($p.HasExited) { Write-Host "[cap] process exited before t=$t"; break }
    switch ($parts[1]) {
      "shoot" { Shoot $parts[2] }
      "key"   { Press $parts[2]; Write-Host "[cap] pressed '$($parts[2])' at t=$([int]$sw.Elapsed.TotalSeconds)s" }
    }
  }
  while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt ($Seconds + 20)) { Start-Sleep -Milliseconds 200 }
}
finally {
  if (-not $p.HasExited) {
    Write-Host "[cap] closing the window"
    [void]$p.CloseMainWindow()
    if (-not $p.WaitForExit(6000)) { $p.Kill() }
  }
  $p.WaitForExit()
  Set-Content -Path (Join-Path $OutDir "stdout.txt") -Value $outTask.Result -Encoding utf8
  Set-Content -Path (Join-Path $OutDir "stderr.txt") -Value $errTask.Result -Encoding utf8
  Write-Host "[cap] exit code $($p.ExitCode); logs in $OutDir"
}
