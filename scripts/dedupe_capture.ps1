param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $OutDir)) {
    Write-Error "Directory not found: $OutDir"
}

$dir = Resolve-Path $OutDir
$pngs = Get-ChildItem -Path $dir -Filter "frame_*.png" | Sort-Object Name
$jsons = Get-ChildItem -Path $dir -Filter "frame_*.json" | Sort-Object Name

# Remove orphan JSON files (no PNG sibling)
foreach ($j in $jsons) {
    $pngPath = [System.IO.Path]::ChangeExtension($j.FullName, ".png")
    if (-not (Test-Path $pngPath)) {
        Remove-Item -Force $j.FullName
        Write-Host "removed orphan json: $($j.Name)"
    }
}

# Remove exact-duplicate PNGs by SHA256 hash and matching JSON
$seen = @{}
$removed = 0
$kept = 0

$pngs = Get-ChildItem -Path $dir -Filter "frame_*.png" | Sort-Object Name
foreach ($p in $pngs) {
    $hash = (Get-FileHash -Algorithm SHA256 -Path $p.FullName).Hash
    if ($seen.ContainsKey($hash)) {
        $jsonPath = [System.IO.Path]::ChangeExtension($p.FullName, ".json")
        Remove-Item -Force $p.FullName
        if (Test-Path $jsonPath) {
            Remove-Item -Force $jsonPath
        }
        $removed += 1
        Write-Host "removed duplicate: $($p.Name) (same as $($seen[$hash]))"
    } else {
        $seen[$hash] = $p.Name
        $kept += 1
    }
}

Write-Host "dedupe complete: kept=$kept removed=$removed"
