$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root "vendor"
$ffmpegDir = Join-Path $vendorDir "ffmpeg_bin"
$zipPath = Join-Path $vendorDir "ffmpeg-release-essentials.zip"
$extractDir = Join-Path $vendorDir "ffmpeg_extract"
$downloadUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

if ((Test-Path (Join-Path $ffmpegDir "ffmpeg.exe")) -and (Test-Path (Join-Path $ffmpegDir "ffprobe.exe"))) {
    Write-Host "Vendored ffmpeg already present at $ffmpegDir"
    exit 0
}

try {
    $localFfmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
    $localFfprobe = (Get-Command ffprobe -ErrorAction Stop).Source
    if ($localFfmpeg -and $localFfprobe -and (Test-Path $localFfmpeg) -and (Test-Path $localFfprobe)) {
        Write-Host "Using locally installed ffmpeg binaries..."
        New-Item -ItemType Directory -Force -Path $ffmpegDir | Out-Null
        Copy-Item -LiteralPath $localFfmpeg -Destination (Join-Path $ffmpegDir "ffmpeg.exe") -Force
        Copy-Item -LiteralPath $localFfprobe -Destination (Join-Path $ffmpegDir "ffprobe.exe") -Force
        Write-Host "Vendored ffmpeg saved to $ffmpegDir"
        exit 0
    }
} catch {
    Write-Host "Local ffmpeg not available; falling back to download."
}

Write-Host "Downloading ffmpeg bundle..."
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

if (Test-Path $extractDir) {
    Remove-Item -LiteralPath $extractDir -Recurse -Force
}
Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force

$binDir = Get-ChildItem -Path $extractDir -Directory -Recurse |
    Where-Object { (Test-Path (Join-Path $_.FullName "ffmpeg.exe")) -and (Test-Path (Join-Path $_.FullName "ffprobe.exe")) } |
    Select-Object -First 1

if (-not $binDir) {
    throw "Could not find ffmpeg.exe and ffprobe.exe in extracted archive."
}

if (Test-Path $ffmpegDir) {
    Remove-Item -LiteralPath $ffmpegDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ffmpegDir | Out-Null
Copy-Item -LiteralPath (Join-Path $binDir.FullName "ffmpeg.exe") -Destination (Join-Path $ffmpegDir "ffmpeg.exe") -Force
Copy-Item -LiteralPath (Join-Path $binDir.FullName "ffprobe.exe") -Destination (Join-Path $ffmpegDir "ffprobe.exe") -Force

Remove-Item -LiteralPath $extractDir -Recurse -Force
Remove-Item -LiteralPath $zipPath -Force

Write-Host "Vendored ffmpeg saved to $ffmpegDir"
