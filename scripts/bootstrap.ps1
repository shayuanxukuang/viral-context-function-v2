param(
    [string[]]$Group = @("mvp_core"),
    [switch]$Force,
    [switch]$ExtractArchives,
    [switch]$BuildTables,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

if (-not $SkipDownload) {
    $downloadArgs = @("scripts/download_data.py")
    foreach ($item in $Group) {
        $downloadArgs += @("--group", $item)
    }
    if ($Force) {
        $downloadArgs += "--force"
    }

    python @downloadArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($ExtractArchives) {
    $extractArgs = @("scripts/extract_archives.py")
    foreach ($item in $Group) {
        $extractArgs += @("--group", $item)
    }
    if ($Force) {
        $extractArgs += "--force"
    }
    python @extractArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($BuildTables) {
    python scripts\parse_refseq_viral.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    python scripts\normalize_taxonomy_hosts.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    python scripts\build_training_index.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

python scripts/build_inventory.py
exit $LASTEXITCODE
