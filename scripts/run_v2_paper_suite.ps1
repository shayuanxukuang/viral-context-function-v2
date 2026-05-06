param(
    [string]$GpuIds = "4,5,6",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
if (-not $env:TOKENIZERS_PARALLELISM) {
    $env:TOKENIZERS_PARALLELISM = "false"
}

$cmd = @("-u", ".\scripts\run_v2_paper_suite.py", "--gpu-ids", $GpuIds)
if ($RemainingArgs) {
    $cmd += $RemainingArgs
}

python @cmd
