# CHTD web console: one-shot startup (backend only; frontend dist already built).
$ErrorActionPreference = "Stop"
$web = Split-Path -Parent $MyInvocation.MyCommand.Path
$be  = Join-Path $web "backend"

# Optional local overrides. `web/.env` holds KEY=VALUE lines and is git-ignored, so a machine can
# point the student route at a model it actually serves without editing any code:
#
#   CHTD_STUDENT_URL=http://100.84.151.62:8003/v1
#   CHTD_STUDENT_MODEL_ID=qwen3-4b-instruct-2507
#
# Values already present in the environment win, so a shell can still override the file.
$envFile = Join-Path $web ".env"
if (Test-Path $envFile) {
    Write-Host "[start] loading $envFile"
    foreach ($line in Get-Content $envFile) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith("#") -or -not $t.Contains("=")) { continue }
        $i = $t.IndexOf("=")
        $k = $t.Substring(0, $i).Trim()
        $v = $t.Substring($i + 1).Trim().Trim('"').Trim("'")
        if ($k -and -not (Test-Path "Env:$k")) { Set-Item -Path "Env:$k" -Value $v }
    }
}

Set-Location $be

if (-not (Test-Path "$be\.venv\Scripts\python.exe")) {
    Write-Host "[start] creating venv..."
    python -m venv .venv
}
$py = "$be\.venv\Scripts\python.exe"

Write-Host "[start] installing deps (tsinghua mirror)..."
& $py -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi "uvicorn[standard]" pydantic httpx pytest

if (-not (Test-Path "$be\data\summary.json")) {
    Write-Host "[start] exporting data/*.json..."
    & $py export_data.py
}

$dist = "$web\frontend\dist\index.html"
$fe = Join-Path $web "frontend"
if (-not (Test-Path $dist)) {
    Write-Host "[start] frontend dist missing -> installing and building (npmmirror)..."
    $npm = if ($env:CHTD_NPM) { $env:CHTD_NPM } else { "npm" }
    Push-Location $fe
    if (-not (Test-Path "$fe\node_modules")) { & $npm install }
    & $npm run build
    Pop-Location
}
if (Test-Path $dist) {
    Write-Host "[start] frontend dist found -> hosting it at /"
} else {
    Write-Host "[start] frontend dist still missing -> backend serves placeholder at / (no crash)"
}

$port = if ($env:CHTD_BACKEND_PORT) { $env:CHTD_BACKEND_PORT } else { "8088" }
Write-Host "[start] uvicorn on http://127.0.0.1:$port  (health: /api/health)"
& $py -m uvicorn app.main:app --host 127.0.0.1 --port $port
