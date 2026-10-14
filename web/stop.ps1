# Stop the CHTD backend (listens on 8088 by default; override with CHTD_BACKEND_PORT).
$port = if ($env:CHTD_BACKEND_PORT) { $env:CHTD_BACKEND_PORT } else { "8088" }
$conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    Write-Host "[stop] killing PID $($c.OwningProcess) on port $port"
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
}
if (-not $conns) { Write-Host "[stop] nothing listening on $port" }
