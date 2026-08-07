$patterns = @(
  "PaDEL-Descriptor",
  "pfas_padel_rerun_46_home",
  "padel_cli_one_by_one.py"
)

$query = "Name = 'java.exe' OR Name = 'python.exe'"
$matches = Get-CimInstance -ClassName Win32_Process -Filter $query | Where-Object {
  $cmd = $_.CommandLine
  $cmd -and ($patterns | Where-Object { $cmd -like "*$_*" })
}

if (-not $matches) {
  Write-Host "No PaDEL-related java/python process found."
  exit 0
}

$matches | Select-Object ProcessId, ParentProcessId, Name, CommandLine | Format-List

foreach ($p in $matches) {
  Write-Host "Terminating PID $($p.ProcessId): $($p.Name)"
  Invoke-CimMethod -InputObject $p -MethodName Terminate | Out-Null
}

Start-Sleep -Seconds 2
Write-Host "Done."
