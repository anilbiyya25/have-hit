# Prompt for the Gemini API key, store it, and verify it immediately.
#
# Read-Host -AsSecureString means the key is never echoed to the screen and
# never written to PowerShell's history file, which is what would happen if it
# were typed as part of a normal command.
#
#   .\run.ps1                          <- not this; run it directly:
#   powershell -ExecutionPolicy Bypass -File scripts\set_api_key.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " HAVE HIT - Gemini API key setup" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Paste your Google AI Studio key below and press Enter."
Write-Host " Input is hidden. Right-click or Ctrl+V pastes." -ForegroundColor DarkGray
Write-Host ""

$secure = Read-Host " GEMINI_API_KEY" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr).Trim()
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Host "`n No key entered. Nothing was changed." -ForegroundColor Yellow
    Read-Host "`n Press Enter to close"
    exit 1
}

# Google AI Studio keys start "AIza". Warn but do not block -- key formats
# change, and refusing a valid key is worse than a soft warning.
if (-not $key.StartsWith("AIza")) {
    Write-Host "`n Note: that does not look like an AI Studio key (expected to start 'AIza')." -ForegroundColor Yellow
    Write-Host " Continuing anyway." -ForegroundColor DarkGray
}

# Persist for future shells, and set it here so the check below can run now.
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $key, "User")
$env:GEMINI_API_KEY = $key

Write-Host ""
Write-Host " stored : GEMINI_API_KEY (User scope, persists across reboots)" -ForegroundColor Green
Write-Host " length : $($key.Length) chars, ends ...$($key.Substring($key.Length - 4))" -ForegroundColor Green
Write-Host ""
Write-Host " Running the voice check..." -ForegroundColor Cyan
Write-Host ""

& "$root\run.ps1" scripts\verify_voice.py --play
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host " Key works. Restart the API server to pick it up:" -ForegroundColor Green
    Write-Host "   .\run.ps1 -u -m uvicorn app:app --host 127.0.0.1 --port 8000" -ForegroundColor DarkGray
} else {
    Write-Host " The check failed - see the reason above." -ForegroundColor Yellow
    Write-Host " The key is still stored; fix the cause and re-run:" -ForegroundColor DarkGray
    Write-Host "   .\run.ps1 scripts\verify_voice.py" -ForegroundColor DarkGray
}

Read-Host "`n Press Enter to close"
