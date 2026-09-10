# Sets the env vars this project needs, then hands you an activated shell.
#   .\run.ps1                    -> activate only
#   .\run.ps1 scripts\check_env.py -> activate and run a script

# C: is nearly full, so every cache lives on D:, consolidated under the project
# so one folder holds the lot instead of four scattered at the drive root.
#
# The old top-level paths (D:\torch-hub, D:\hf-cache, D:\pip-cache, D:\pip-tmp)
# are now directory junctions pointing here, so a process started before the
# move -- or anything with a hardcoded old path -- still resolves. Delete the
# junctions only once nothing references them.
$CACHE = "$PSScriptRoot\.cache"
$env:HF_HOME = "$CACHE\huggingface"
$env:TORCH_HOME = "$CACHE\torch"
$env:TMP = "$CACHE\tmp"
$env:TEMP = "$CACHE\tmp"
$env:PIP_CACHE_DIR = "$CACHE\pip"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
# Keeps __pycache__ out of the source tree and off C:.
$env:PYTHONPYCACHEPREFIX = "$CACHE\pycache"

# winget put ffmpeg on the User PATH, but shells spawned from an editor that
# started before the install inherit a stale environment and cannot see it.
# Re-read PATH from the registry so ffmpeg works without restarting anything.
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:PATH = "$machinePath;$userPath"

# Same staleness trap, and it bit exactly once: the API key was set with
# SetEnvironmentVariable(..., "User") while this shell's parent was already
# running, so the server started with no key and reported gemini=false despite
# the key being correctly stored. Re-read it from the registry unless the
# session already carries one, so a freshly-set key works without logging out.
foreach ($name in "GEMINI_API_KEY", "GOOGLE_API_KEY") {
    if (-not [System.Environment]::GetEnvironmentVariable($name, "Process")) {
        $stored = [System.Environment]::GetEnvironmentVariable($name, "User")
        if ($stored) { Set-Item -Path "env:$name" -Value $stored }
    }
}

Set-Location $PSScriptRoot

if ($args.Count -gt 0) {
    & "$PSScriptRoot\venv\Scripts\python.exe" @args
} else {
    & "$PSScriptRoot\venv\Scripts\Activate.ps1"
    Write-Host "have-hit env ready. HF_HOME=$env:HF_HOME" -ForegroundColor Green
}
