Write-Host 'Creating Python virtual environment "scripts/.venv"'
python -m venv .\scripts\.venv

Write-Host 'Installing dependencies from "requirements.txt" into virtual environment'
.\scripts\.venv\Scripts\python -m pip install -r .\requirements.txt

Write-Host 'Installing development dependencies from "requirements-dev.txt" into virtual environment'
if (Test-Path requirements-dev.txt) {
    .\scripts\.venv\Scripts\python -m pip install -r .\requirements-dev.txt
}

Write-Host 'Function app dependencies installed successfully'
