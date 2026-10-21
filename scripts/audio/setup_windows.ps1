<#>
.SYNOPSIS
    Windows audio routing setup for BP — VB-Cable virtual mic + loopback.

.DESCRIPTION
    Idempotent: safe to run multiple times. Verifies and reports the selected devices.
    Requires: Administrator (for VB-Cable install), Windows 10/11.

.USAGE
    powershell -ExecutionPolicy Bypass -File scripts/audio/setup_windows.ps1           # install + verify
    powershell -ExecutionPolicy Bypass -File scripts/audio/setup_windows.ps1 -Check    # only verify
    powershell -ExecutionPolicy Bypass -File scripts/audio/setup_windows.ps1 -Json     # machine-readable output

.NOTES
    VB-Cable from https://vb-audio.com/Cable/
    After install, reboot is required for driver to load.
#>

[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Json,
    [switch]$ForceInstall
)

function Write-Result {
    param($Result)
    if ($Json) {
        $Result | ConvertTo-Json -Depth 5
    } else {
        Write-Host "VB-Cable installed: $($Result.VBCableInstalled)"
        Write-Host "Loopback available: $($Result.LoopbackAvailable)"
        Write-Host "Virtual mic: $($Result.VirtualMicName ?? 'NOT FOUND')"
        Write-Host "Loopback device: $($Result.LoopbackDevice ?? 'NOT FOUND')"
        Write-Host "Input device (mic): $($Result.InputDevice ?? 'NOT FOUND')"
        Write-Host "Output device: $($Result.OutputDevice ?? 'NOT FOUND')"
        if ($Result.Error) { Write-Host "ERROR: $($Result.Error)" -ForegroundColor Red }
        if ($Result.CommandsRun) {
            Write-Host "`nCommands run:"
            $Result.CommandsRun | ForEach-Object { Write-Host "  $_" }
        }
    }
    exit (if ($Result.Ok) { 0 } else { 1 })
}

function IsAdmin {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-AudioDevices {
    # Get audio devices via Get-PnpDevice and Get-WmiObject
    $devices = @()
    
    # Input devices (microphones)
    $inputs = Get-PnpDevice -Class AudioEndpoint -PresentOnly | Where-Object { $_.FriendlyName -match "Microphone|Input|VB-Cable" }
    foreach ($d in $inputs) {
        $devices += [pscustomobject]@{
            Name = $d.FriendlyName
            InstanceId = $d.InstanceId
            IsInput = $true
            IsOutput = $false
            IsVirtual = $d.FriendlyName -match "VB-Cable"
        }
    }
    
    # Output devices (speakers)
    $outputs = Get-PnpDevice -Class AudioEndpoint -PresentOnly | Where-Object { $_.FriendlyName -match "Speaker|Output|VB-Cable|Headphone" }
    foreach ($d in $outputs) {
        $devices += [pscustomobject]@{
            Name = $d.FriendlyName
            InstanceId = $d.InstanceId
            IsInput = $false
            IsOutput = $true
            IsVirtual = $d.FriendlyName -match "VB-Cable"
        }
    }
    
    return $devices
}

function Test-VBCableInstalled {
    $devices = Get-AudioDevices
    return ($devices | Where-Object { $_.Name -match "VB-Cable" }).Count -gt 0
}

function Select-Devices {
    param($Devices)
    
    $virtualMic = $null
    $loopback = $null
    $inputDev = $null
    $outputDev = $null
    
    # Virtual mic = VB-Cable (output device that apps use as mic input)
    $vbOut = $Devices | Where-Object { $_.IsOutput -and $_.IsVirtual } | Select-Object -First 1
    if ($vbOut) { $virtualMic = $vbOut.Name; $outputDev = $vbOut.Name }
    
    # Loopback = VB-Cable input (monitor of output)
    $vbIn = $Devices | Where-Object { $_.IsInput -and $_.IsVirtual } | Select-Object -First 1
    if ($vbIn) { $loopback = $vbIn.Name }
    
    # Input = real microphone (not virtual)
    $realMic = $Devices | Where-Object { $_.IsInput -and -not $_.IsVirtual -and ($_.Name -match "Microphone|Mic|Headset") } | Select-Object -First 1
    if (-not $realMic) { $realMic = $Devices | Where-Object { $_.IsInput -and -not $_.IsVirtual } | Select-Object -First 1 }
    if ($realMic) { $inputDev = $realMic.Name }
    
    return @{ VirtualMic = $virtualMic; Loopback = $loopback; Input = $inputDev; Output = $outputDev }
}

function Install-VBCable {
    if (-not (IsAdmin)) {
        return @{ Ok = $false; Msg = "Run as Administrator to install VB-Cable" }
    }
    
    $url = "https://vb-audio.com/Cable/VB-CABLE_Driver_Pack.zip"
    $tempDir = Join-Path $env:TEMP "VBCable_$(Get-Random)"
    $zipPath = Join-Path $tempDir "VBCable.zip"
    
    try {
        Write-Host "Downloading VB-Cable driver pack..."
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
        
        Write-Host "Extracting..."
        Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force
        
        # Find the installer (usually in VB-CABLE_Driver_Pack43 or similar)
        $installer = Get-ChildItem -Path $tempDir -Recurse -Filter "*.exe" | Where-Object { $_.Name -match "VBCable|setup" } | Select-Object -First 1
        if (-not $installer) {
            return @{ Ok = $false; Msg = "VB-Cable installer not found in extracted archive" }
        }
        
        Write-Host "Running installer (silent)..."
        $proc = Start-Process -FilePath $installer.FullName -ArgumentList "/S" -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            return @{ Ok = $false; Msg = "VB-Cable installer exited with code $($proc.ExitCode)" }
        }
        
        # Cleanup
        Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
        
        return @{ Ok = $true; Msg = "VB-Cable installed. REBOOT REQUIRED for driver to load." }
    }
    catch {
        return @{ Ok = $false; Msg = "Install failed: $_" }
    }
}

function Setup-Windows {
    param($Install)
    
    $result = [pscustomobject]@{
        Ok = $false
        VBCableInstalled = $false
        LoopbackAvailable = $false
        VirtualMicName = $null
        LoopbackDevice = $null
        InputDevice = $null
        OutputDevice = $null
        Error = $null
        CommandsRun = @()
    }
    
    # Check VB-Cable
    $vbInstalled = Test-VBCableInstalled
    $result.VBCableInstalled = $vbInstalled
    $result.CommandsRun += "Checked VB-Cable installation: $vbInstalled"
    
    if (-not $vbInstalled) {
        if ($Install -or $ForceInstall) {
            if (-not (IsAdmin)) {
                $result.Error = "VB-Cable not installed. Re-run as Administrator with -Install or -ForceInstall"
                return $result
            }
            $installResult = Install-VBCable
            $result.CommandsRun += "Installed VB-Cable: $($installResult.Msg)"
            if (-not $installResult.Ok) {
                $result.Error = $installResult.Msg
                return $result
            }
            $result.VBCableInstalled = $true
            $result.Error = "VB-Cable installed. REBOOT REQUIRED. Run again after reboot."
            return $result
        } else {
            $result.Error = "VB-Cable not installed. Install from https://vb-audio.com/Cable/ or run as Admin with -Install"
            return $result
        }
    }
    
    # Select devices
    $devices = Get-AudioDevices
    $selected = Select-Devices $devices
    $result.VirtualMicName = $selected.VirtualMic
    $result.LoopbackDevice = $selected.Loopback
    $result.InputDevice = $selected.Input
    $result.OutputDevice = $selected.Output
    $result.LoopbackAvailable = [bool]$selected.Loopback
    $result.Ok = [bool]($selected.VirtualMic -and $selected.Loopback)
    
    if (-not $result.Ok) {
        $result.Error = "VB-Cable installed but virtual mic/loopback not detected. Try reboot."
    }
    
    return $result
}

# Main
try {
    $result = Setup-Windows -Install (-not $Check)
    Write-Result $result
}
catch {
    $err = [pscustomobject]@{
        Ok = $false
        Error = "Script error: $_"
    }
    Write-Result $err
}