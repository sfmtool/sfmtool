param(
    [Parameter(Mandatory = $true)]
    [string]$Phase
)

$ErrorActionPreference = "Stop"

function Write-SnapshotValue {
    param(
        [string]$Name,
        [scriptblock]$Query
    )

    try {
        $value = & $Query
        if ($null -eq $value) {
            $value = "unavailable"
        }
        elseif ($value -isnot [string]) {
            $value = $value | ConvertTo-Json -Compress -Depth 4
        }
        Write-Host "UIENV $Name=$value"
    }
    catch {
        Write-Host "UIENV $Name=unavailable error=$($_.Exception.Message)"
    }
}

Write-Host "UIENV phase=$Phase timestamp=$([DateTimeOffset]::UtcNow.ToString('o'))"
Write-Host "UIENV runner_os=$env:RUNNER_OS runner_arch=$env:RUNNER_ARCH image_os=$env:ImageOS image_version=$env:ImageVersion number_of_processors=$env:NUMBER_OF_PROCESSORS"

Write-SnapshotValue "cpu" {
    @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores,
        NumberOfLogicalProcessors, CurrentClockSpeed, MaxClockSpeed, LoadPercentage)
}
Write-SnapshotValue "computer" {
    Get-CimInstance Win32_ComputerSystem |
        Select-Object Manufacturer, Model, TotalPhysicalMemory
}
Write-SnapshotValue "os" {
    Get-CimInstance Win32_OperatingSystem |
        Select-Object Version, BuildNumber, LastBootUpTime, FreePhysicalMemory
}
Write-SnapshotValue "load" {
    Get-CimInstance Win32_PerfFormattedData_PerfOS_System |
        Select-Object ProcessorQueueLength, SystemUpTime
}
Write-SnapshotValue "memory" {
    Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory |
        Select-Object AvailableMBytes, PagesPersec
}
Write-SnapshotValue "disk" {
    @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType = 3" |
        Select-Object DeviceID, Size, FreeSpace)
}

Write-SnapshotValue "session" {
    $current = Get-Process -Id $PID
    [pscustomobject]@{
        Id          = $current.SessionId
        Name        = $env:SESSIONNAME
        Interactive = [Environment]::UserInteractive
        User        = "$env:USERDOMAIN\$env:USERNAME"
    }
}
Write-SnapshotValue "quser" {
    $output = (& quser 2>&1) -join " | "
    if ($LASTEXITCODE -ne 0) {
        throw "quser exited $LASTEXITCODE`: $output"
    }
    $output
}
Write-SnapshotValue "desktop_processes" {
    @(Get-Process -Name explorer, dwm -ErrorAction SilentlyContinue |
        Select-Object ProcessName, Id, SessionId, CPU, WorkingSet64)
}

Write-SnapshotValue "screen" {
    Add-Type -AssemblyName System.Windows.Forms
    $screen = [System.Windows.Forms.Screen]::PrimaryScreen
    [pscustomobject]@{
        Bounds      = "$($screen.Bounds.Width)x$($screen.Bounds.Height)"
        WorkingArea = "$($screen.WorkingArea.Width)x$($screen.WorkingArea.Height)"
        Primary     = $screen.Primary
    }
}
Write-SnapshotValue "video" {
    @(Get-CimInstance Win32_VideoController |
        Select-Object Name, CurrentHorizontalResolution,
            CurrentVerticalResolution, CurrentRefreshRate, DriverVersion)
}
Write-SnapshotValue "power" {
    $output = (& powercfg /getactivescheme 2>&1) -join " "
    if ($LASTEXITCODE -ne 0) {
        throw "powercfg exited $LASTEXITCODE`: $output"
    }
    $output.Trim()
}

Write-SnapshotValue "defender_status" {
    Get-MpComputerStatus |
        Select-Object AMServiceEnabled, AntivirusEnabled, BehaviorMonitorEnabled,
            IoavProtectionEnabled, NISEnabled, OnAccessProtectionEnabled,
            RealTimeProtectionEnabled, IsTamperProtected
}
Write-SnapshotValue "defender_preferences" {
    Get-MpPreference |
        Select-Object DisableRealtimeMonitoring, DisableBehaviorMonitoring,
            DisableIOAVProtection, ExclusionPath, ScanAvgCPULoadFactor
}
Write-SnapshotValue "services" {
    @(Get-Service -Name WinDefend, Themes, FontCache -ErrorAction SilentlyContinue |
        Select-Object Name, Status, StartType)
}
Write-SnapshotValue "uia" {
    (Get-Item "$env:WINDIR\System32\UIAutomationCore.dll").VersionInfo |
        Select-Object FileVersion, ProductVersion
}

# A missing native diagnostic such as `quser` may leave LASTEXITCODE non-zero
# even though Write-SnapshotValue caught and reported it. This is an
# observation-only script: unavailable probes must not suppress the UI tests.
exit 0
