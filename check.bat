@echo off
if exist "check_env.log" del check_env.log
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cargo check > check_env.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Cargo Check Succeeded
    exit /b 0
) else (
    echo Cargo Check Failed
    exit /b 1
)
