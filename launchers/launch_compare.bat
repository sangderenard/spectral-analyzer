@echo off
REM Launch bass_analysis with default composite: 15-25s vs last 25-15s
REM Usage:  launch_compare.bat myfile.wav [extra args...]
python "%~dp0..\bass_analysis.py" %* --composite default
