@echo off
REM Launch bass_analysis with reflect padding on both ends of the whole file
REM Usage:  launch_reflect.bat myfile.wav [extra args...]
python "%~dp0..\bass_analysis.py" %* --composite "r:0:L:r"
