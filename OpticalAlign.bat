
set PYTHON_CALL="../env_oa/Scripts/python.exe" 

@ECHO OFF 
:: This batch file to open the Optical Align Visualization  
TITLE Esper Optical Align V1.0

ECHO RUNNING SPOT TABLE VISUALIZATION
%PYTHON_CALL% "../OpticalAlign/src/svbeammeasure_v10/app.py"

ECHO Finished Optical Align V1.0 ... 

PAUSE