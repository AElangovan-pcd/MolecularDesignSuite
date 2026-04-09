@echo off
cd /d C:\Users\easam\Documents\ClaudeProjects\RDKitProjects

set CONDA_ENV=C:\Users\easam\.conda\envs\moldesign
set PATH=%CONDA_ENV%;%CONDA_ENV%\Library\bin;%CONDA_ENV%\Library\mingw-w64\bin;%CONDA_ENV%\Scripts;%PATH%

python -m streamlit run app.py --server.port 8501 --server.headless true
pause
