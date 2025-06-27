@echo off
echo "Building executable for Windows..."
pyinstaller --icon="data/icon.ico" --add-data="data;data" --add-data="model;model" --add-data="C:\Users\andre\Documents\GitHub\ML-GUI\.venv\Lib\site-packages\rdkit\Data;rdkit/Data" --collect-submodules="sklearn" --collect-submodules="scipy" LIBtoolkit.py
echo "Build complete. Check the 'dist' folder."
pause