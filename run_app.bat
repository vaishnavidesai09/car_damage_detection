@echo off
cd /d E:\car_damage_detection\files

:: Set environment variable to fix torch.classes error
set STREAMLIT_WATCH_USE_POLLING=true

streamlit run app.py --server.runOnSave false
pause
