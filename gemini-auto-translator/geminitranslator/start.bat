@echo off
REM 设置 Python 路径
set PYTHON_PATH=C:/Users/gycchris/AppData/Local/Microsoft/WindowsApps/python3.11.exe

REM 设置 Python 脚本路径
set SCRIPT_PATH=d:/gemini/geminitranslator/main.py

REM 启动 Python 脚本
"%PYTHON_PATH%" "%SCRIPT_PATH%"

REM 打开默认浏览器
start http://127.0.0.1:8001
