# Python Virtual Environment — Windows

## 1. Create the environment
```cmd
python -m venv venv
```

## 2. Activate the environment
```cmd
venv\Scripts\activate
```

> **Using PowerShell and getting an error?** Run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then activate the environment again.

## 3. Install dependencies
```cmd
pip install -r requirements.txt
```

## 4. Deactivate the environment
```cmd
deactivate
```

# ALL TOGETHER
```cmd
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser; python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
```