import sys
import os
import subprocess
import site

def find_feast_executable():
    """
    Feast 실행 파일의 위치를 찾습니다.
    """
    possible_paths = []
    
    # 1. 현재 Python 인터프리터의 Scripts 폴더
    base_path = os.path.dirname(sys.executable)
    possible_paths.append(os.path.join(base_path, "Scripts", "feast.exe"))
    possible_paths.append(os.path.join(base_path, "Scripts", "feast"))
    
    # 2. 사용자 사이트 패키지의 Scripts 폴더 (Windows Store Python 등)
    user_base = site.getuserbase()
    if user_base:
        possible_paths.append(os.path.join(user_base, "Scripts", "feast.exe"))
        possible_paths.append(os.path.join(user_base, "Scripts", "feast"))
        # Python 버전별 경로 (예: Python311/Scripts)
        possible_paths.append(os.path.join(user_base, "Python311", "Scripts", "feast.exe"))

    print(">>> Searching for feast executable in:")
    for p in possible_paths:
        print(f"  - {p}")
        if os.path.exists(p):
            return p
            
    return None

def run_feast():
    feast_exe = find_feast_executable()
    
    if not feast_exe:
        print("❌ Could not find 'feast.exe'. Please ensure feast is installed and in your PATH.")
        # 최후의 수단: 모듈 실행 시도
        print(">>> Trying 'python -m feast' as fallback...")
        subprocess.run([sys.executable, "-m", "feast", "apply"], check=False)
        return

    print(f"\n✅ Found Feast: {feast_exe}")
    
    print("\n>>> Running 'feast apply'...")
    subprocess.run([feast_exe, "apply"], check=True)
    
    print("\n>>> Running 'feast materialize'...")
    from datetime import datetime
    end_date = datetime.now().isoformat()
    subprocess.run([feast_exe, "materialize-incremental", end_date], check=True)
    
    print("\n🎉 Feast setup completed successfully!")

if __name__ == "__main__":
    run_feast()
