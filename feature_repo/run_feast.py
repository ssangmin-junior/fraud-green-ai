from feast import FeatureStore
from datetime import datetime

def apply_and_materialize():
    repo_path = "." # feature_repo 폴더에서 실행한다고 가정
    fs = FeatureStore(repo_path=repo_path)

    print(">>> Applying Feature Store configuration...")
    fs.apply([
        # features.py에서 정의한 객체들을 가져와야 함
        # 하지만 apply()는 보통 CLI에서 파일 스캔을 수행하므로,
        # SDK에서는 apply() 메서드가 조금 다르게 동작합니다.
        # 여기서는 CLI를 subprocess로 호출하는 게 낫습니다.
    ])
    # ... SDK로 apply 하려면 객체를 다 import 해야 해서 복잡합니다.

if __name__ == "__main__":
    import subprocess
    import sys
    
    # feast 모듈이 설치된 위치를 찾아서 cli 모듈 실행
    # python -m feast.cli apply
    
    print(">>> Running 'feast apply'...")
    subprocess.run([sys.executable, "-m", "feast.cli", "apply"], check=True)
    
    print("\n>>> Running 'feast materialize'...")
    from datetime import datetime
    end_date = datetime.now().isoformat()
    subprocess.run([sys.executable, "-m", "feast.cli", "materialize-incremental", end_date], check=True)
    
    print("\n✅ Feast setup completed successfully!")
