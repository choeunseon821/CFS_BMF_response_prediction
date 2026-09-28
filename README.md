# CFS-BMF Design & Response Prediction Platform

## 실행
기존 Python 가상환경을 활성화한 뒤 프로젝트 폴더에서 한 줄만 실행합니다.

```powershell
python run_platform.py
```

`run_platform.py`는 필요한 Python 패키지를 확인하고 Streamlit 앱을 실행합니다.

## 데이터베이스
대용량 `.mat` DB는 `data/` 폴더에 둡니다. `database.mat` 또는 다른 `.mat` 파일을 자동으로 인식합니다.

## DL model
`BCRP_dataset_X.pkl`, `BCRP_dataset_Y.pkl`은 패키지에 포함되어 있습니다.
`BCRP_script.pt`는 다음 중 하나로 한 번만 준비하면 됩니다.

1. `models/BCRP_script.pt`로 직접 넣기
2. 프로젝트 상위 연구 폴더에 기존 파일이 있으면 `run_platform.py`가 자동 검색/복사
3. 홈페이지의 **First-time DL setup**에서 파일을 한 번 선택하여 저장

저장 후 다음 실행부터 다시 업로드할 필요가 없습니다.

## 사용 흐름
왼쪽 Workflow 메뉴가 없습니다. 한 페이지에서 순서대로 진행합니다.

1. Design Screening
2. Candidate Selection & Editing
3. ML Critical Response Point Prediction
4. DL Full Response Prediction
5. Design Report

처음 접속하면 AXIS 로고가 표시되는 랜딩 페이지가 먼저 열립니다.
