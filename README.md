
# Vision AI 기반 교차로 충돌 방지 시스템 (VITAS)

이 프로젝트는 **보행자 및 차량이 혼재된 교차로에서의 충돌 사고를 방지**하기 위해 Vision AI 기술을 활용하여 개발된 실시간 모니터링 및 예측 시스템입니다.  
본 시스템은 **2024 ICT 한이음 공모전 입선작** 으로서, 영상 기반 객체 탐지 및 속도 추정을 바탕으로 교차로 내 충돌 가능성을 예측하고 차량을 제어해 교통사고 방지를 할 수 있습니다.  
https://www.youtube.com/watch?v=sPzyF6GiYPM

---

## 🔍 프로젝트 개요

> **제목:** 보행자 차량 혼용 도로에서의 사고 위험 감소를 위한 V2I 기반 스마트 교차로 시스템  
> **형태:** PyQt 기반 GUI + YOLOv8 객체 탐지 + 속도 추정 + 충돌 예측  
> **모델:** YOLOv8 (Ultralytics) + 커스텀 학습 weight (`epoch52.pt`)  
> **영역:** A, B 교차 지점 구역을 통한 위치 기반 추론  
> **용도:** 실제 배포가 아닌 테스트 환경용 실험 코드

---

## 🧠 주요 기능

- ✅ YOLOv8 기반 객체 탐지 및 ID 트래킹  
- ✅ 객체 중심 좌표를 변환하여 교차로 영역(A/B) 매핑  
- ✅ 속도 계산 (픽셀 기반 추정 + 시간 보정)  
- ✅ 충돌 위험 예측 및 시각적 경고  
- ✅ PyQt 기반 실시간 GUI (표, 경고문구, bounding box 등)

---

## 📁 디렉토리 구조

```
VITAS/
├── data/                 # 테스트 영상 폴더
├── main.py               # 전체 GUI 및 로직 구현 스크립트
├── epoch52.pt            # 학습된 YOLOv8 모델
├── requirements.txt      # 의존성 리스트
└── README.md             # 프로젝트 문서
```

---

## 🚀 실행 방법

0. Clone the repository:

```bash
git clone https://github.com/ksh0330/2024_Hanium_ICT_Contest.git
cd 2024_Hanium_ICT_Contest
```

1. 필수 라이브러리 설치:

```bash
pip install -r requirements.txt
```

2. 테스트 영상(`.mp4`)을 `data/` 폴더에 넣고, `main.py` 내 경로 수정:

```python
video_source = "data/your_video.mp4"
```

3. 실행:

```bash
python main.py
```

> 웹캠을 사용할 경우, `video_source = 0` 으로 설정하세요.

---

## 📄 라이선스

본 프로젝트는 MIT 라이선스로 배포됩니다.  

---

## 🙋‍♀️ 개발 및 수상 이력

- **수상:** 2024 한이음 ICT 공모전 입선  
- **개발:** VITAS 팀  
- **비고:** 본 코드는 실제 시스템에 바로 적용하기보다는 시뮬레이션 기반 테스트용으로 구성됨
