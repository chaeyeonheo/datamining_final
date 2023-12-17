# datamining_final

실행 환경

python=3.9

GPU RAM 11GB  이상의 단일 GPU로 실행하였습니다.

윈도우에서 실행이 안됩니다 !! (GroundingDINO 설치가 불가능 !)



## 설치 ##

### Grounded-SAM 설치

https://github.com/IDEA-Research/Grounded-Segment-Anything

위 링크에 들어가셔서 도커를 사용하지 않고 설치를 진행하시면 됩니다. (많이 까다롭습니다.)



### Streamlit 설치(FrontEnd) ###

```bash
pip install streamlit
pip install streamlit-cropper
```





## 가중치 다운로드 ##

두 가중치를 모두 Grounded-Segment-Anything 폴더 안에 다운로드하시면 됩니다 !!

https://drive.google.com/file/d/1ns64Cz_B0WBpj0B1vI6gRD0QNJllD97B/view?usp=drive_link

https://drive.google.com/file/d/1vfx4ES1dYK19UL9WZ1Qv00muD8amtnAk/view?usp=sharing



## 실행 ##

```bash
streamlit run ./frontend/app.py
```

웹페이지 들어가셔서 이미지 업로드하신 후 기다리시면 처리된 이미지, 마스크 이미지, label(coco form)을 모두 다운로드하실 수 있습니다.
