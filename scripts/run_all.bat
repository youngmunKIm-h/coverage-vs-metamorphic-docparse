@echo off
REM 전체 파이프라인 실행 (Anaconda Prompt에서, 저장소 최상위 폴더에서 실행)
REM 사전 조건: covexp / covtarget 환경 설치, data\pubtabnet_raw\pubtabnet.tar.gz 다운로드 완료

call conda activate covexp || exit /b 1
python fetch_pubtabnet.py data\pubtabnet_raw\pubtabnet.tar.gz || exit /b 1
python run.py build || exit /b 1
python run.py surrogate || exit /b 1

call conda activate covtarget || exit /b 1
python run_target.py docling || exit /b 1

call conda activate covexp || exit /b 1
python run.py judge --tgt docling || exit /b 1
python run.py rq1 --tgt docling || exit /b 1
python run.py select --tgt docling || exit /b 1
python run.py analyze --tgt docling || exit /b 1
python run.py analyze --tgt docling --metric tg_unique || exit /b 1
echo 완료. 결과는 work\ 폴더에 있습니다.
