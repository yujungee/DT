# Multi-Task RL for `gym` & `metaworld`

이 레포는 `gym` 환경과 `metaworld` 환경에서 Multi-Task Reinforcement Learning을 수행하기 위한 예시 코드입니다. 핵심적으로 `gym/Decision_transformer/method`에 Adapter 기반의 PEFT(PErsonalization Finetuning) 모듈을 추가하여, 보다 효율적으로 모델을 튜닝할 수 있도록 구성했습니다.

---

## 구조

- **gym/Decision_transformer/method/**  
  - Adapter 기반의 PEFT 모듈이 포함되어 있습니다.
  - LoRA(Low-Rank Adaptation)와 같은 모듈에서 `rank`는 수동으로 지정해주어야 합니다.  
    > **주의**: 현재 기본값은 `1`입니다. 사용 시 원하는 값을 `4 ~ 32`로 수정해 주세요.

- **gym/models/{method}_dt.py**  
  - Adapter가 부착된 Transformer 파일이 포함되어 있습니다.

- **metaworld/scripts/**  
  - Shell Script 파일들이 존재하며, 이를 통해 각종 실험 환경을 실행할 수 있습니다.

---

## 실행 방법

1. **환경 설정**
   - Python 버전 및 라이브러리 의존성을 맞춰주세요.
   - `gym`, `metaworld`, `torch`, 기타 필요한 라이브러리를 설치하세요.

2. **PEFT 모듈 설정**
   - `gym/Decision_transformer/method` 내부에서 LoRA 등 다양한 어댑터 기반 기법을 사용해볼 수 있습니다.
   - LoRA 등을 사용할 경우, `rank`를 원하는 값(예: 4, 8, 16, 32)으로 수정해 주십시오.

3. **Transformer 파일 확인**
   - `gym/models/{method}_dt.py`에서 Adapter가 적용된 Transformer 코드가 있는지 확인하고, 필요한 부분을 수정 또는 설정합니다.

4. **학습 스크립트 실행**
   - `metaworld/scripts` 디렉터리 안의 Shell Script를 통해 각종 학습 및 실험 환경을 실행할 수 있습니다.
   - 예시:
     ```bash
     cd metaworld/scripts
     bash run_training.sh
     ```
   - 위 스크립트는 메타월드 환경에서 Multi-Task RL을 수행하는 예시로, 적절한 하이퍼파라미터를 설정해주면 됩니다.

---

## 참고 사항

- Adapter 기반의 PEFT(LoRA, Prompt Tuning 등)를 적절히 활용하면, 기존 Transformer 모델을 작은 파라미터 변경만으로 효과적으로 튜닝할 수 있습니다.
- PEFT 모듈을 사용할 때는 GPU 메모리 사용량, 학습 속도를 고려하여 `rank` 값을 조정하세요.
- 환경별 세부 설정(예: 과제 수, 보상 구조, 에피소드 길이 등)은 `run_training.sh` 등의 스크립트 혹은 Python 코드 내부에 정의되어 있습니다.

---

## 라이선스

- 본 프로젝트는 오픈 소스 라이선스를 따르며, 상세 정보는 `LICENSE` 파일을 참고하시기 바랍니다.

---
