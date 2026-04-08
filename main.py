"""
실시간 사물 인식 사운드 플레이어
- 노트북 카메라로 사물을 감지(YOLOv8)
- 감지된 사물별로 지정된 사운드를 스피커로 재생
사용법:
    python main.py
종료: q 키
"""
import json
import os
import time
from pathlib import Path

import cv2
import pygame
from ultralytics import YOLO

# ---- 설정 ----
MODEL_NAME = "yolov8n.pt"          # 가장 가벼운 YOLOv8 모델 (자동 다운로드)
SOUND_MAP_FILE = "sound_map.json"  # 클래스명 → 사운드 파일 매핑
CONF_THRESHOLD = 0.5               # 감지 신뢰도 임계값
COOLDOWN_SEC = 2.0                 # 같은 사물 연속 재생 방지 (초)
CAMERA_INDEX = 0                   # 기본 노트북 카메라


def load_sound_map(path: str) -> dict:
    """클래스명 → pygame Sound 객체 매핑을 로드한다."""
    if not Path(path).exists():
        print(f"[경고] {path} 파일이 없습니다. 빈 매핑으로 시작합니다.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sounds = {}
    for label, sound_path in raw.items():
        if not Path(sound_path).exists():
            print(f"[경고] '{label}' 사운드 파일이 없습니다: {sound_path}")
            continue
        try:
            sounds[label] = pygame.mixer.Sound(sound_path)
            print(f"[로드] {label} -> {sound_path}")
        except pygame.error as e:
            print(f"[오류] '{label}' 사운드 로드 실패: {e}")
    return sounds


def main() -> None:
    # pygame mixer 초기화 (사운드 재생용)
    pygame.mixer.init()

    sounds = load_sound_map(SOUND_MAP_FILE)
    if not sounds:
        print("[안내] 재생 가능한 사운드가 없습니다. sounds/ 폴더에 .wav 파일을 추가하세요.")

    # YOLO 모델 로드 (최초 실행 시 자동 다운로드)
    print(f"[로드] YOLO 모델: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # 카메라 열기
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={CAMERA_INDEX}).")

    print("[시작] 'q' 키를 누르면 종료됩니다.")
    last_played = {}  # label -> 마지막 재생 시각

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[경고] 프레임을 읽지 못했습니다.")
                break

            # YOLO 추론 (스트리밍 모드 X, 단일 프레임)
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            result = results[0]

            detected_labels = set()
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                detected_labels.add(label)

                # 박스 그리기
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # 사운드 재생 (쿨다운 적용)
            now = time.time()
            for label in detected_labels:
                if label not in sounds:
                    continue
                if now - last_played.get(label, 0.0) < COOLDOWN_SEC:
                    continue
                sounds[label].play()
                last_played[label] = now
                print(f"[재생] {label}")

            cv2.imshow("Object Detection Sound Player (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("[종료]")


if __name__ == "__main__":
    main()
