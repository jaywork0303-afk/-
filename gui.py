"""
실시간 사물 인식 사운드 플레이어 (GUI 버전)
- tkinter 기반 GUI
- 다중 사물 동시 재생 모드 지원
- 실시간 감지 결과 표시 / 임계값 조정 / 시작·정지 컨트롤
사용법:
    python gui.py
"""
import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import pygame
from PIL import Image, ImageTk
from ultralytics import YOLO

# ---- 설정 ----
MODEL_NAME = "yolov8n.pt"
SOUND_MAP_FILE = "sound_map.json"
CAMERA_INDEX = 0
MAX_CHANNELS = 16   # pygame 동시 재생 채널 수 (다중 모드용)
PREVIEW_MAX_W = 640
PREVIEW_MAX_H = 480


class ObjectSoundApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Object Detection Sound Player")
        self.root.geometry("1000x600")

        # 런타임 상태
        self.running = False
        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.last_played: dict[str, float] = {}
        self.sounds: dict[str, pygame.mixer.Sound] = {}

        # tk 변수 (GUI 바인딩)
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.cooldown_sec = tk.DoubleVar(value=2.0)
        self.multi_mode = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="초기화 중...")

        # pygame mixer (다중 채널 확보)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(MAX_CHANNELS)

        # UI 먼저 만들고 모델 로드 (사용자에게 진행 상황 표시)
        self._build_ui()
        self.root.update()

        self.status_text.set("YOLO 모델 로딩 중...")
        self.root.update()
        self.model = YOLO(MODEL_NAME)

        self._load_sounds()
        self._refresh_sound_panel()
        self.status_text.set(f"준비 완료 — 사운드 {len(self.sounds)}개 로드됨")

    # ---------- 사운드 ----------
    def _load_sounds(self) -> None:
        if not Path(SOUND_MAP_FILE).exists():
            print(f"[경고] {SOUND_MAP_FILE} 파일이 없습니다.")
            return

        with open(SOUND_MAP_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for label, sound_path in raw.items():
            if not Path(sound_path).exists():
                print(f"[경고] '{label}' 사운드 파일 없음: {sound_path}")
                continue
            try:
                self.sounds[label] = pygame.mixer.Sound(sound_path)
                print(f"[로드] {label} -> {sound_path}")
            except pygame.error as e:
                print(f"[오류] '{label}' 로드 실패: {e}")

    def _play_sounds(self, labels: list[str]) -> None:
        """감지된 라벨에 대해 모드에 따라 사운드 재생."""
        now = time.time()
        cd = self.cooldown_sec.get()
        unique = list(dict.fromkeys(labels))  # 순서 유지 dedup

        if self.multi_mode.get():
            # 다중 동시 재생: 감지된 모든 사물의 사운드를 병렬로 재생
            for label in unique:
                if label not in self.sounds:
                    continue
                if now - self.last_played.get(label, 0.0) < cd:
                    continue
                self.sounds[label].play()
                self.last_played[label] = now
        else:
            # 단일 재생: 어떤 사운드라도 재생 중이면 대기, 아니면 첫 번째 후보 1개만 재생
            if pygame.mixer.get_busy():
                return
            for label in unique:
                if label not in self.sounds:
                    continue
                if now - self.last_played.get(label, 0.0) < cd:
                    continue
                self.sounds[label].play()
                self.last_played[label] = now
                break

    # ---------- UI ----------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ----- 좌측: 영상 -----
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(left, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ----- 우측: 컨트롤 -----
        right = ttk.Frame(main, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self.start_btn = ttk.Button(right, text="▶ 시작", command=self.toggle)
        self.start_btn.pack(fill=tk.X, pady=(0, 10))

        # 재생 모드
        mode_frame = ttk.LabelFrame(right, text="재생 모드", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            mode_frame,
            text="다중 사물 동시 재생",
            variable=self.multi_mode,
        ).pack(anchor=tk.W)
        ttk.Label(
            mode_frame,
            text="(끄면 한 번에 한 사운드만 재생)",
            foreground="#666",
            font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W)

        # 임계값/쿨다운
        param_frame = ttk.LabelFrame(right, text="설정", padding=8)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(param_frame, text="신뢰도 임계값").pack(anchor=tk.W)
        ttk.Scale(
            param_frame, from_=0.1, to=0.95,
            variable=self.conf_threshold, orient=tk.HORIZONTAL,
        ).pack(fill=tk.X)
        self.conf_label = ttk.Label(param_frame, text="0.50")
        self.conf_label.pack(anchor=tk.W)
        self.conf_threshold.trace_add(
            "write",
            lambda *_: self.conf_label.config(text=f"{self.conf_threshold.get():.2f}"),
        )

        ttk.Label(param_frame, text="쿨다운 (초)").pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(
            param_frame, from_=0.0, to=5.0,
            variable=self.cooldown_sec, orient=tk.HORIZONTAL,
        ).pack(fill=tk.X)
        self.cd_label = ttk.Label(param_frame, text="2.0")
        self.cd_label.pack(anchor=tk.W)
        self.cooldown_sec.trace_add(
            "write",
            lambda *_: self.cd_label.config(text=f"{self.cooldown_sec.get():.1f}"),
        )

        # 감지 목록
        det_frame = ttk.LabelFrame(right, text="감지 중인 사물", padding=8)
        det_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.det_listbox = tk.Listbox(det_frame, height=6)
        self.det_listbox.pack(fill=tk.BOTH, expand=True)

        # 로드된 사운드
        self.snd_frame = ttk.LabelFrame(right, text="로드된 사운드", padding=8)
        self.snd_frame.pack(fill=tk.X)

        # 상태 바
        status_bar = ttk.Label(
            self.root, textvariable=self.status_text,
            relief=tk.SUNKEN, anchor=tk.W, padding=4,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _refresh_sound_panel(self) -> None:
        for widget in self.snd_frame.winfo_children():
            widget.destroy()
        self.snd_frame.config(text=f"로드된 사운드 ({len(self.sounds)})")
        if not self.sounds:
            ttk.Label(
                self.snd_frame, text="(없음)",
                foreground="#888",
            ).pack(anchor=tk.W)
            return
        for label in self.sounds:
            ttk.Label(self.snd_frame, text=f"♪ {label}").pack(anchor=tk.W)

    # ---------- 실행 제어 ----------
    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.status_text.set(f"카메라를 열 수 없습니다 (index={CAMERA_INDEX})")
            self.cap = None
            return
        self.running = True
        self.start_btn.config(text="■ 정지")
        self.status_text.set("실행 중")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        pygame.mixer.stop()
        self.start_btn.config(text="▶ 시작")
        self.status_text.set("정지됨")

    def _loop(self) -> None:
        """워커 스레드: 카메라 캡처 + YOLO 추론 + 사운드 재생."""
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                break

            results = self.model(
                frame,
                conf=self.conf_threshold.get(),
                verbose=False,
            )
            result = results[0]

            detected: list[tuple[str, float]] = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                detected.append((label, conf))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 사운드가 매핑된 사물은 초록, 아닌 사물은 노랑
                color = (0, 255, 0) if label in self.sounds else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            self._play_sounds([l for l, _ in detected])

            # GUI 업데이트는 메인 스레드에서
            self.root.after(0, self._update_video, frame, detected)

    def _update_video(self, frame, detected: list[tuple[str, float]]) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo  # GC 방지 참조 유지

        self.det_listbox.delete(0, tk.END)
        for label, conf in detected:
            mark = "♪" if label in self.sounds else " "
            self.det_listbox.insert(tk.END, f"{mark} {label}  ({conf:.2f})")

    def on_close(self) -> None:
        self.stop()
        pygame.mixer.quit()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = ObjectSoundApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
