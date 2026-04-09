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
MASCOT_FILE = "mascot.png"

# ---- 테마: 모노톤 크레파스 + 반사광(iridescent) 악센트 ----
# 어둡고 거친 종이 위에 크레파스로 칠한 듯한 고대비 + 홀로그래픽 포인트
THEME = {
    "bg":         "#14121A",   # 먹지 같은 암흑
    "panel":      "#1F1C28",   # 살짝 떠오르는 패널
    "panel_alt":  "#2A2634",
    "ink":        "#F5F1E6",   # 아이보리 크레파스 잉크
    "muted":      "#9690A8",
    "line":       "#3B3548",
    # 반사광 팔레트 (홀로그래픽 느낌)
    "iris_pink":  "#F5B6C9",
    "iris_lilac": "#B8A4E3",
    "iris_cyan":  "#7FD4E0",
    "iris_gold":  "#E8D98B",
    "iris_lime":  "#C6E89A",
    "accent":     "#B8A4E3",   # 기본 악센트 = 라일락
    "danger":     "#F5B6C9",
}
# 손글씨/크레파스 느낌 폰트 (Windows 기본 내장)
FONT_DISPLAY = ("Comic Sans MS", 14, "bold")
FONT_BODY    = ("Comic Sans MS", 10)
FONT_SMALL   = ("Comic Sans MS", 8)
FONT_MONO    = ("Consolas", 9)


class ObjectSoundApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("✦ Object Detection Sound Player ✦")
        self.root.geometry("1080x680")
        self.root.configure(bg=THEME["bg"])
        self._apply_theme()
        self._iris_cycle = [
            THEME["iris_lilac"], THEME["iris_cyan"],
            THEME["iris_pink"], THEME["iris_gold"], THEME["iris_lime"],
        ]
        self._iris_idx = 0

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

    # ---------- 테마 ----------
    def _apply_theme(self) -> None:
        """ttk 위젯에 모노톤 크레파스 + 반사광 테마 적용."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")  # clam 이 가장 커스터마이즈 잘 먹음
        except tk.TclError:
            pass

        T = THEME
        style.configure(".",
            background=T["bg"], foreground=T["ink"],
            fieldbackground=T["panel"], bordercolor=T["line"],
            lightcolor=T["line"], darkcolor=T["line"],
            font=FONT_BODY,
        )
        style.configure("TFrame", background=T["bg"])
        style.configure("Panel.TFrame", background=T["panel"])
        style.configure("TLabel", background=T["bg"], foreground=T["ink"], font=FONT_BODY)
        style.configure("Panel.TLabel", background=T["panel"], foreground=T["ink"])
        style.configure("Muted.TLabel", background=T["panel"], foreground=T["muted"], font=FONT_SMALL)
        style.configure("Title.TLabel",
            background=T["bg"], foreground=T["iris_lilac"], font=FONT_DISPLAY,
        )
        style.configure("Sub.TLabel",
            background=T["bg"], foreground=T["iris_cyan"], font=FONT_SMALL,
        )
        style.configure("TLabelframe",
            background=T["panel"], foreground=T["iris_cyan"],
            bordercolor=T["line"], relief="solid", borderwidth=1,
        )
        style.configure("TLabelframe.Label",
            background=T["panel"], foreground=T["iris_cyan"], font=FONT_BODY,
        )
        # 버튼 — 크레파스 굵은 아웃라인 느낌
        style.configure("Crayon.TButton",
            background=T["iris_lilac"], foreground=T["bg"],
            bordercolor=T["ink"], focuscolor=T["iris_pink"],
            relief="flat", padding=(14, 10), font=FONT_DISPLAY,
        )
        style.map("Crayon.TButton",
            background=[("active", T["iris_pink"]), ("pressed", T["iris_cyan"])],
            foreground=[("active", T["bg"])],
        )
        # 체크박스
        style.configure("TCheckbutton",
            background=T["panel"], foreground=T["ink"],
            focuscolor=T["iris_pink"], font=FONT_BODY,
        )
        style.map("TCheckbutton",
            background=[("active", T["panel_alt"])],
            foreground=[("active", T["iris_gold"])],
        )
        # 슬라이더
        style.configure("Iris.Horizontal.TScale",
            background=T["panel"], troughcolor=T["panel_alt"],
            bordercolor=T["line"], lightcolor=T["iris_cyan"],
            darkcolor=T["iris_lilac"],
        )

    def _make_mascot(self) -> ImageTk.PhotoImage | None:
        """mascot.png 가 있으면 아이보리 톤으로 tint 해서 헤더용 이미지로 반환."""
        p = Path(MASCOT_FILE)
        if not p.exists():
            return None
        try:
            img = Image.open(p).convert("RGBA")
            # 64px 높이로 축소
            ratio = 64 / img.height
            img = img.resize((int(img.width * ratio), 64), Image.LANCZOS)
            # 검은 선화를 아이보리 잉크 컬러로 tint
            r, g, b, a = img.split()
            ink = Image.new("RGB", img.size, THEME["ink"])
            tinted = Image.composite(ink, Image.new("RGB", img.size, THEME["bg"]), a)
            tinted.putalpha(a)
            return ImageTk.PhotoImage(tinted)
        except Exception as e:
            print(f"[마스코트 로드 실패] {e}")
            return None

    # ---------- UI ----------
    def _build_ui(self) -> None:
        T = THEME
        outer = ttk.Frame(self.root, padding=0)
        outer.pack(fill=tk.BOTH, expand=True)

        # ----- 상단 헤더: 마스코트 + 타이틀 -----
        header = tk.Frame(outer, bg=T["bg"], height=88)
        header.pack(fill=tk.X, padx=16, pady=(12, 6))
        header.pack_propagate(False)

        self._mascot_photo = self._make_mascot()
        if self._mascot_photo is not None:
            tk.Label(header, image=self._mascot_photo, bg=T["bg"], bd=0).pack(
                side=tk.LEFT, padx=(0, 14)
            )
        else:
            # 폴백: 유니코드 마스코트
            tk.Label(header, text="☠", bg=T["bg"], fg=T["ink"],
                     font=("Segoe UI Symbol", 48)).pack(side=tk.LEFT, padx=(0, 14))

        title_box = tk.Frame(header, bg=T["bg"])
        title_box.pack(side=tk.LEFT, anchor=tk.W)
        ttk.Label(title_box, text="OBJECT ✦ SOUND", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(title_box, text="~ a cute little haunted crayon ~", style="Sub.TLabel").pack(anchor=tk.W)

        # 헤더 하단 반사광 라인 (그라디언트 흉내)
        iris_bar = tk.Canvas(outer, height=4, bg=T["bg"], highlightthickness=0, bd=0)
        iris_bar.pack(fill=tk.X, padx=16, pady=(0, 10))
        self._draw_iris_bar(iris_bar)
        self.root.bind("<Configure>", lambda e: self._draw_iris_bar(iris_bar))

        # ----- 본문 -----
        main = ttk.Frame(outer, padding=(16, 6, 16, 10))
        main.pack(fill=tk.BOTH, expand=True)

        # ----- 좌측: 영상 -----
        left = ttk.Frame(main, style="Panel.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        video_wrap = tk.Frame(left, bg=T["ink"], padx=3, pady=3)  # 아이보리 크레파스 테두리
        video_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.video_label = tk.Label(video_wrap, background="#000000", bd=0)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ----- 우측: 컨트롤 -----
        right = ttk.Frame(main, style="Panel.TFrame", width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        inner = ttk.Frame(right, style="Panel.TFrame", padding=12)
        inner.pack(fill=tk.BOTH, expand=True)

        self.start_btn = ttk.Button(
            inner, text="▶  START", style="Crayon.TButton", command=self.toggle,
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 12))

        # 재생 모드
        mode_frame = ttk.LabelFrame(inner, text="♪  PLAY MODE", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            mode_frame, text="다중 사물 동시 재생",
            variable=self.multi_mode,
        ).pack(anchor=tk.W)
        ttk.Label(
            mode_frame, text="(끄면 한 번에 한 사운드만 재생)",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        # 임계값/쿨다운
        param_frame = ttk.LabelFrame(inner, text="✦  PARAMETERS", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(param_frame, text="신뢰도 임계값", style="Panel.TLabel").pack(anchor=tk.W)
        ttk.Scale(
            param_frame, from_=0.1, to=0.95,
            variable=self.conf_threshold, orient=tk.HORIZONTAL,
            style="Iris.Horizontal.TScale",
        ).pack(fill=tk.X)
        self.conf_label = ttk.Label(param_frame, text="0.50", style="Panel.TLabel")
        self.conf_label.pack(anchor=tk.E)
        self.conf_threshold.trace_add(
            "write",
            lambda *_: self.conf_label.config(text=f"{self.conf_threshold.get():.2f}"),
        )

        ttk.Label(param_frame, text="쿨다운 (초)", style="Panel.TLabel").pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(
            param_frame, from_=0.0, to=5.0,
            variable=self.cooldown_sec, orient=tk.HORIZONTAL,
            style="Iris.Horizontal.TScale",
        ).pack(fill=tk.X)
        self.cd_label = ttk.Label(param_frame, text="2.0", style="Panel.TLabel")
        self.cd_label.pack(anchor=tk.E)
        self.cooldown_sec.trace_add(
            "write",
            lambda *_: self.cd_label.config(text=f"{self.cooldown_sec.get():.1f}"),
        )

        # 감지 목록
        det_frame = ttk.LabelFrame(inner, text="☉  NOW WATCHING", padding=10)
        det_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.det_listbox = tk.Listbox(
            det_frame, height=6,
            bg=T["panel_alt"], fg=T["iris_gold"],
            selectbackground=T["iris_lilac"], selectforeground=T["bg"],
            highlightthickness=1, highlightbackground=T["line"],
            bd=0, relief=tk.FLAT, font=FONT_MONO,
        )
        self.det_listbox.pack(fill=tk.BOTH, expand=True)

        # 로드된 사운드
        self.snd_frame = ttk.LabelFrame(inner, text="♬  SOUND BANK", padding=10)
        self.snd_frame.pack(fill=tk.X)

        # 상태 바
        status_bar = tk.Label(
            self.root, textvariable=self.status_text,
            bg=T["panel"], fg=T["iris_cyan"],
            anchor=tk.W, padx=12, pady=6, bd=0,
            font=FONT_SMALL,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _draw_iris_bar(self, canvas: tk.Canvas) -> None:
        """헤더 아래 반사광(홀로그래픽) 그라디언트 라인."""
        canvas.delete("all")
        w = canvas.winfo_width() or 1000
        stops = [
            THEME["iris_lilac"], THEME["iris_cyan"],
            THEME["iris_pink"], THEME["iris_gold"], THEME["iris_lime"],
            THEME["iris_lilac"],
        ]
        seg = max(1, w // (len(stops) - 1))
        for i in range(len(stops) - 1):
            c1 = self._hex_to_rgb(stops[i])
            c2 = self._hex_to_rgb(stops[i + 1])
            for x in range(seg):
                t = x / seg
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
                canvas.create_line(
                    i * seg + x, 0, i * seg + x, 4,
                    fill=f"#{r:02x}{g:02x}{b:02x}",
                )

    @staticmethod
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _refresh_sound_panel(self) -> None:
        for widget in self.snd_frame.winfo_children():
            widget.destroy()
        self.snd_frame.config(text=f"♬  SOUND BANK ({len(self.sounds)})")
        if not self.sounds:
            ttk.Label(self.snd_frame, text="(비어 있음)", style="Muted.TLabel").pack(anchor=tk.W)
            return
        palette = [
            THEME["iris_lilac"], THEME["iris_cyan"],
            THEME["iris_pink"], THEME["iris_gold"], THEME["iris_lime"],
        ]
        for i, label in enumerate(self.sounds):
            row = tk.Frame(self.snd_frame, bg=THEME["panel"])
            row.pack(anchor=tk.W, fill=tk.X)
            tk.Label(row, text="♪", bg=THEME["panel"],
                     fg=palette[i % len(palette)], font=FONT_BODY).pack(side=tk.LEFT)
            tk.Label(row, text=f"  {label}", bg=THEME["panel"],
                     fg=THEME["ink"], font=FONT_BODY).pack(side=tk.LEFT)

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
        self.start_btn.config(text="■  STOP")
        self.status_text.set("실행 중")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        pygame.mixer.stop()
        self.start_btn.config(text="▶  START")
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
