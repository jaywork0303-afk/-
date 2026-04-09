// 사물 인식 사운드 플레이어 (브라우저 버전 / 서정월드)
// - TensorFlow.js + COCO-SSD 로 카메라 영상에서 사물 감지
// - 사용자가 만든 매핑 entries 만 트리거 (기본 톤 자동 매핑 없음)
// - 한 매핑은 여러 COCO 클래스를 묶어 카테고리로 사용 가능

// 자주 보이는 클래스들의 추천 톤 (entry 생성 시 기본값으로 사용)
const TONE_PRESETS = {
  person:       { freq: 261.63, type: 'sine' },
  'cell phone': { freq: 329.63, type: 'square' },
  cup:          { freq: 392.00, type: 'sine' },
  book:         { freq: 440.00, type: 'triangle' },
  bottle:       { freq: 523.25, type: 'sine' },
  laptop:       { freq: 587.33, type: 'square' },
  keyboard:     { freq: 659.25, type: 'sine' },
  mouse:        { freq: 698.46, type: 'triangle' },
  scissors:     { freq: 783.99, type: 'sine' },
  remote:       { freq: 880.00, type: 'square' },
  tv:           { freq: 349.23, type: 'triangle' },
  chair:        { freq: 466.16, type: 'sine' },
  dog:          { freq: 311.13, type: 'square' },
  cat:          { freq: 369.99, type: 'triangle' },
  banana:       { freq: 622.25, type: 'sine' },
  apple:        { freq: 739.99, type: 'triangle' },
  orange:       { freq: 830.61, type: 'sine' },
};

const TONE_DURATION = 0.25; // seconds

// 클래스 이름을 해시해서 고유한 톤 생성 (TONE_PRESETS 에 없는 클래스용)
function defaultToneForClass(cls) {
  if (TONE_PRESETS[cls]) return { ...TONE_PRESETS[cls] };
  let hash = 0;
  for (let i = 0; i < cls.length; i++) {
    hash = (hash * 31 + cls.charCodeAt(i)) | 0;
  }
  const freq = 220 + (Math.abs(hash) % 660); // 220~880 Hz
  const types = ['sine', 'triangle', 'square'];
  const type = types[Math.abs(hash) % types.length];
  return { freq, type };
}

// 새 entry 용 고유 ID
function generateId() {
  return `m_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

// 상태
let model = null;
let stream = null;
let running = false;
let audioContext = null;
const lastPlayed = {};
// label -> 재생 종료 예정 시각(초). 아직 재생 중이면 재트리거 금지.
const playingUntil = {};
// 현재 재생 중인 BufferSource 들 (정지 시 clean stop 용)
const activeSources = new Set();
// 현재 재생 중인 HTMLMediaElement 들 (비디오/폴백 경로용)
const activeElements = new Set();

// 추론은 메인 스레드에서 무겁기 때문에 비디오 페인트와 분리해서 운영한다.
// - latestPredictions: 마지막으로 받은 추론 결과 (rAF 가 매 프레임 그리기만 함)
// - inferenceInProgress: 추론 in-flight 플래그 (중복 호출 방지)
// - MIN_INFERENCE_INTERVAL_MS: 추론 최소 간격 (메인 스레드 숨돌릴 시간 확보)
let latestPredictions = [];
let inferenceInProgress = false;
let lastInferenceStart = 0;

// DOM
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');
const statusEl = document.getElementById('status');
const overlayMessage = document.getElementById('overlayMessage');
const multiModeEl = document.getElementById('multiMode');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const cdSlider = document.getElementById('cdSlider');
const cdValue = document.getElementById('cdValue');
const intervalSlider = document.getElementById('intervalSlider');
const intervalValue = document.getElementById('intervalValue');
const detectionList = document.getElementById('detectionList');
const soundList = document.getElementById('soundList');
const distanceMode = document.getElementById('distanceMode');
const falloffSlider = document.getElementById('falloffSlider');
const falloffValue = document.getElementById('falloffValue');
const videoContainer = document.getElementById('videoContainer');
const camSwitchBtn = document.getElementById('camSwitchBtn');
// 새 매핑 추가 UI
const mapName = document.getElementById('mapName');
const classPicker = document.getElementById('classPicker');
const classChips = document.getElementById('classChips');
const addToneBtn = document.getElementById('addToneBtn');
const addFileBtn = document.getElementById('addFileBtn');
const mapFileInput = document.getElementById('mapFileInput');
// 사물 직접 가르치기 (Teachable Machine) UI
const teachName = document.getElementById('teachName');
const teachBtn = document.getElementById('teachBtn');
const teachStatus = document.getElementById('teachStatus');
const taughtList = document.getElementById('taughtList');

// 현재 카메라 방향 ('user' = 전면, 'environment' = 후면)
let facingMode = 'user';

// COCO-SSD 의 80개 클래스 — 업로드 UI 의 사물 선택 드롭다운에 사용
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
  'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
  'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
  'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
  'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

// 클래스 드롭다운 채우기 (chip 추가용)
COCO_CLASSES.forEach((cls) => {
  const opt = document.createElement('option');
  opt.value = cls;
  opt.textContent = cls;
  classPicker.appendChild(opt);
});

// 사용자가 만든 사운드 매핑
// id -> entry
const customSounds = {};

// 사물 추가 진행 중인 chip 상태 (사용자가 "추가" 누르기 전)
const pendingChipClasses = new Set();

// ---- Teachable Machine (사용자가 카메라로 직접 가르친 사물) ----
// customClasses: { [name]: { name, exampleCount } }
//   KNN 분류기에 학습된 사용자 정의 클래스들.
//   chipPicker 드롭다운에 COCO 80 클래스와 함께 나타난다.
const customClasses = {};
let mobilenetModel = null;  // feature extractor (~4MB)
let knnModel = null;        // KNN classifier
let mobilenetLoading = null; // 중복 로드 방지용 Promise
const CUSTOM_CONF_THRESHOLD = 0.8;

// 슬라이더 값 표시
confSlider.addEventListener('input', () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});
cdSlider.addEventListener('input', () => {
  cdValue.textContent = parseFloat(cdSlider.value).toFixed(1) + '초';
});
intervalSlider.addEventListener('input', () => {
  intervalValue.textContent = intervalSlider.value + 'ms';
});
falloffSlider.addEventListener('input', () => {
  falloffValue.textContent = parseFloat(falloffSlider.value).toFixed(1);
});

// ---------- IndexedDB: 업로드 사운드 + KNN 학습 데이터 영구 저장 ----------
const DB_NAME = 'object-sound-db';
const STORE_NAME = 'sounds';
const KNN_STORE = 'knn';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 2);
    req.onupgradeneeded = (e) => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
      if (!db.objectStoreNames.contains(KNN_STORE)) {
        db.createObjectStore(KNN_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// KNN 학습 데이터 저장/로드
async function saveKnnDataset() {
  if (!knnModel) return;
  try {
    const dataset = knnModel.getClassifierDataset();
    const serialized = {};
    for (const [label, tensor] of Object.entries(dataset)) {
      const data = await tensor.data();
      serialized[label] = {
        data: Array.from(data),
        shape: tensor.shape,
      };
    }
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(KNN_STORE, 'readwrite');
      tx.objectStore(KNN_STORE).put(serialized, 'dataset');
      tx.objectStore(KNN_STORE).put(customClasses, 'classes');
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (e) {
    console.error('KNN 저장 실패', e);
  }
}

async function loadKnnDataset() {
  try {
    const db = await openDB();
    const raw = await new Promise((resolve, reject) => {
      const tx = db.transaction(KNN_STORE, 'readonly');
      const store = tx.objectStore(KNN_STORE);
      const datasetReq = store.get('dataset');
      const classesReq = store.get('classes');
      tx.oncomplete = () => {
        resolve({
          dataset: datasetReq.result,
          classes: classesReq.result,
        });
      };
      tx.onerror = () => reject(tx.error);
    });
    if (raw.classes) {
      Object.assign(customClasses, raw.classes);
    }
    if (raw.dataset && Object.keys(raw.dataset).length > 0) {
      // 텐서 복원은 KNN 초기화 후 수행
      return raw.dataset;
    }
  } catch (e) {
    console.error('KNN 로드 실패', e);
  }
  return null;
}

async function dbPut(label, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(value, label);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbDelete(label) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).delete(label);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbGetAll() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const result = {};
    const cursorReq = store.openCursor();
    cursorReq.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        result[cursor.key] = cursor.value;
        cursor.continue();
      } else {
        resolve(result);
      }
    };
    cursorReq.onerror = () => reject(cursorReq.error);
  });
}

// AudioContext 보장 (사용자 제스처 안에서 호출되어야 함)
function ensureAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioContext.state === 'suspended') {
    audioContext.resume();
  }
  return audioContext;
}

// 페이지 진입 직후 IndexedDB 에서 사운드 불러오기 (디코딩은 audioContext 생성 후)
// pendingSoundBuffers: id -> 직렬화된 entry 데이터 (audio context 생성 후 디코딩)
const pendingSoundBuffers = {};

async function loadSavedSoundsFromDB() {
  try {
    const all = await dbGetAll();
    for (const [key, value] of Object.entries(all)) {
      // 마이그레이션: 구버전 entry 는 key 가 COCO 클래스 이름이고 classes/id 필드 없음
      if (!value.id || !Array.isArray(value.classes)) {
        value.id = key;
        value.name = value.name || key;
        value.classes = [key];
        // kind 가 없으면 기본은 buffer (file 업로드 항목), 없으면 tone
        if (!value.kind) {
          value.kind = value.arrayBuffer ? 'buffer' : 'tone';
          if (value.kind === 'tone' && !value.tone) {
            value.tone = defaultToneForClass(key);
          }
        }
      }

      // tone 타입은 즉시 customSounds 로 (디코딩 불필요)
      if (value.kind === 'tone') {
        customSounds[value.id] = {
          id: value.id,
          name: value.name,
          classes: value.classes,
          kind: 'tone',
          tone: value.tone || defaultToneForClass(value.classes[0]),
          volume: value.volume ?? 1.0,
          rate: value.rate ?? 1.0,
        };
      } else {
        // buffer/element 는 audioContext 가 생성되면 디코딩
        pendingSoundBuffers[value.id] = value;
      }
    }
    renderSoundList();
  } catch (e) {
    console.error('사운드 DB 로드 실패', e);
  }
}

// 보류 중인 ArrayBuffer 들을 audioContext 로 디코딩해서 customSounds 에 옮긴다.
// decodeAudioData 실패 시 HTMLMediaElement 폴백 경로 시도.
async function decodePendingSounds() {
  if (!audioContext) return;
  const ids = Object.keys(pendingSoundBuffers);
  for (const id of ids) {
    const data = pendingSoundBuffers[id];
    try {
      const decoded = await createSoundDataFromArrayBuffer(
        data.arrayBuffer,
        data.mimeType,
      );
      customSounds[id] = {
        id: data.id || id,
        name: data.name,
        classes: data.classes,
        kind: decoded.kind,
        buffer: decoded.buffer,
        mediaElement: decoded.mediaElement,
        blobUrl: decoded.blobUrl,
        duration: decoded.duration,
        mimeType: data.mimeType,
        soundFileName: data.soundFileName || data.name,
        volume: data.volume ?? 1.0,
        rate: data.rate ?? 1.0,
      };
      delete pendingSoundBuffers[id];
    } catch (e) {
      console.error(`'${id}' 디코딩 실패`, e);
    }
  }
  renderSoundList();
}

// 매핑 목록 UI 표시 — 사용자가 만든 entries 만 보여준다 (기본 톤 80개 자동 추가 X)
function renderSoundList() {
  soundList.innerHTML = '';

  const entries = [
    ...Object.values(customSounds),
    ...Object.values(pendingSoundBuffers),
  ].sort((a, b) => (a.name || '').localeCompare(b.name || ''));

  if (entries.length === 0) {
    const li = document.createElement('li');
    li.className = 'empty-hint';
    li.textContent = '위에서 사물을 골라 사운드를 추가해보세요 ♡';
    soundList.appendChild(li);
    return;
  }

  for (const entry of entries) {
    soundList.appendChild(buildEntryListItem(entry));
  }
}

// 매핑 entry 한 개에 대한 li 요소 생성
function buildEntryListItem(entry) {
  const li = document.createElement('li');
  li.className = 'has-sound sound-item';

  // 첫 줄: 아이콘 + 이름 + 클래스 chip + 액션
  const row = document.createElement('div');
  row.className = 'sound-item-row';

  const titleWrap = document.createElement('div');
  titleWrap.className = 'sound-title';

  const title = document.createElement('span');
  const icon = entry.kind === 'tone' ? '♪' : '🔊';
  title.innerHTML = `${icon} <strong>${escapeHtml(entry.name)}</strong>`;
  if (entry.soundFileName && entry.soundFileName !== entry.name) {
    title.innerHTML += ` <em class="filename">— ${escapeHtml(entry.soundFileName)}</em>`;
  }
  titleWrap.appendChild(title);

  // 클래스 chip 표시 (이 매핑이 트리거되는 사물들)
  const classRow = document.createElement('div');
  classRow.className = 'entry-classes';
  for (const cls of entry.classes || []) {
    const chip = document.createElement('span');
    chip.className = 'class-tag';
    chip.textContent = cls;
    classRow.appendChild(chip);
  }
  titleWrap.appendChild(classRow);

  row.appendChild(titleWrap);

  const actions = document.createElement('span');
  actions.className = 'sound-actions';

  // 미리듣기
  const testBtn = document.createElement('button');
  testBtn.textContent = '▶';
  testBtn.title = '미리듣기';
  testBtn.onclick = (e) => {
    e.stopPropagation();
    ensureAudioContext();
    if (Object.keys(pendingSoundBuffers).length > 0) decodePendingSounds();
    playEntry(customSounds[entry.id] || entry, 1.0);
  };
  actions.appendChild(testBtn);

  // 볼륨/속도 편집 토글 (tone 도 적용 가능)
  const editBtn = document.createElement('button');
  editBtn.textContent = '⚙';
  editBtn.title = '볼륨/속도 조정';
  editBtn.onclick = (e) => {
    e.stopPropagation();
    li.classList.toggle('expanded');
  };
  actions.appendChild(editBtn);

  // 삭제
  const delBtn = document.createElement('button');
  delBtn.textContent = '✕';
  delBtn.title = '삭제';
  delBtn.onclick = async (e) => {
    e.stopPropagation();
    const live = customSounds[entry.id];
    if (live && live.kind === 'element' && live.blobUrl) {
      URL.revokeObjectURL(live.blobUrl);
    }
    delete customSounds[entry.id];
    delete pendingSoundBuffers[entry.id];
    await dbDelete(entry.id);
    renderSoundList();
  };
  actions.appendChild(delBtn);

  row.appendChild(actions);
  li.appendChild(row);

  // 편집 영역 (펼쳤을 때만)
  const edit = document.createElement('div');
  edit.className = 'sound-edit';

  const volume = entry.volume ?? 1.0;
  const rate = entry.rate ?? 1.0;

  const volLabel = document.createElement('label');
  volLabel.innerHTML = `볼륨 <span class="val">${volume.toFixed(2)}</span>`;
  const volSlider = document.createElement('input');
  volSlider.type = 'range';
  volSlider.min = '0';
  volSlider.max = '2';
  volSlider.step = '0.05';
  volSlider.value = String(volume);
  volSlider.oninput = async () => {
    const v = parseFloat(volSlider.value);
    volLabel.querySelector('.val').textContent = v.toFixed(2);
    if (customSounds[entry.id]) customSounds[entry.id].volume = v;
    if (pendingSoundBuffers[entry.id]) pendingSoundBuffers[entry.id].volume = v;
    await persistEntryOptions(entry.id);
  };
  volLabel.appendChild(volSlider);
  edit.appendChild(volLabel);

  const rateLabel = document.createElement('label');
  rateLabel.innerHTML = `재생 속도 <span class="val">${rate.toFixed(2)}x</span>`;
  const rateSlider = document.createElement('input');
  rateSlider.type = 'range';
  rateSlider.min = '0.25';
  rateSlider.max = '3';
  rateSlider.step = '0.05';
  rateSlider.value = String(rate);
  rateSlider.oninput = async () => {
    const r = parseFloat(rateSlider.value);
    rateLabel.querySelector('.val').textContent = `${r.toFixed(2)}x`;
    if (customSounds[entry.id]) customSounds[entry.id].rate = r;
    if (pendingSoundBuffers[entry.id]) pendingSoundBuffers[entry.id].rate = r;
    await persistEntryOptions(entry.id);
  };
  rateLabel.appendChild(rateSlider);
  edit.appendChild(rateLabel);

  li.appendChild(edit);

  return li;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// 볼륨/속도만 IndexedDB 에 다시 저장
async function persistEntryOptions(id) {
  try {
    const all = await dbGetAll();
    const existing = all[id];
    if (!existing) return;
    const live = customSounds[id] || pendingSoundBuffers[id];
    if (!live) return;
    await dbPut(id, { ...existing, volume: live.volume ?? 1, rate: live.rate ?? 1 });
  } catch (e) {
    console.error('옵션 저장 실패', e);
  }
}

// 업로드 처리
// ArrayBuffer 를 HTMLMediaElement (Audio) 로 재생 가능한지 검증하고
// 재사용 가능한 element + blob URL + duration 을 반환. 실패하면 에러 throw.
// 비디오 파일이나 Web Audio API 가 디코드 못 하는 포맷에 대한 폴백 경로.
// 핵심: element 를 재사용해야 iOS/모바일의 동시 미디어 엘리먼트 제한을
// 회피할 수 있음. 매 재생마다 새 element 를 만들면 2~3회 이후 무음이 됨.
async function probeAsMediaElement(arrayBuffer, mimeType) {
  const blob = new Blob([arrayBuffer], { type: mimeType || 'audio/mpeg' });
  const url = URL.createObjectURL(blob);
  const el = document.createElement('audio');
  el.preload = 'auto';
  el.src = url;

  try {
    // canplaythrough / loadeddata 까지 기다려서 duration 을 정확히 얻는다
    // (loadedmetadata 만으론 일부 비디오 컨테이너에서 duration 이 Infinity)
    await new Promise((resolve, reject) => {
      let settled = false;
      const ok = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      const fail = (msg) => {
        if (settled) return;
        settled = true;
        reject(new Error(msg));
      };
      el.oncanplaythrough = ok;
      el.onloadeddata = ok;
      el.onerror = () => fail('이 파일은 브라우저가 재생할 수 없는 형식입니다');
      // 5초 안에 canplaythrough 가 안 와도 진행 (메타데이터만 있어도 플레이는 가능)
      setTimeout(() => {
        if (el.readyState >= 1) ok();
        else fail('파일 로딩 시간 초과');
      }, 5000);
      el.load();
    });

    let duration = el.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      // 일부 비디오 컨테이너는 duration 을 알 수 없음 → 재생 중 실시간 측정 fallback
      duration = 0;
    }
    return { el, url, duration };
  } catch (e) {
    URL.revokeObjectURL(url);
    throw e;
  }
}

// ArrayBuffer 를 디코딩 가능한 사운드 데이터로 변환
// (kind, buffer / mediaElement, blobUrl, duration 반환)
async function createSoundDataFromArrayBuffer(arrayBuffer, mimeType) {
  if (mimeType && mimeType.startsWith('image/')) {
    throw new Error('이미지 파일은 사운드로 사용할 수 없어요. 오디오/비디오 파일을 골라주세요.');
  }
  // 1) Web Audio API 디코딩 시도
  try {
    const buffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    return { kind: 'buffer', buffer };
  } catch (e) {
    console.warn('decodeAudioData 실패, HTMLMediaElement 폴백 시도', e);
  }
  // 2) HTMLMediaElement 폴백 (비디오/컨테이너 포맷)
  const { el, url, duration } = await probeAsMediaElement(arrayBuffer, mimeType);
  return { kind: 'element', mediaElement: el, blobUrl: url, duration };
}

// 클래스 chip 추가/제거
function addChip(cls) {
  if (!cls || pendingChipClasses.has(cls)) return;
  pendingChipClasses.add(cls);
  renderChips();
}
function removeChip(cls) {
  pendingChipClasses.delete(cls);
  renderChips();
}
function renderChips() {
  classChips.innerHTML = '';
  for (const cls of pendingChipClasses) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = cls;
    const x = document.createElement('button');
    x.type = 'button';
    x.textContent = '×';
    x.title = '제거';
    x.onclick = () => removeChip(cls);
    chip.appendChild(x);
    classChips.appendChild(chip);
  }
}
function clearAddForm() {
  pendingChipClasses.clear();
  renderChips();
  mapName.value = '';
  classPicker.value = '';
}

classPicker.addEventListener('change', () => {
  const cls = classPicker.value;
  if (cls) {
    addChip(cls);
    classPicker.value = '';
  }
});

// 매핑 entry 생성 (tone 또는 buffer/element) 후 저장
async function createMappingEntry({ kind, file }) {
  if (pendingChipClasses.size === 0) {
    alert('사물을 1개 이상 선택해주세요.');
    return;
  }

  ensureAudioContext();

  const classes = [...pendingChipClasses];
  const userName = mapName.value.trim();
  const name = userName || classes.join(' / ');
  const id = generateId();

  const baseEntry = {
    id,
    name,
    classes,
    volume: 1.0,
    rate: 1.0,
  };

  let entry = null;
  let dbValue = null;

  if (kind === 'tone') {
    const tone = defaultToneForClass(classes[0]);
    entry = { ...baseEntry, kind: 'tone', tone };
    dbValue = { ...baseEntry, kind: 'tone', tone };
  } else if (kind === 'file' && file) {
    statusEl.textContent = `'${file.name}' 디코딩 중...`;
    try {
      const arrayBuffer = await file.arrayBuffer();
      const decoded = await createSoundDataFromArrayBuffer(arrayBuffer, file.type);
      entry = {
        ...baseEntry,
        kind: decoded.kind,
        buffer: decoded.buffer,
        mediaElement: decoded.mediaElement,
        blobUrl: decoded.blobUrl,
        duration: decoded.duration,
        soundFileName: file.name,
        mimeType: file.type,
      };
      dbValue = {
        ...baseEntry,
        kind: decoded.kind, // buffer 또는 element
        arrayBuffer,
        mimeType: file.type,
        soundFileName: file.name,
      };
    } catch (e) {
      console.error(e);
      statusEl.textContent = '사운드 로드 실패';
      alert(e.message);
      return;
    }
  } else {
    return;
  }

  customSounds[id] = entry;
  try {
    await dbPut(id, dbValue);
  } catch (e) {
    console.error('DB 저장 실패', e);
  }

  statusEl.textContent = `'${name}' 매핑 추가됨 ♡`;
  clearAddForm();
  renderSoundList();
}

// 추가 버튼: 기본 톤
addToneBtn.addEventListener('click', () => {
  createMappingEntry({ kind: 'tone' });
});

// 추가 버튼: 파일 → file picker → change → 매핑 생성
addFileBtn.addEventListener('click', () => {
  if (pendingChipClasses.size === 0) {
    alert('먼저 사물을 1개 이상 선택해주세요.');
    return;
  }
  mapFileInput.click();
});
mapFileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await createMappingEntry({ kind: 'file', file });
  mapFileInput.value = '';
});

// 페이지 로드 시 저장된 사운드 불러오기
loadSavedSoundsFromDB();

// ---------- Teachable Machine: 사용자 사물 학습 ----------

// 드롭다운을 COCO + customClasses 로 재구성
function rebuildClassPicker() {
  const current = classPicker.value;
  classPicker.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '+ 사물 추가하기...';
  classPicker.appendChild(placeholder);

  // 커스텀 학습 사물 먼저
  const customNames = Object.keys(customClasses);
  if (customNames.length > 0) {
    const group1 = document.createElement('optgroup');
    group1.label = '✨ 내가 가르친 사물';
    for (const name of customNames) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = `✨ ${name}`;
      group1.appendChild(opt);
    }
    classPicker.appendChild(group1);
  }

  const group2 = document.createElement('optgroup');
  group2.label = 'COCO 80 클래스';
  for (const cls of COCO_CLASSES) {
    const opt = document.createElement('option');
    opt.value = cls;
    opt.textContent = cls;
    group2.appendChild(opt);
  }
  classPicker.appendChild(group2);

  classPicker.value = current;
}

// MobileNet + KNN 로드 (lazy, 중복 로드 방지)
async function ensureTeachModels() {
  if (mobilenetModel && knnModel) return;
  if (mobilenetLoading) return mobilenetLoading;

  mobilenetLoading = (async () => {
    if (typeof mobilenet === 'undefined' || typeof knnClassifier === 'undefined') {
      throw new Error('학습 모듈을 불러오지 못했습니다. 네트워크를 확인해주세요.');
    }
    statusEl.textContent = 'MobileNet 로딩 중... (최초 1회)';
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 0.5 });
    knnModel = knnClassifier.create();

    // 저장된 학습 데이터 복원
    const savedDataset = await loadKnnDataset();
    if (savedDataset) {
      const reconstructed = {};
      for (const [label, { data, shape }] of Object.entries(savedDataset)) {
        reconstructed[label] = tf.tensor(data, shape);
      }
      knnModel.setClassifierDataset(reconstructed);
    }
    statusEl.textContent = '학습 모듈 준비 완료';
  })();

  try {
    await mobilenetLoading;
  } finally {
    mobilenetLoading = null;
  }
}

// 학습 상태 UI
function setTeachStatus(text) {
  if (!text) {
    teachStatus.classList.add('empty');
    teachStatus.textContent = '';
  } else {
    teachStatus.classList.remove('empty');
    teachStatus.textContent = text;
  }
}

function renderTaughtList() {
  taughtList.innerHTML = '';
  const names = Object.keys(customClasses);
  if (names.length === 0) {
    const li = document.createElement('li');
    li.className = 'empty-hint';
    li.textContent = '아직 가르친 사물이 없어요';
    taughtList.appendChild(li);
    return;
  }
  for (const name of names) {
    const info = customClasses[name];
    const li = document.createElement('li');
    const span = document.createElement('span');
    span.innerHTML = `✨ <strong>${escapeHtml(name)}</strong> <em class="filename">(${info.exampleCount}개 샘플)</em>`;
    li.appendChild(span);

    const delBtn = document.createElement('button');
    delBtn.textContent = '✕';
    delBtn.title = '삭제';
    delBtn.onclick = async () => {
      delete customClasses[name];
      if (knnModel) {
        try {
          knnModel.clearClass(name);
        } catch (_) {}
      }
      await saveKnnDataset();
      renderTaughtList();
      rebuildClassPicker();
    };
    li.appendChild(delBtn);
    taughtList.appendChild(li);
  }
}

// 학습 버튼: 3초간 카메라로 프레임을 캡처해 KNN 에 추가
teachBtn.addEventListener('click', async () => {
  const name = teachName.value.trim();
  if (!name) {
    alert('사물 이름을 입력해주세요.');
    return;
  }
  if (!running) {
    alert('먼저 카메라를 시작해주세요. (▶ 시작 버튼)');
    return;
  }

  teachBtn.disabled = true;
  setTeachStatus('학습 모듈 준비 중...');

  try {
    await ensureTeachModels();

    const duration = 3000;
    const frameCount = 30;
    const interval = duration / frameCount;

    setTeachStatus(`'${name}' 학습 중... 카메라에 사물을 크게 비춰주세요!`);

    let added = 0;
    for (let i = 0; i < frameCount; i++) {
      if (video.readyState < 2) {
        await new Promise((r) => setTimeout(r, interval));
        continue;
      }
      const features = mobilenetModel.infer(video, true); // embedding
      knnModel.addExample(features, name);
      features.dispose();
      added++;
      setTeachStatus(`학습 중... ${i + 1}/${frameCount}`);
      await new Promise((r) => setTimeout(r, interval));
    }

    // 누적 기록
    const prevCount = customClasses[name]?.exampleCount || 0;
    customClasses[name] = {
      name,
      exampleCount: prevCount + added,
    };
    await saveKnnDataset();

    setTeachStatus(`✓ '${name}' 학습 완료! (총 ${customClasses[name].exampleCount}개 샘플)`);
    teachName.value = '';
    renderTaughtList();
    rebuildClassPicker();
  } catch (e) {
    console.error(e);
    setTeachStatus(`오류: ${e.message}`);
  } finally {
    teachBtn.disabled = false;
  }
});

// 페이지 로드 시 저장된 커스텀 클래스 목록 복원 (텐서는 lazy)
(async () => {
  try {
    const db = await openDB();
    const raw = await new Promise((resolve, reject) => {
      const tx = db.transaction(KNN_STORE, 'readonly');
      const store = tx.objectStore(KNN_STORE);
      const classesReq = store.get('classes');
      tx.oncomplete = () => resolve(classesReq.result);
      tx.onerror = () => reject(tx.error);
    });
    if (raw) {
      Object.assign(customClasses, raw);
    }
  } catch (_) {
    /* ignore */
  }
  rebuildClassPicker();
  renderTaughtList();
  setTeachStatus('');
})();

// 카메라 전환 버튼: facingMode 를 토글한 뒤 재시작
camSwitchBtn.addEventListener('click', async () => {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  // 버튼 라벨 업데이트 (시각적 힌트)
  camSwitchBtn.title = facingMode === 'user'
    ? '전면 카메라 (클릭: 후면으로)'
    : '후면 카메라 (클릭: 전면으로)';

  // unflip 클래스는 즉시 토글 (카메라 안 켜져 있어도 다음 시작 시 반영)
  applyFacingTransform();

  // 실행 중이면 스트림을 새 facingMode 로 교체
  if (running) {
    camSwitchBtn.disabled = true;
    statusEl.textContent = '카메라 전환 중...';
    try {
      await startCamera();
      statusEl.textContent = `실행 중 — ${facingMode === 'user' ? '전면' : '후면'}`;
    } catch (e) {
      console.error('카메라 전환 실패', e);
      statusEl.textContent = `카메라 전환 실패: ${e.message}`;
      // 실패 시 원래 facingMode 로 복구 시도
      facingMode = facingMode === 'user' ? 'environment' : 'user';
      applyFacingTransform();
      try {
        await startCamera();
      } catch (_) {
        /* 복구도 실패하면 포기 */
      }
    } finally {
      camSwitchBtn.disabled = false;
    }
  }
});

// 사물(클래스)에 사용자가 만든 매핑이 있는지
function hasSound(cls) {
  for (const entry of Object.values(customSounds)) {
    if (entry.classes && entry.classes.includes(cls)) return true;
  }
  return false;
}

// 특정 클래스를 트리거하는 모든 entry 들 반환
function findEntriesForClass(cls) {
  const result = [];
  for (const entry of Object.values(customSounds)) {
    if (entry.classes && entry.classes.includes(cls)) result.push(entry);
  }
  return result;
}

// 사람 bbox 들 중 가장 가까운 것을 찾아 근접도(0~1) 계산
// 1.0 = 사람 위 / 매우 가까움, 0에 가까움 = 멀리 떨어짐
function computeProximity(objBox, personBoxes, falloff) {
  if (personBoxes.length === 0) return 1.0;

  const [ox, oy, ow, oh] = objBox;
  const ocx = ox + ow / 2;
  const ocy = oy + oh / 2;

  let minDist = Infinity;
  let bestPersonScale = 1;

  for (const pb of personBoxes) {
    const [px, py, pw, ph] = pb;
    const pcx = px + pw / 2;
    const pcy = py + ph / 2;
    const dx = pcx - ocx;
    const dy = pcy - ocy;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < minDist) {
      minDist = distance;
      // 사람 bbox 의 절반 대각선을 단위 척도로 (사람 한 명 정도 거리 = 1)
      bestPersonScale = Math.sqrt(pw * pw + ph * ph) / 2;
    }
  }

  const proximityUnits = minDist / Math.max(bestPersonScale, 1);
  // 부드러운 감쇠 곡선: 1 / (1 + (d/1.5)^falloff)
  // proximity 0 → 1.0,  1.5 → 0.5,  멀수록 0 에 수렴
  return 1 / (1 + Math.pow(proximityUnits / 1.5, falloff));
}

// ---- 재생 (entry 기반) ----

// 톤 entry 재생: 오실레이터 + 짧은 envelope
function playToneEntry(entry, multiplier) {
  if (!audioContext) return 0;
  const tone = entry.tone || defaultToneForClass(entry.classes[0]);
  const volume = entry.volume ?? 1.0;

  const osc = audioContext.createOscillator();
  const gain = audioContext.createGain();
  osc.type = tone.type || 'sine';
  osc.frequency.value = tone.freq || 440;

  const now = audioContext.currentTime;
  const peak = Math.max(0.0001, 0.22 * volume * multiplier);
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(peak, now + 0.02);
  gain.gain.exponentialRampToValueAtTime(peak * 0.001, now + TONE_DURATION);

  osc.connect(gain).connect(audioContext.destination);
  osc.start(now);
  osc.stop(now + TONE_DURATION);
  return TONE_DURATION;
}

// AudioBuffer entry 재생 (Web Audio API 경로)
function playBufferEntry(entry, multiplier) {
  if (!audioContext || !entry.buffer) return 0;
  const volume = entry.volume ?? 1.0;
  const rate = entry.rate ?? 1.0;

  const source = audioContext.createBufferSource();
  source.buffer = entry.buffer;
  source.playbackRate.value = rate;

  const gain = audioContext.createGain();
  gain.gain.value = volume * multiplier;

  source.connect(gain).connect(audioContext.destination);

  activeSources.add(source);
  source.onended = () => {
    activeSources.delete(source);
    if ((playingUntil[entry.id] ?? 0) <= performance.now() / 1000 + 0.05) {
      delete playingUntil[entry.id];
    }
  };

  source.start();
  return entry.buffer.duration / rate;
}

// HTMLMediaElement entry 재생 (비디오/폴백 포맷)
// 엔트리당 단일 엘리먼트 재사용 → iOS 모바일 동시 미디어 제한 회피
function playElementEntry(entry, multiplier) {
  const src = entry.mediaElement;
  if (!src) return 0;
  const volume = entry.volume ?? 1.0;
  const rate = entry.rate ?? 1.0;

  // 진짜 동시 재생: 매번 clone 해서 재생 → 같은 파일이 겹쳐 들린다
  // (원본 src 는 메타데이터/디코딩 캐시 역할만)
  const el = src.cloneNode(true);
  el.volume = Math.max(0, Math.min(1, volume * multiplier));
  el.playbackRate = rate;
  el.currentTime = 0;

  activeElements.add(el);
  el.onended = () => {
    activeElements.delete(el);
    el.src = '';
  };

  const p = el.play();
  if (p && typeof p.catch === 'function') {
    p.catch((e) => {
      console.error('element 재생 실패', e);
      activeElements.delete(el);
    });
  }

  const realDuration =
    Number.isFinite(src.duration) && src.duration > 0 ? src.duration : entry.duration;
  return (realDuration > 0 ? realDuration : 1) / rate;
}

// 통합 재생 디스패처
// 같은 entry 라도 buffer/tone/element 모두 진짜 동시 재생되도록
// playingUntil 차단을 제거 (Web Audio / clone 으로 자연스럽게 겹친다)
function playEntry(entry, multiplier = 1.0) {
  if (!entry) return;
  const nowSec = performance.now() / 1000;

  let duration = 0;
  if (entry.kind === 'tone') duration = playToneEntry(entry, multiplier);
  else if (entry.kind === 'buffer') duration = playBufferEntry(entry, multiplier);
  else if (entry.kind === 'element') duration = playElementEntry(entry, multiplier);

  if (duration > 0) {
    playingUntil[entry.id] = nowSec + duration;
  }
}

// 모든 재생 중 사운드 즉시 중단 + 상태 리셋
function stopAllSounds() {
  for (const src of activeSources) {
    try {
      src.stop();
    } catch (_) {
      /* already stopped */
    }
  }
  activeSources.clear();

  // element 들은 재사용되므로 정지만 하고 Set 에서만 제거
  for (const el of activeElements) {
    try {
      el.pause();
      el.currentTime = 0;
    } catch (_) {
      /* ignore */
    }
  }
  activeElements.clear();

  for (const k of Object.keys(playingUntil)) delete playingUntil[k];
  for (const k of Object.keys(lastPlayed)) delete lastPlayed[k];
}

// 감지된 prediction 들에 대해 모드/쿨다운/거리에 따라 사운드 재생
// predictions: [{ class, score, bbox, proximityVolume }]
function playSounds(predictions) {
  const now = performance.now() / 1000;
  const cd = parseFloat(cdSlider.value);

  // 클래스별 최고 볼륨 (거리 모드 multiplier) 계산
  const classVol = new Map();
  for (const p of predictions) {
    const v = p.proximityVolume ?? 1;
    if (!classVol.has(p.class) || classVol.get(p.class) < v) {
      classVol.set(p.class, v);
    }
  }

  // 매핑 entry 별로 후보 추출. 같은 entry 가 여러 클래스로 트리거되면
  // 그 중 가장 큰 볼륨을 사용.
  const entryCandidates = new Map(); // entry.id -> { entry, multiplier }
  for (const [cls, vol] of classVol) {
    for (const entry of findEntriesForClass(cls)) {
      const existing = entryCandidates.get(entry.id);
      if (!existing || existing.multiplier < vol) {
        entryCandidates.set(entry.id, { entry, multiplier: vol });
      }
    }
  }

  // 쿨다운/재생중 필터
  const candidates = [];
  for (const c of entryCandidates.values()) {
    const id = c.entry.id;
    if (now < (playingUntil[id] || 0)) continue;
    if (now - (lastPlayed[id] || 0) < cd) continue;
    candidates.push(c);
  }

  if (multiModeEl.checked) {
    // 다중 모드: 모두 재생
    for (const { entry, multiplier } of candidates) {
      playEntry(entry, multiplier);
      lastPlayed[entry.id] = now;
    }
  } else {
    // 단일 모드: 가장 큰 볼륨의 entry 1개만
    if (now - (lastPlayed.__last__ || 0) < TONE_DURATION) return;
    candidates.sort((a, b) => b.multiplier - a.multiplier);
    if (candidates.length > 0) {
      const { entry, multiplier } = candidates[0];
      playEntry(entry, multiplier);
      lastPlayed[entry.id] = now;
      lastPlayed.__last__ = now;
    }
  }

  // ── TTS 폴백: 매핑이 없는 클래스는 클래스 이름을 동시에 읽어준다 ──
  for (const [cls, vol] of classVol) {
    if (findEntriesForClass(cls).length > 0) continue; // 이미 매핑 있음
    const ttsId = `__tts__${cls}`;
    if (now - (lastPlayed[ttsId] || 0) < cd) continue;
    // meSpeak 가 준비됐으면 진짜 병렬, 아니면 speechSynthesis 폴백
    if (__meSpeakReady) {
      speakParallel(cls, vol);
    } else {
      speakClassName(cls, vol);
    }
    lastPlayed[ttsId] = now;
  }
}

// ══════════════════════════════════════════════════════════════
// meSpeak: 진짜 동시 재생 가능한 TTS
// speechSynthesis 와 달리 raw PCM 을 받아서 AudioBufferSourceNode 로
// 여러 개를 동시에 재생할 수 있다 (브라우저 단일 음성 채널 한계 우회).
// ══════════════════════════════════════════════════════════════
let __meSpeakReady = false;
const __ttsBufferCache = new Map(); // class name -> AudioBuffer

async function ensureMeSpeak() {
  if (__meSpeakReady) return true;
  if (typeof meSpeak === 'undefined') return false;
  try {
    // meSpeak v2: loadConfig 는 빈 함수, isConfigLoaded 는 항상 true.
    // loadVoice(url, callback) 의 callback 시그니처: (success, msgOrName)
    await new Promise((resolve, reject) => {
      meSpeak.loadVoice('en/en', (success, msg) => {
        if (success) resolve(msg);
        else reject(new Error('voice load failed: ' + msg));
      });
    });
    __meSpeakReady = true;
    console.log('[meSpeak] 준비 완료 — 진짜 병렬 TTS 활성화');
    return true;
  } catch (e) {
    console.warn('meSpeak 초기화 실패:', e);
    return false;
  }
}
// 페이지 로드 즉시 meSpeak 초기화 (START 안 눌러도 매핑 없는 클래스 TTS 가능)
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ensureMeSpeak());
  } else {
    ensureMeSpeak();
  }
}

// 클래스 이름을 PCM buffer 로 합성해서 캐시에 저장
// meSpeak v2: speak(text, {rawdata:'array'}, callback) — callback(success, id, data)
async function getOrSynthBuffer(cls) {
  if (__ttsBufferCache.has(cls)) return __ttsBufferCache.get(cls);
  if (!__meSpeakReady) {
    const ok = await ensureMeSpeak();
    if (!ok) return null;
  }
  const wav = await new Promise((resolve) => {
    meSpeak.speak(cls, {
      rawdata: 'array',
      speed: 175,
      pitch: 50,
      amplitude: 100,
    }, (success, id, data) => {
      resolve(success ? data : null);
    });
  });
  if (!wav || !wav.length) return null;
  const u8 = wav instanceof Uint8Array ? wav : new Uint8Array(wav);
  const ctx = ensureAudioContext();
  try {
    const buf = await ctx.decodeAudioData(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
    __ttsBufferCache.set(cls, buf);
    return buf;
  } catch (e) {
    console.warn('TTS decode 실패:', cls, e);
    return null;
  }
}

// 진짜 병렬 재생: 새 AudioBufferSourceNode 를 매번 생성해 동시에 출력
async function speakParallel(cls, volume = 1.0) {
  const buf = await getOrSynthBuffer(cls);
  if (!buf) return;
  const ctx = ensureAudioContext();
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const gain = ctx.createGain();
  gain.gain.value = Math.max(0.15, Math.min(1, volume));
  src.connect(gain).connect(ctx.destination);
  src.start();
}

// 모바일에서 speechSynthesis 를 사용자 제스처 안에서 "깨우기" 위한 helper.
// iOS Safari 는 첫 speak() 가 사용자 제스처 안에서 호출되지 않으면 이후 호출도
// 소리가 나오지 않는다. 또 일부 안드로이드 Chrome 은 getVoices() 가 비동기라
// voiceschanged 이벤트를 기다려야 한다.
let __speechUnlocked = false;
function unlockSpeechSynthesis() {
  if (__speechUnlocked) return;
  if (!('speechSynthesis' in window)) return;
  try {
    // 일부 브라우저는 resume() 도 필요
    window.speechSynthesis.resume();
    const u = new SpeechSynthesisUtterance(' ');
    u.volume = 0.01;
    u.rate = 1;
    u.pitch = 1;
    window.speechSynthesis.speak(u);
    // 음성 리스트 비동기 로딩 트리거
    window.speechSynthesis.getVoices();
    if (typeof window.speechSynthesis.onvoiceschanged !== 'undefined') {
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }
    __speechUnlocked = true;
  } catch (e) {
    console.warn('TTS unlock 실패:', e);
  }
}

// ── Web Speech API TTS: 감지된 모든 사물 이름을 동시에 겹쳐서 읽는다 ──
// Web Speech API 는 단일 큐라 기본적으로 utterance 가 직렬로 재생된다.
// "겹쳐서" 들리도록 하려면 매번 cancel() 하고 새 utterance 를 즉시 speak 해
// 이전 음성을 끊고 새로 시작 + 다른 utterance 들도 큐에 같이 쌓아 빠르게
// 연속 발음시킨다. (브라우저 한계상 진짜 동시 재생은 불가)
function speakClassName(cls, volume = 1.0) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(cls);
  u.rate = 1.25;
  u.pitch = 1.0;
  u.volume = Math.max(0.15, Math.min(1, volume));
  u.lang = /^[a-zA-Z\s_-]+$/.test(cls) ? 'en-US' : 'ko-KR';
  try {
    // 큐에 쌓아둔 채로 그대로 추가 — 순차로 빠르게 모두 읽는다
    window.speechSynthesis.speak(u);
  } catch (e) {
    console.warn('TTS 실패:', e);
  }
}

async function loadModel() {
  // WebGL 백엔드 강제 — CPU 백엔드면 매우 느림
  statusEl.textContent = 'GPU 백엔드 초기화 중...';
  try {
    await tf.setBackend('webgl');
  } catch (e) {
    console.warn('WebGL 백엔드 사용 불가, CPU 폴백', e);
  }
  await tf.ready();
  console.log('TF backend:', tf.getBackend());

  statusEl.textContent = '모델 로딩 중... (최초 1회)';
  // lite_mobilenet_v2: 가장 가벼운 변형, 기본 모델 대비 약 3배 빠름
  model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  statusEl.textContent = `모델 로드 완료 (${tf.getBackend()})`;
}

async function startCamera() {
  // 이전 스트림이 있으면 먼저 정리 (카메라 전환 시)
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  // 해상도를 낮추면 추론 속도가 크게 빨라짐 (입력 텐서가 작아짐)
  // 화면에는 CSS 로 100% 늘려 표시되므로 화질 차이는 거의 안 보임
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 384 },
      height: { ideal: 288 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => {
    if (video.readyState >= 1) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // 전면 카메라는 브라우저가 프리뷰를 좌우반전시키는 경우가 많아서
  // unflip 클래스로 CSS scaleX(-1) 적용 → 실제 좌우 복원
  // 후면 카메라는 그대로 표시
  applyFacingTransform();
}

// facingMode 에 맞춰 비디오 컨테이너에 unflip 클래스 토글
function applyFacingTransform() {
  if (facingMode === 'user') {
    videoContainer.classList.add('unflip');
  } else {
    videoContainer.classList.remove('unflip');
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  detectionList.innerHTML = '';
}

function drawDetections(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 거리 모드일 때: 사람 중심점을 미리 모아두고, 각 사물에서 가장 가까운 사람으로
  // 옅은 선을 그어 시각적으로 페어링을 보여준다
  const distMode = distanceMode.checked;
  const personCenters = distMode
    ? predictions
        .filter((p) => p.class === 'person')
        .map((p) => [p.bbox[0] + p.bbox[2] / 2, p.bbox[1] + p.bbox[3] / 2])
    : [];

  predictions.forEach((p) => {
    const [x, y, w, h] = p.bbox;
    const isPerson = p.class === 'person';
    const isCustom = !!p.isCustom;
    const hasSnd = hasSound(p.class);
    // 커스텀 클래스는 핑크 (서정월드 테마), COCO 매핑된 건 초록, 안 된 건 노랑
    const baseColor = isCustom ? '#ff2e7e' : hasSnd ? '#6acc6a' : '#ffcc4a';

    // 거리 볼륨에 따라 박스 두께/투명도 변조
    const vol = p.proximityVolume ?? 1;
    const lineWidth = distMode && !isPerson ? 1 + vol * 4 : 3;

    ctx.strokeStyle = baseColor;
    ctx.globalAlpha = distMode && !isPerson ? 0.3 + vol * 0.7 : 1;
    ctx.lineWidth = isCustom ? 4 : lineWidth;
    if (isCustom) {
      ctx.setLineDash([10, 6]);
    }
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    // 사람 - 사물 페어링 라인
    if (distMode && !isPerson && personCenters.length > 0) {
      const ocx = x + w / 2;
      const ocy = y + h / 2;
      // 가장 가까운 사람 찾기
      let near = personCenters[0];
      let minD = Infinity;
      for (const c of personCenters) {
        const d = (c[0] - ocx) ** 2 + (c[1] - ocy) ** 2;
        if (d < minD) {
          minD = d;
          near = c;
        }
      }
      ctx.beginPath();
      ctx.moveTo(near[0], near[1]);
      ctx.lineTo(ocx, ocy);
      ctx.strokeStyle = baseColor;
      ctx.globalAlpha = 0.2 + vol * 0.6;
      ctx.lineWidth = 1 + vol * 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.globalAlpha = 1;

    // 라벨 텍스트 (거리 모드면 vol% 도 표시)
    let text = `${isCustom ? '✨ ' : ''}${p.class} ${(p.score * 100).toFixed(0)}%`;
    if (distMode && !isPerson && !isCustom) {
      text += ` · vol ${(vol * 100).toFixed(0)}%`;
    }
    ctx.font = '600 14px sans-serif';
    const textW = ctx.measureText(text).width;
    ctx.fillStyle = baseColor;
    ctx.fillRect(x, Math.max(0, y - 20), textW + 10, 20);
    ctx.fillStyle = '#0a0a10';
    ctx.fillText(text, x + 5, Math.max(14, y - 6));
  });
}

function updateDetectionList(predictions) {
  detectionList.innerHTML = '';
  if (predictions.length === 0) {
    const li = document.createElement('li');
    li.textContent = '(감지된 사물 없음)';
    detectionList.appendChild(li);
    return;
  }
  const distMode = distanceMode.checked;
  predictions.forEach((p) => {
    const li = document.createElement('li');
    const has = hasSound(p.class);
    if (has) li.className = 'has-sound';
    let text = `${has ? '♪' : ' '} ${p.class}  (${(p.score * 100).toFixed(0)}%)`;
    if (distMode && p.class !== 'person') {
      text += ` · 🔊${((p.proximityVolume ?? 1) * 100).toFixed(0)}%`;
    }
    li.textContent = text;
    detectionList.appendChild(li);
  });
}

// FPS 계산용 (추론 / 페인트 분리)
let inferCount = 0;
let paintCount = 0;
let lastFpsTime = 0;

// 추론을 fire-and-forget 으로 트리거. await 하지 않아 메인 스레드 블로킹 없음.
function triggerInference() {
  if (inferenceInProgress) return;
  const now = performance.now();
  const minInterval = parseInt(intervalSlider.value, 10);
  if (now - lastInferenceStart < minInterval) return;
  if (video.readyState < 2) return;

  inferenceInProgress = true;
  lastInferenceStart = now;

  // tf.tidy 는 detect 내부에서 처리됨. 결과 Promise 를 받아 저장만 한다.
  model
    .detect(video, 10)
    .then(async (predictions) => {
      const conf = parseFloat(confSlider.value);
      const filtered = predictions.filter((p) => p.score >= conf);

      // KNN 사용자 사물 추론 (MobileNet 로드되어 있고 학습 데이터가 있을 때만)
      if (
        mobilenetModel &&
        knnModel &&
        knnModel.getNumClasses &&
        knnModel.getNumClasses() > 0
      ) {
        try {
          const features = mobilenetModel.infer(video, true);
          const result = await knnModel.predictClass(features);
          features.dispose();
          const topConf = result.confidences[result.label] || 0;
          if (topConf >= CUSTOM_CONF_THRESHOLD) {
            // 전체 프레임의 80% 를 덮는 가상 bbox (badge 용)
            const w = video.videoWidth || canvas.width || 640;
            const h = video.videoHeight || canvas.height || 480;
            filtered.push({
              class: result.label,
              score: topConf,
              bbox: [w * 0.1, h * 0.1, w * 0.8, h * 0.8],
              isCustom: true,
            });
          }
        } catch (e) {
          console.error('KNN predict failed', e);
        }
      }

      // 거리 모드: 사람 bbox 들을 모아서 각 사물에 proximityVolume 부여
      if (distanceMode.checked) {
        const personBoxes = filtered
          .filter((p) => p.class === 'person')
          .map((p) => p.bbox);
        const falloff = parseFloat(falloffSlider.value);
        for (const p of filtered) {
          if (p.class === 'person') {
            p.proximityVolume = 1.0;
          } else {
            p.proximityVolume = computeProximity(p.bbox, personBoxes, falloff);
          }
        }
      } else {
        for (const p of filtered) p.proximityVolume = 1.0;
      }

      latestPredictions = filtered;
      updateDetectionList(latestPredictions);
      playSounds(latestPredictions);
      inferCount++;
    })
    .catch((e) => console.error('detection error', e))
    .finally(() => {
      inferenceInProgress = false;
    });
}

// 메인 렌더 루프: 매 페인트마다 (1) 추론 트리거 시도 (2) 최신 박스 그리기
function renderLoop() {
  if (!running) return;

  triggerInference();
  drawDetections(latestPredictions);
  paintCount++;

  // 1초마다 FPS 갱신
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    const inferFps = (inferCount * 1000) / (now - lastFpsTime);
    const paintFps = (paintCount * 1000) / (now - lastFpsTime);
    statusEl.textContent = `실행 중 — 영상 ${paintFps.toFixed(0)}fps · 추론 ${inferFps.toFixed(1)}fps`;
    inferCount = 0;
    paintCount = 0;
    lastFpsTime = now;
  }

  requestAnimationFrame(renderLoop);
}

// ── CSS 기반 의사 전체화면 토글 ──
// iOS Safari 는 div 에 requestFullscreen 이 안 통하고 video 엘리먼트만 지원하는데
// 그러면 우리가 입힌 CRT 필터/오버레이가 사라진다. 그래서 native API 대신
// body 에 클래스를 토글해서 videoContainer 를 viewport 가득 채우는 방식으로 처리.
if (fullscreenBtn) {
  fullscreenBtn.addEventListener('click', () => {
    const isFs = document.body.classList.toggle('pseudo-fullscreen');
    fullscreenBtn.textContent = isFs ? '×' : '⛶';
    fullscreenBtn.title = isFs ? '전체화면 종료' : '전체화면';
    // 스크롤 잠금
    document.documentElement.style.overflow = isFs ? 'hidden' : '';
  });
  // ESC 로도 닫기
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('pseudo-fullscreen')) {
      fullscreenBtn.click();
    }
  });
}

startBtn.addEventListener('click', async () => {
  if (running) {
    running = false;
    startBtn.textContent = '▶ 시작';
    statusEl.textContent = '정지됨';
    overlayMessage.classList.remove('hidden');
    overlayMessage.textContent = '시작 버튼을 눌러 카메라를 켜세요';
    stopCamera();
    stopAllSounds();
    return;
  }

  startBtn.disabled = true;
  overlayMessage.classList.remove('hidden');

  try {
    // AudioContext는 사용자 제스처 안에서 생성해야 자동재생 정책에 걸리지 않음
    ensureAudioContext();
    // 모바일(iOS Safari 포함) speechSynthesis unlock: 사용자 제스처 안에서
    // 한 번 빈 utterance 를 실행해야 이후 speak() 호출이 동작한다
    unlockSpeechSynthesis();
    // meSpeak (진짜 병렬 TTS) 비동기 로드 — 끝나면 자동으로 fallback 대신 사용
    ensureMeSpeak();

    // IndexedDB 에서 불러왔지만 아직 디코딩 안 된 사운드가 있으면 지금 처리
    if (Object.keys(pendingSoundBuffers).length > 0) {
      overlayMessage.textContent = '저장된 사운드 디코딩 중...';
      await decodePendingSounds();
    }

    if (!model) {
      overlayMessage.textContent = '모델 로딩 중...';
      await loadModel();
    }

    overlayMessage.textContent = '카메라 권한 요청 중...';
    await startCamera();

    overlayMessage.classList.add('hidden');
    running = true;
    startBtn.textContent = '■ 정지';
    statusEl.textContent = '실행 중';
    latestPredictions = [];
    inferCount = 0;
    paintCount = 0;
    lastFpsTime = performance.now();
    lastInferenceStart = 0;
    inferenceInProgress = false;
    renderLoop();
  } catch (e) {
    console.error(e);
    statusEl.textContent = `오류: ${e.message}`;
    overlayMessage.textContent = `오류: ${e.message}`;
  } finally {
    startBtn.disabled = false;
  }
});
