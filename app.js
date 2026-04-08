// 사물 인식 사운드 플레이어 (브라우저 버전)
// - TensorFlow.js + COCO-SSD 로 카메라 영상에서 사물 감지
// - Web Audio API 로 사물별 고유 톤 재생
// - 다중 사물 동시 재생 모드 지원

// 사물 클래스 → 사운드 (오실레이터 주파수 + 파형)
// COCO-SSD 의 80개 클래스 중 자주 보이는 것들 매핑
const SOUND_MAP = {
  person:       { freq: 261.63, type: 'sine' },     // C4
  'cell phone': { freq: 329.63, type: 'square' },   // E4
  cup:          { freq: 392.00, type: 'sine' },     // G4
  book:         { freq: 440.00, type: 'triangle' }, // A4
  bottle:       { freq: 523.25, type: 'sine' },     // C5
  laptop:       { freq: 587.33, type: 'square' },   // D5
  keyboard:     { freq: 659.25, type: 'sine' },     // E5
  mouse:        { freq: 698.46, type: 'triangle' }, // F5
  scissors:     { freq: 783.99, type: 'sine' },     // G5
  remote:       { freq: 880.00, type: 'square' },   // A5
  tv:           { freq: 349.23, type: 'triangle' }, // F4
  chair:        { freq: 466.16, type: 'sine' },     // A#4
  backpack:     { freq: 311.13, type: 'square' },   // D#4
  handbag:      { freq: 369.99, type: 'triangle' }, // F#4
  tie:          { freq: 415.30, type: 'sine' },     // G#4
  banana:       { freq: 622.25, type: 'sine' },     // D#5
  apple:        { freq: 739.99, type: 'triangle' }, // F#5
  orange:       { freq: 830.61, type: 'sine' },     // G#5
};

const TONE_DURATION = 0.25; // seconds

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
const classSelect = document.getElementById('classSelect');
const soundFileInput = document.getElementById('soundFileInput');
const distanceMode = document.getElementById('distanceMode');
const falloffSlider = document.getElementById('falloffSlider');
const falloffValue = document.getElementById('falloffValue');
const videoContainer = document.getElementById('videoContainer');
const camSwitchBtn = document.getElementById('camSwitchBtn');

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

// 클래스 드롭다운 채우기
COCO_CLASSES.forEach((cls) => {
  const opt = document.createElement('option');
  opt.value = cls;
  opt.textContent = cls;
  classSelect.appendChild(opt);
});

// 사용자가 업로드한 사운드
// label -> {
//   kind: 'buffer',  buffer: AudioBuffer,  name, volume, rate
//   kind: 'element', blobUrl, duration, mimeType, name, volume, rate
// }
//   buffer: 순수 오디오 파일 (Web Audio API 경로, 동시 재생 최적)
//   element: 비디오 파일 / decodeAudioData 실패한 케이스 (HTMLMediaElement 경로)
const customSounds = {};

// 슬라이더 값 표시
confSlider.addEventListener('input', () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});
cdSlider.addEventListener('input', () => {
  cdValue.textContent = parseFloat(cdSlider.value).toFixed(1);
});
intervalSlider.addEventListener('input', () => {
  intervalValue.textContent = intervalSlider.value;
});
falloffSlider.addEventListener('input', () => {
  falloffValue.textContent = parseFloat(falloffSlider.value).toFixed(1);
});

// ---------- IndexedDB: 업로드 사운드 영구 저장 ----------
const DB_NAME = 'object-sound-db';
const STORE_NAME = 'sounds';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
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
const pendingSoundBuffers = {}; // label -> { arrayBuffer, name }

async function loadSavedSoundsFromDB() {
  try {
    const all = await dbGetAll();
    for (const [label, value] of Object.entries(all)) {
      pendingSoundBuffers[label] = value;
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
  const labels = Object.keys(pendingSoundBuffers);
  for (const label of labels) {
    const { arrayBuffer, name, mimeType, volume, rate } = pendingSoundBuffers[label];
    try {
      const entry = await createCustomSoundEntry(arrayBuffer, mimeType, name);
      entry.volume = volume ?? 1.0;
      entry.rate = rate ?? 1.0;
      customSounds[label] = entry;
      delete pendingSoundBuffers[label];
    } catch (e) {
      console.error(`'${label}' 디코딩 실패`, e);
    }
  }
  renderSoundList();
}

// 매핑된 사운드 목록 UI 표시 (기본 톤 + 업로드 사운드 + 편집/삭제)
function renderSoundList() {
  soundList.innerHTML = '';

  const allLabels = new Set([
    ...Object.keys(SOUND_MAP),
    ...Object.keys(customSounds),
    ...Object.keys(pendingSoundBuffers),
  ]);

  if (allLabels.size === 0) {
    const li = document.createElement('li');
    li.textContent = '(매핑된 사운드 없음)';
    soundList.appendChild(li);
    return;
  }

  [...allLabels].sort().forEach((label) => {
    const li = document.createElement('li');
    li.className = 'has-sound sound-item';

    const custom = customSounds[label] || pendingSoundBuffers[label];
    const isCustom = !!custom;

    // 첫 줄: 아이콘 + 라벨 + 액션 버튼들
    const row = document.createElement('div');
    row.className = 'sound-item-row';

    const span = document.createElement('span');
    const icon = isCustom ? '🔊' : '♪';
    span.textContent = isCustom
      ? `${icon} ${label} — ${custom.name}`
      : `${icon} ${label}`;
    row.appendChild(span);

    const actions = document.createElement('span');
    actions.className = 'sound-actions';

    // 미리듣기 (기본 톤도 미리듣기 가능)
    const testBtn = document.createElement('button');
    testBtn.textContent = '▶';
    testBtn.title = '미리듣기';
    testBtn.onclick = (e) => {
      e.stopPropagation();
      ensureAudioContext();
      if (Object.keys(pendingSoundBuffers).length > 0) decodePendingSounds();
      playOne(label);
    };
    actions.appendChild(testBtn);

    if (isCustom) {
      // 편집 토글 (볼륨/속도)
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
        // element 타입이면 blob URL 해제 (메모리 누수 방지)
        const entry = customSounds[label];
        if (entry && entry.kind === 'element' && entry.blobUrl) {
          URL.revokeObjectURL(entry.blobUrl);
        }
        delete customSounds[label];
        delete pendingSoundBuffers[label];
        await dbDelete(label);
        renderSoundList();
      };
      actions.appendChild(delBtn);
    }

    row.appendChild(actions);
    li.appendChild(row);

    // 편집 영역: 커스텀 사운드만
    if (isCustom) {
      const edit = document.createElement('div');
      edit.className = 'sound-edit';

      const volume = customSounds[label]?.volume ?? pendingSoundBuffers[label]?.volume ?? 1.0;
      const rate = customSounds[label]?.rate ?? pendingSoundBuffers[label]?.rate ?? 1.0;

      // 볼륨
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
        if (customSounds[label]) customSounds[label].volume = v;
        if (pendingSoundBuffers[label]) pendingSoundBuffers[label].volume = v;
        await persistSoundOptions(label);
      };
      volLabel.appendChild(volSlider);
      edit.appendChild(volLabel);

      // 재생 속도 (피치도 함께 변함)
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
        if (customSounds[label]) customSounds[label].rate = r;
        if (pendingSoundBuffers[label]) pendingSoundBuffers[label].rate = r;
        await persistSoundOptions(label);
      };
      rateLabel.appendChild(rateSlider);
      edit.appendChild(rateLabel);

      li.appendChild(edit);
    }

    soundList.appendChild(li);
  });
}

// 볼륨/속도만 IndexedDB 에 다시 저장 (ArrayBuffer 는 그대로 유지)
async function persistSoundOptions(label) {
  try {
    const all = await dbGetAll();
    const existing = all[label];
    if (!existing) return;
    const volume = customSounds[label]?.volume ?? pendingSoundBuffers[label]?.volume ?? 1.0;
    const rate = customSounds[label]?.rate ?? pendingSoundBuffers[label]?.rate ?? 1.0;
    await dbPut(label, { ...existing, volume, rate });
  } catch (e) {
    console.error('옵션 저장 실패', e);
  }
}

// 업로드 처리
// ArrayBuffer 를 HTMLMediaElement (Audio) 로 재생 가능한지 검증하고
// 재생용 blob URL + duration 을 반환. 실패하면 에러 throw.
// 비디오 파일이나 Web Audio API 가 디코드 못 하는 포맷에 대한 폴백 경로.
async function probeAsMediaElement(arrayBuffer, mimeType) {
  const blob = new Blob([arrayBuffer], { type: mimeType || 'audio/mpeg' });
  const url = URL.createObjectURL(blob);
  const probe = document.createElement('audio');
  probe.preload = 'metadata';
  probe.src = url;

  try {
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
      probe.onloadedmetadata = ok;
      probe.oncanplay = ok;
      probe.onerror = () => fail('이 파일은 브라우저가 재생할 수 없는 형식입니다');
      setTimeout(() => fail('파일 로딩 시간 초과'), 5000);
    });
    const duration = Number.isFinite(probe.duration) && probe.duration > 0
      ? probe.duration
      : 1.0;
    return { url, duration };
  } catch (e) {
    URL.revokeObjectURL(url);
    throw e;
  }
}

// 파일 하나를 customSounds 엔트리 형태로 변환.
// 1) 우선 Web Audio API 의 decodeAudioData 시도 (가장 빠르고 동시 재생 최적)
// 2) 실패하면 HTMLMediaElement 폴백 (비디오 파일, 일부 컨테이너 포맷 등)
// 3) 이미지 파일이면 명확한 에러
async function createCustomSoundEntry(arrayBuffer, mimeType, name) {
  if (mimeType && mimeType.startsWith('image/')) {
    throw new Error('이미지 파일은 사운드로 사용할 수 없습니다. 오디오 또는 비디오 파일을 선택해주세요.');
  }

  // 1) Web Audio API 디코딩 시도
  try {
    const buffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    return { kind: 'buffer', buffer, name, mimeType };
  } catch (decodeErr) {
    console.warn('decodeAudioData 실패, HTMLMediaElement 폴백 시도', decodeErr);
  }

  // 2) HTMLMediaElement 폴백 (비디오/컨테이너 포맷 등)
  try {
    const { url, duration } = await probeAsMediaElement(arrayBuffer, mimeType);
    return { kind: 'element', blobUrl: url, duration, name, mimeType };
  } catch (elemErr) {
    throw new Error(
      `디코딩 실패: ${elemErr.message}\n` +
      `파일 형식: ${mimeType || '알 수 없음'}\n` +
      `팁: mp3/wav/m4a/ogg 오디오 파일이나 mp4 비디오 파일을 사용해보세요.`
    );
  }
}

soundFileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const label = classSelect.value;
  if (!label) {
    alert('먼저 사물을 선택하세요.');
    soundFileInput.value = '';
    return;
  }

  statusEl.textContent = `'${file.name}' 디코딩 중...`;

  try {
    ensureAudioContext();
    const arrayBuffer = await file.arrayBuffer();

    // 같은 라벨에 이미 매핑이 있으면 볼륨/속도 유지, 없으면 기본값 1.0
    const prevVolume = customSounds[label]?.volume ?? 1.0;
    const prevRate = customSounds[label]?.rate ?? 1.0;

    // 이전 엔트리가 element 타입이면 blob URL 해제 (메모리 누수 방지)
    const prev = customSounds[label];
    if (prev && prev.kind === 'element' && prev.blobUrl) {
      URL.revokeObjectURL(prev.blobUrl);
    }

    const entry = await createCustomSoundEntry(arrayBuffer, file.type, file.name);
    entry.volume = prevVolume;
    entry.rate = prevRate;
    customSounds[label] = entry;

    // IndexedDB 에는 원본 arrayBuffer + mimeType 저장
    // (element 엔트리로 복원되었더라도 원본 파일을 그대로 저장해 다음 로드 시 재디코딩)
    await dbPut(label, {
      arrayBuffer,
      mimeType: file.type,
      name: file.name,
      volume: prevVolume,
      rate: prevRate,
    });
    statusEl.textContent = `'${label}' ← ${file.name} 매핑 완료 (${entry.kind})`;
    renderSoundList();
  } catch (err) {
    console.error(err);
    statusEl.textContent = '사운드 로드 실패';
    alert(err.message);
  } finally {
    soundFileInput.value = '';
  }
});

// 페이지 로드 시 저장된 사운드 불러오기
loadSavedSoundsFromDB();

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

// 사물에 사운드가 매핑되어 있는지 (커스텀 우선, 없으면 기본 톤)
function hasSound(label) {
  return customSounds[label] != null || SOUND_MAP[label] != null;
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

// 한 번의 톤 재생 (오실레이터 + 짧은 envelope). 재생 지속 시간(초)을 반환.
function playTone(label, multiplier = 1.0) {
  if (!audioContext || !SOUND_MAP[label]) return 0;
  const { freq, type } = SOUND_MAP[label];

  const osc = audioContext.createOscillator();
  const gain = audioContext.createGain();
  osc.type = type;
  osc.frequency.value = freq;

  const now = audioContext.currentTime;
  const peak = Math.max(0.0001, 0.18 * multiplier);
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(peak, now + 0.02);
  gain.gain.exponentialRampToValueAtTime(peak * 0.001, now + TONE_DURATION);

  osc.connect(gain).connect(audioContext.destination);
  osc.start(now);
  osc.stop(now + TONE_DURATION);
  return TONE_DURATION;
}

// 업로드된 사운드 재생. 엔트리 kind 에 따라 두 경로로 분기.
// 재생 지속 시간(초)을 반환. playbackRate 가 반영된 실제 길이.
function playCustom(label, multiplier = 1.0) {
  if (!audioContext || !customSounds[label]) return 0;
  const entry = customSounds[label];
  const volume = entry.volume ?? 1.0;
  const rate = entry.rate ?? 1.0;

  // 1) Web Audio API 경로 (buffer) — 동시 재생 최적
  if (entry.kind === 'buffer') {
    const source = audioContext.createBufferSource();
    source.buffer = entry.buffer;
    source.playbackRate.value = rate;

    const gain = audioContext.createGain();
    gain.gain.value = volume * multiplier;

    source.connect(gain).connect(audioContext.destination);

    activeSources.add(source);
    source.onended = () => {
      activeSources.delete(source);
      if ((playingUntil[label] ?? 0) <= performance.now() / 1000 + 0.05) {
        delete playingUntil[label];
      }
    };

    source.start();
    return entry.buffer.duration / rate;
  }

  // 2) HTMLMediaElement 경로 (element) — 비디오 파일 / 폴백 포맷
  // 동시 재생을 위해 매번 새 Audio 요소를 blob URL 로부터 생성한다.
  if (entry.kind === 'element') {
    const el = document.createElement('audio');
    el.src = entry.blobUrl;
    el.preload = 'auto';
    // HTMLMediaElement.volume 은 0~1 클램프 — multiplier 가 1 초과여도 1로 제한
    el.volume = Math.max(0, Math.min(1, volume * multiplier));
    el.playbackRate = rate;

    activeElements.add(el);
    const cleanup = () => {
      activeElements.delete(el);
      if ((playingUntil[label] ?? 0) <= performance.now() / 1000 + 0.05) {
        delete playingUntil[label];
      }
    };
    el.onended = cleanup;
    el.onerror = cleanup;

    el.play().catch((e) => {
      console.error('element 재생 실패', e);
      cleanup();
    });
    return (entry.duration || 1) / rate;
  }

  return 0;
}

// 라벨 1개 재생 + playingUntil 갱신. 이미 재생 중이면 아무 것도 하지 않음.
function playOne(label, multiplier = 1.0) {
  const nowSec = performance.now() / 1000;
  // 가드: 아직 재생 중이면 재트리거 금지 (긴 사운드 중복 재생 방지)
  if (nowSec < (playingUntil[label] ?? 0)) return;

  let duration = 0;
  if (customSounds[label]) {
    duration = playCustom(label, multiplier);
  } else if (SOUND_MAP[label]) {
    duration = playTone(label, multiplier);
  }
  if (duration > 0) {
    playingUntil[label] = nowSec + duration;
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

  // 같은 클래스가 여러 개 감지되면 가장 가까운(=볼륨 큰) 것만 사용
  const byClass = new Map();
  for (const p of predictions) {
    const existing = byClass.get(p.class);
    if (!existing || (p.proximityVolume ?? 1) > (existing.proximityVolume ?? 1)) {
      byClass.set(p.class, p);
    }
  }

  const candidates = [];
  for (const [label, pred] of byClass) {
    if (!hasSound(label)) continue;
    // 아직 이전 재생이 끝나지 않았으면 스킵 (긴 사운드 중복 방지)
    if (now < (playingUntil[label] || 0)) continue;
    // 쿨다운은 마지막 재생 시작 시점 기준 (짧은 톤용)
    if (now - (lastPlayed[label] || 0) < cd) continue;
    candidates.push({ label, multiplier: pred.proximityVolume ?? 1 });
  }

  if (multiModeEl.checked) {
    // 다중 모드: 모두 재생 (각자 자기 multiplier 로)
    for (const { label, multiplier } of candidates) {
      playOne(label, multiplier);
      lastPlayed[label] = now;
    }
  } else {
    // 단일 모드: 직전 사운드 끝났을 때만, 그리고 가장 가까운 사물 1개
    if (now - (lastPlayed.__last__ || 0) < TONE_DURATION) return;
    candidates.sort((a, b) => b.multiplier - a.multiplier);
    if (candidates.length > 0) {
      const { label, multiplier } = candidates[0];
      playOne(label, multiplier);
      lastPlayed[label] = now;
      lastPlayed.__last__ = now;
    }
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
    const baseColor = hasSound(p.class) ? '#6acc6a' : '#ffcc4a';

    // 거리 볼륨에 따라 박스 두께/투명도 변조
    const vol = p.proximityVolume ?? 1;
    const lineWidth = distMode && !isPerson ? 1 + vol * 4 : 3;

    ctx.strokeStyle = baseColor;
    ctx.globalAlpha = distMode && !isPerson ? 0.3 + vol * 0.7 : 1;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, w, h);

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
    let text = `${p.class} ${(p.score * 100).toFixed(0)}%`;
    if (distMode && !isPerson) {
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
    .then((predictions) => {
      const conf = parseFloat(confSlider.value);
      const filtered = predictions.filter((p) => p.score >= conf);

      // 거리 모드: 사람 bbox 들을 모아서 각 사물에 proximityVolume 부여
      if (distanceMode.checked) {
        const personBoxes = filtered
          .filter((p) => p.class === 'person')
          .map((p) => p.bbox);
        const falloff = parseFloat(falloffSlider.value);
        for (const p of filtered) {
          if (p.class === 'person') {
            p.proximityVolume = 1.0; // 사람 자신은 항상 풀볼륨
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
