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
const detectionList = document.getElementById('detectionList');
const soundList = document.getElementById('soundList');

// 슬라이더 값 표시
confSlider.addEventListener('input', () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});
cdSlider.addEventListener('input', () => {
  cdValue.textContent = parseFloat(cdSlider.value).toFixed(1);
});

// 매핑된 사운드 목록 표시
function renderSoundList() {
  soundList.innerHTML = '';
  Object.keys(SOUND_MAP).forEach((label) => {
    const li = document.createElement('li');
    li.className = 'has-sound';
    li.textContent = `♪ ${label}`;
    soundList.appendChild(li);
  });
}
renderSoundList();

// 한 번의 톤 재생 (오실레이터 + 짧은 envelope)
function playTone(label) {
  if (!audioContext || !SOUND_MAP[label]) return;
  const { freq, type } = SOUND_MAP[label];

  const osc = audioContext.createOscillator();
  const gain = audioContext.createGain();
  osc.type = type;
  osc.frequency.value = freq;

  const now = audioContext.currentTime;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.18, now + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.001, now + TONE_DURATION);

  osc.connect(gain).connect(audioContext.destination);
  osc.start(now);
  osc.stop(now + TONE_DURATION);
}

// 감지된 라벨에 대해 모드/쿨다운에 따라 사운드 재생
function playSounds(labels) {
  const now = performance.now() / 1000;
  const cd = parseFloat(cdSlider.value);
  const unique = [...new Set(labels)];

  if (multiModeEl.checked) {
    // 다중 모드: 감지된 모든 매핑 사물의 사운드를 병렬 재생
    unique.forEach((label) => {
      if (!SOUND_MAP[label]) return;
      if (now - (lastPlayed[label] || 0) < cd) return;
      playTone(label);
      lastPlayed[label] = now;
    });
  } else {
    // 단일 모드: 직전 사운드가 끝났을 때만 1개 재생
    if (now - (lastPlayed.__last__ || 0) < TONE_DURATION) return;
    for (const label of unique) {
      if (!SOUND_MAP[label]) continue;
      if (now - (lastPlayed[label] || 0) < cd) continue;
      playTone(label);
      lastPlayed[label] = now;
      lastPlayed.__last__ = now;
      break;
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
  // 해상도를 낮추면 추론 속도가 크게 빨라짐 (입력 텐서가 작아짐)
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width: { ideal: 480 },
      height: { ideal: 360 },
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
  predictions.forEach((p) => {
    const [x, y, w, h] = p.bbox;
    const hasSound = SOUND_MAP[p.class] != null;
    const color = hasSound ? '#6acc6a' : '#ffcc4a';

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    const text = `${p.class} ${(p.score * 100).toFixed(0)}%`;
    ctx.font = '600 16px sans-serif';
    const textW = ctx.measureText(text).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 22), textW + 10, 22);
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
  predictions.forEach((p) => {
    const li = document.createElement('li');
    const hasSound = SOUND_MAP[p.class] != null;
    if (hasSound) li.className = 'has-sound';
    li.textContent = `${hasSound ? '♪' : ' '} ${p.class}  (${(p.score * 100).toFixed(0)}%)`;
    detectionList.appendChild(li);
  });
}

// 추론 FPS 계산용
let frameCount = 0;
let lastFpsTime = 0;

async function detectionLoop() {
  if (!running) return;

  try {
    // model.detect 의 두 번째 인자는 maxNumBoxes (기본 20). 줄이면 후처리가 빨라짐
    const predictions = await model.detect(video, 10);
    const conf = parseFloat(confSlider.value);
    const filtered = predictions.filter((p) => p.score >= conf);

    drawDetections(filtered);
    updateDetectionList(filtered);
    playSounds(filtered.map((p) => p.class));

    // FPS 표시 (1초마다 갱신)
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      const fps = (frameCount * 1000) / (now - lastFpsTime);
      statusEl.textContent = `실행 중 — ${fps.toFixed(1)} FPS`;
      frameCount = 0;
      lastFpsTime = now;
    }
  } catch (e) {
    console.error('detection error', e);
  }

  // requestAnimationFrame: 브라우저가 페인트와 동기화해 부드럽게 처리
  requestAnimationFrame(detectionLoop);
}

startBtn.addEventListener('click', async () => {
  if (running) {
    running = false;
    startBtn.textContent = '▶ 시작';
    statusEl.textContent = '정지됨';
    overlayMessage.classList.remove('hidden');
    overlayMessage.textContent = '시작 버튼을 눌러 카메라를 켜세요';
    stopCamera();
    return;
  }

  startBtn.disabled = true;
  overlayMessage.classList.remove('hidden');

  try {
    // AudioContext는 사용자 제스처 안에서 생성해야 자동재생 정책에 걸리지 않음
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
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
    frameCount = 0;
    lastFpsTime = performance.now();
    detectionLoop();
  } catch (e) {
    console.error(e);
    statusEl.textContent = `오류: ${e.message}`;
    overlayMessage.textContent = `오류: ${e.message}`;
  } finally {
    startBtn.disabled = false;
  }
});
