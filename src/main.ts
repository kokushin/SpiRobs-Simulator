import './styles.css';
import * as THREE from 'three';
import * as CANNON from 'cannon-es';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { autoGraspCommand, DEFAULT_SETTINGS, SimSettings, SpiRobPhysics, UNIT_DATA, Mount, ObjectKind } from './physics';

const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark"><i></i><i></i><i></i></div>
        <div><strong>SpiRob</strong><span>DYNAMICS LAB</span></div>
      </div>
      <div class="model-meta"><span class="status-dot"></span><b>3-CABLE / 20 UNIT</b><span>STL-RESOLVED</span></div>
      <nav class="toolbar" aria-label="シミュレーション操作">
        <button id="pause-btn" class="tool-button" title="一時停止"><span class="pause-icon">Ⅱ</span><span>一時停止</span></button>
        <button id="reset-btn" class="tool-button" title="リセット">↻<span>リセット</span></button>
        <button id="capture-btn" class="icon-button" title="画像を保存">⌁</button>
        <button id="help-btn" class="icon-button" title="モデルについて">?</button>
      </nav>
    </header>

    <aside class="panel left-panel">
      <div class="panel-heading"><span>ACTUATION</span><small>ケーブル張力</small></div>
      <div class="cable-control cable-a">
        <div class="control-label"><span><i></i>CABLE A <small>0°</small></span><output id="force-a-out">0.0 N</output></div>
        <input id="force-a" type="range" min="0" max="30" step="0.1" value="0" aria-label="Cable A tension" />
      </div>
      <div class="cable-control cable-b">
        <div class="control-label"><span><i></i>CABLE B <small>120°</small></span><output id="force-b-out">0.0 N</output></div>
        <input id="force-b" type="range" min="0" max="30" step="0.1" value="0" aria-label="Cable B tension" />
      </div>
      <div class="cable-control cable-c">
        <div class="control-label"><span><i></i>CABLE C <small>240°</small></span><output id="force-c-out">0.0 N</output></div>
        <input id="force-c" type="range" min="0" max="30" step="0.1" value="0" aria-label="Cable C tension" />
      </div>

      <div class="section-rule"></div>
      <div class="panel-heading compact"><span>MOUNT</span><small>固定方向</small></div>
      <div class="segmented" id="mount-control">
        <button data-mount="horizontal" class="active">水平</button>
        <button data-mount="hanging">吊下</button>
        <button data-mount="upright">垂直</button>
      </div>

      <div class="section-rule"></div>
      <div class="panel-heading compact"><span>TEST OBJECT</span><small>把持対象</small></div>
      <label class="field-row"><span>形状</span><select id="object-kind">
        <option value="sphere">球</option><option value="cylinder">円柱</option><option value="box">立方体</option><option value="none">なし</option>
      </select></label>
      <label class="field-row"><span>直径 / 幅</span><div><input id="object-size" type="number" min="10" max="160" step="1" value="52"><em>mm</em></div></label>
      <label class="field-row"><span>質量</span><div><input id="object-mass" type="number" min="0" max="10" step="0.005" value="0.045"><em>kg</em></div></label>
      <p class="drag-hint">オブジェクトはビュー上でドラッグできます</p>
      <button id="run-grasp-btn" class="run-grasp-button" type="button" aria-pressed="false">
        <i>◎</i><span><b>把持シミュレーション実行</b><small>PACK → REACH → WRAP → GRASP</small></span>
      </button>

      <div class="section-rule"></div>
      <details>
        <summary><span>CALIBRATION</span><small>実機同定パラメータ</small></summary>
        <div class="details-body">
          <label class="field-stack"><span>基部曲げ剛性 <output id="stiffness-out">0.090</output> N·m/rad</span><input id="stiffness" type="range" min="0.005" max="0.3" step="0.0025" value="0.09"></label>
          <label class="field-stack"><span>関節減衰 <output id="damping-out">0.0008</output> N·m·s/rad</span><input id="damping" type="range" min="0.0001" max="0.006" step="0.0001" value="0.0008"></label>
          <label class="field-stack"><span>ケーブル摩擦 <output id="cable-friction-out">0.08</output></span><input id="cable-friction" type="range" min="0" max="0.5" step="0.01" value="0.08"></label>
          <label class="field-stack"><span>TPU接触摩擦 <output id="body-friction-out">0.72</output></span><input id="body-friction" type="range" min="0.1" max="1.4" step="0.01" value="0.72"></label>
          <label class="field-row"><span>総質量</span><div><input id="total-mass" type="number" min="0.005" max="2" step="0.001" value="0.0384"><em>kg</em></div></label>
          <div class="io-buttons"><button id="export-btn">設定を書き出す</button><button id="import-btn">読み込む</button><input id="import-file" type="file" accept="application/json" hidden></div>
        </div>
      </details>
    </aside>

    <main class="viewport-wrap">
      <div id="viewport" aria-label="3D SpiRobシミュレーション"></div>
      <div class="view-badge"><span>LIVE PHYSICS</span><b id="solver-state">120 Hz</b></div>
      <div class="axis-widget"><span class="axis-y">Y</span><span class="axis-z">Z</span><span class="axis-x">X</span></div>
      <div class="camera-help">左ドラッグ: 回転 · 右ドラッグ: 移動 · ホイール: ズーム</div>
      <div id="grasp-hud" class="grasp-hud" aria-live="polite">
        <div class="grasp-hud-head"><span>AUTO GRASP</span><b id="grasp-stage-hud">PACKING</b></div>
        <div class="grasp-steps">
          <i data-grasp-step="packing"><span>01</span>PACK</i>
          <i data-grasp-step="reaching"><span>02</span>REACH</i>
          <i data-grasp-step="wrapping"><span>03</span>WRAP</i>
          <i data-grasp-step="grasping"><span>04</span>GRASP</i>
        </div>
      </div>
      <div id="loading" class="loading"><i></i><span>STLメッシュを解析中</span></div>
    </main>

    <aside class="panel right-panel">
      <div class="panel-heading"><span>TELEMETRY</span><small>実時間解析</small></div>
      <div class="telemetry-primary">
        <div><span>TIP SPEED</span><strong id="tip-speed">0.000</strong><small>m/s</small></div>
        <canvas id="velocity-chart" width="240" height="58"></canvas>
      </div>
      <div class="metric-grid">
        <div><span>TIP X</span><b id="tip-x">0.0</b><small>mm</small></div>
        <div><span>TIP Y</span><b id="tip-y">0.0</b><small>mm</small></div>
        <div><span>TIP Z</span><b id="tip-z">0.0</b><small>mm</small></div>
        <div><span>CONTACTS</span><b id="contacts">0</b><small>pairs</small></div>
      </div>
      <div id="grasp-monitor" class="grasp-monitor">
        <div class="grasp-monitor-head"><span>GRASP STATE</span><b id="grasp-state">IDLE</b></div>
        <div class="quality-row"><span>GRASP QUALITY</span><output id="grasp-quality">0%</output></div>
        <div class="quality-track"><i id="quality-fill"></i></div>
        <div class="grasp-data"><span>NORMAL FORCE <b id="grasp-force">0.0 N</b></span><span>UNITS <b id="contact-units">—</b></span></div>
        <div class="object-data"><span>OBJECT XYZ</span><b id="object-position">—</b></div>
      </div>
      <div class="section-rule"></div>
      <div class="panel-heading compact"><span>MODEL LAYERS</span><small>表示</small></div>
      <label class="toggle-row"><span><i class="layer-swatch mesh"></i>STLサーフェス</span><input id="show-mesh" type="checkbox" checked><i></i></label>
      <label class="toggle-row"><span><i class="layer-swatch cables"></i>UHMWPEケーブル</span><input id="show-cables" type="checkbox" checked><i></i></label>
      <label class="toggle-row"><span><i class="layer-swatch bodies"></i>衝突ボディ</span><input id="show-bodies" type="checkbox"><i></i></label>
      <label class="toggle-row"><span><i class="layer-swatch grid"></i>基準グリッド</span><input id="show-grid" type="checkbox" checked><i></i></label>
      <div class="section-rule"></div>
      <div class="panel-heading compact"><span>SOLVER</span><small>数値計算</small></div>
      <label class="field-stack"><span>時間倍率 <output id="timescale-out">1.00×</output></span><input id="timescale" type="range" min="0.1" max="2" step="0.05" value="1"></label>
      <label class="field-stack"><span>重力 <output id="gravity-out">9.81 m/s²</output></span><input id="gravity" type="range" min="0" max="19.62" step="0.1" value="9.81"></label>
      <div class="solver-note"><b>DISCRETE ELASTIC ROD</b><p>20剛体ユニット / 19弾性関節<br>120 Hz · 28 solver iterations</p></div>
    </aside>

    <footer class="sequence-bar">
      <div class="sequence-label"><span>MOTION PRESETS</span><small>論文ベースの入力シーケンス</small></div>
      <button data-preset="relax"><i>○</i><span>RELAX<small>無張力</small></span></button>
      <button data-preset="pack-a"><i class="a">A</i><span>PACK A<small>単一ケーブル</small></span></button>
      <button data-preset="balanced"><i>≋</i><span>BALANCED<small>等張力</small></span></button>
      <button data-preset="steer"><i class="bc">B+C</i><span>STEER<small>方向制御</small></span></button>
      <button data-preset="whip" class="accent"><i>⌁</i><span>WHIP<small>鋸歯状入力</small></span></button>
      <button data-preset="grasp" class="grasp-button"><i>◎</i><span>AUTO GRASP<small>把持シーケンス</small></span></button>
      <div class="timecode"><span>SIM TIME</span><b id="timecode">00:00.000</b></div>
    </footer>
  </div>

  <dialog id="about-dialog">
    <button class="dialog-close" aria-label="閉じる">×</button>
    <span class="eyebrow">MODEL BASIS</span>
    <h2>現実との差を隠さない<br>SpiRobシミュレータ</h2>
    <p>添付STLから20個の剛体ユニットと弾性軸を抽出し、各ユニットの実寸・質量分布・接触形状を物理モデルへ反映しています。3本のケーブル張力は断面上120°間隔のモーメントとして作用します。</p>
    <p>論文で未公開のTPU印刷物の実効弾性率、層方向、ケーブル摩擦などは校正値です。実機の先端軌道またはモータ電流を計測し、CALIBRATIONで同定すると再現性が上がります。</p>
    <div class="equation">ρ = ae<sup>bθ</sup><span>logarithmic spiral</span></div>
    <a href="https://www.cell.com/device/fulltext/S2666-9986(24)00603-3" target="_blank" rel="noreferrer">論文を開く ↗</a>
  </dialog>
`;

const settings: SimSettings = structuredClone(DEFAULT_SETTINGS);
let physics = new SpiRobPhysics(settings);
let paused = false;
let automation: 'whip' | 'grasp' | null = null;
let automationStart = 0;
type GraspStage = 'idle' | 'packing' | 'reaching' | 'wrapping' | 'grasping' | 'holding';
let graspStage: GraspStage = 'idle';
let graspPrimaryCable = 0;
let graspTargetStaged = false;

const viewport = document.querySelector<HTMLDivElement>('#viewport')!;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#081210');
scene.fog = new THREE.FogExp2('#081210', 1.25);
const camera = new THREE.PerspectiveCamera(39, 1, 0.005, 10);
camera.position.set(0.44, 0.32, 0.53);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.08, 0.05, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.12;
controls.maxDistance = 2;

scene.add(new THREE.HemisphereLight('#b9e8d7', '#101614', 1.65));
const keyLight = new THREE.DirectionalLight('#d8fff1', 5.5);
keyLight.position.set(-0.2, 0.75, 0.55);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.left = -0.5;
keyLight.shadow.camera.right = 0.5;
keyLight.shadow.camera.top = 0.5;
keyLight.shadow.camera.bottom = -0.5;
scene.add(keyLight);
const rimLight = new THREE.PointLight('#20d6a6', 6, 1.2);
rimLight.position.set(0.35, 0.12, -0.35);
scene.add(rimLight);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  new THREE.MeshStandardMaterial({ color: '#091311', roughness: 0.95, metalness: 0.02 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.121;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(1.5, 30, '#21423a', '#142822');
grid.position.y = -0.119;
grid.material.opacity = 0.46;
grid.material.transparent = true;
scene.add(grid);

const clampGroup = new THREE.Group();
const clampMaterial = new THREE.MeshStandardMaterial({ color: '#1a2925', roughness: 0.34, metalness: 0.68 });
const clamp = new THREE.Mesh(new THREE.BoxGeometry(0.048, 0.06, 0.066), clampMaterial);
clamp.castShadow = true;
clampGroup.add(clamp);
scene.add(clampGroup);

const debugGroup = new THREE.Group();
const debugMeshes = UNIT_DATA.map(([, length, h, w]) => {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(length * 0.001 * 0.94, h * 0.001 * 0.86, w * 0.001 * 0.86),
    new THREE.MeshBasicMaterial({ color: '#43ffd1', wireframe: true, transparent: true, opacity: 0.27 }),
  );
  debugGroup.add(mesh);
  return mesh;
});
debugGroup.visible = false;
scene.add(debugGroup);

const cableColors = ['#ff6b61', '#54e6a5', '#78a8ff'];
const cableLines = cableColors.map((color) => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(UNIT_DATA.length * 3), 3));
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  line.frustumCulled = false;
  scene.add(line);
  return line;
});

let stlMesh: THREE.Mesh | undefined;
const boneMatrices = UNIT_DATA.map(() => new THREE.Matrix4());
const restY = UNIT_DATA.map((_, i) => -77.947 - (1.8477 * i) / 19);
const restZ = 963.64163;
const flip = new THREE.Matrix4().makeRotationY(Math.PI);
const scaleMM = new THREE.Matrix4().makeScale(0.001, 0.001, 0.001);

new STLLoader().load('/models/3-cable-spirob.stl', (geometry) => {
  geometry.computeVertexNormals();
  const position = geometry.getAttribute('position');
  const bone0 = new Float32Array(position.count);
  const bone1 = new Float32Array(position.count);
  const weight = new Float32Array(position.count);
  const centers = UNIT_DATA.map((u) => u[0]);
  for (let v = 0; v < position.count; v++) {
    const x = position.getX(v);
    let i = 0;
    while (i < centers.length - 1 && x < centers[i + 1]) i++;
    if (i === centers.length - 1 || x >= centers[0]) {
      bone0[v] = i; bone1[v] = i; weight[v] = 0;
    } else {
      bone0[v] = i; bone1[v] = i + 1;
      weight[v] = THREE.MathUtils.clamp((centers[i] - x) / (centers[i] - centers[i + 1]), 0, 1);
    }
  }
  geometry.setAttribute('aBone0', new THREE.BufferAttribute(bone0, 1));
  geometry.setAttribute('aBone1', new THREE.BufferAttribute(bone1, 1));
  geometry.setAttribute('aWeight', new THREE.BufferAttribute(weight, 1));

  const material = new THREE.MeshPhysicalMaterial({
    color: '#beddd2', roughness: 0.48, metalness: 0.02, clearcoat: 0.18, clearcoatRoughness: 0.66,
  });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uBones = { value: boneMatrices };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\nuniform mat4 uBones[20];\nattribute float aBone0;\nattribute float aBone1;\nattribute float aWeight;`)
      .replace('#include <beginnormal_vertex>', `mat4 skinMatrix = uBones[int(aBone0)] * (1.0 - aWeight) + uBones[int(aBone1)] * aWeight;\nvec3 objectNormal = normalize(mat3(skinMatrix) * normal);`)
      .replace('#include <begin_vertex>', `vec3 transformed = (skinMatrix * vec4(position, 1.0)).xyz;`);
  };
  material.customProgramCacheKey = () => 'spirob-skin-v2';
  stlMesh = new THREE.Mesh(geometry, material);
  stlMesh.castShadow = true;
  stlMesh.receiveShadow = true;
  stlMesh.frustumCulled = false;
  scene.add(stlMesh);
  document.querySelector('#loading')?.classList.add('done');
}, undefined, () => {
  const loading = document.querySelector('#loading')!;
  loading.innerHTML = '<span>STLを読み込めませんでした</span>';
});

let objectMesh: THREE.Mesh | undefined;
const objectMaterial = new THREE.MeshPhysicalMaterial({ color: '#d8a64b', roughness: 0.32, metalness: 0.08, clearcoat: 0.5 });
function rebuildObjectMesh() {
  if (objectMesh) scene.remove(objectMesh);
  objectMesh = undefined;
  const r = settings.objectSize * 0.0005;
  let geometry: THREE.BufferGeometry | undefined;
  if (settings.objectKind === 'sphere') geometry = new THREE.SphereGeometry(r, 36, 24);
  if (settings.objectKind === 'box') geometry = new THREE.BoxGeometry(r * 2, r * 2, r * 2);
  if (settings.objectKind === 'cylinder') geometry = new THREE.CylinderGeometry(r, r, r * 2, 36);
  if (geometry) {
    objectMesh = new THREE.Mesh(geometry, objectMaterial);
    objectMesh.castShadow = true;
    objectMesh.receiveShadow = true;
    scene.add(objectMesh);
  }
}
rebuildObjectMesh();

function updateBoneMatrices() {
  const tq = new THREE.Quaternion();
  const tp = new THREE.Vector3();
  const rot = new THREE.Matrix4();
  const tr = new THREE.Matrix4();
  const invRest = new THREE.Matrix4();
  physics.bodies.forEach((body, i) => {
    tp.set(body.position.x, body.position.y, body.position.z);
    tq.set(body.quaternion.x, body.quaternion.y, body.quaternion.z, body.quaternion.w);
    tr.makeTranslation(tp.x, tp.y, tp.z);
    rot.makeRotationFromQuaternion(tq);
    invRest.makeTranslation(-UNIT_DATA[i][0], -restY[i], -restZ);
    boneMatrices[i].copy(tr).multiply(rot).multiply(flip).multiply(scaleMM).multiply(invRest);
    debugMeshes[i].position.copy(tp);
    debugMeshes[i].quaternion.copy(tq);
  });
  if (stlMesh) (stlMesh.material as THREE.Material).needsUpdate = false;
}

const cableOffsets = [0, (Math.PI * 2) / 3, (Math.PI * 4) / 3];
function updateCables() {
  cableLines.forEach((line, cable) => {
    const positions = line.geometry.getAttribute('position') as THREE.BufferAttribute;
    physics.bodies.forEach((body, i) => {
      const radius = ((UNIT_DATA[i][2] + UNIT_DATA[i][3]) * 0.25) * 0.001 * 0.76;
      const local = new CANNON.Vec3(0, radius * Math.cos(cableOffsets[cable]), radius * Math.sin(cableOffsets[cable]));
      const offset = body.quaternion.vmult(local);
      positions.setXYZ(i, body.position.x + offset.x, body.position.y + offset.y, body.position.z + offset.z);
    });
    positions.needsUpdate = true;
  });
}

function updateSceneObjects() {
  updateBoneMatrices();
  updateCables();
  if (objectMesh && physics.objectBody) {
    objectMesh.position.set(physics.objectBody.position.x, physics.objectBody.position.y, physics.objectBody.position.z);
    objectMesh.quaternion.set(physics.objectBody.quaternion.x, physics.objectBody.quaternion.y, physics.objectBody.quaternion.z, physics.objectBody.quaternion.w);
    objectMaterial.emissive.set(physics.isGrasping ? '#236b53' : '#000000');
    objectMaterial.emissiveIntensity = physics.isGrasping ? 0.65 : physics.graspQuality * 0.24;
  }
  const base = physics.bodies[0];
  if (base) {
    clampGroup.position.set(base.position.x, base.position.y, base.position.z);
    clampGroup.quaternion.set(base.quaternion.x, base.quaternion.y, base.quaternion.z, base.quaternion.w);
    clamp.position.x = -0.026;
  }
}

const forceInputs = ['a', 'b', 'c'].map((id) => document.querySelector<HTMLInputElement>(`#force-${id}`)!);
function syncForceUI() {
  forceInputs.forEach((input, i) => {
    input.value = settings.cableForces[i].toFixed(1);
    document.querySelector<HTMLOutputElement>(`#force-${['a', 'b', 'c'][i]}-out`)!.value = `${settings.cableForces[i].toFixed(1)} N`;
  });
}
forceInputs.forEach((input, i) => input.addEventListener('input', () => {
  releaseGraspTarget();
  physics.releaseGrasp();
  physics.wrapProgress = 0;
  automation = null;
  setGraspStage('idle');
  settings.cableForces[i] = Number(input.value);
  syncForceUI();
}));

function bindRange(id: string, key: keyof SimSettings, formatter: (v: number) => string) {
  const input = document.querySelector<HTMLInputElement>(`#${id}`)!;
  const output = document.querySelector<HTMLOutputElement>(`#${id}-out`)!;
  input.addEventListener('input', () => {
    (settings[key] as number) = Number(input.value);
    output.value = formatter(Number(input.value));
  });
}
bindRange('stiffness', 'stiffness', (v) => v.toFixed(3));
bindRange('damping', 'damping', (v) => v.toFixed(4));
bindRange('cable-friction', 'cableFriction', (v) => v.toFixed(2));
bindRange('body-friction', 'bodyFriction', (v) => v.toFixed(2));
bindRange('timescale', 'timeScale', (v) => `${v.toFixed(2)}×`);
bindRange('gravity', 'gravity', (v) => `${v.toFixed(2)} m/s²`);

document.querySelectorAll<HTMLButtonElement>('[data-mount]').forEach((button) => button.addEventListener('click', () => {
  automation = null;
  graspTargetStaged = false;
  setGraspStage('idle');
  settings.mount = button.dataset.mount as Mount;
  document.querySelectorAll('[data-mount]').forEach((b) => b.classList.toggle('active', b === button));
  physics.reset();
}));

const objectKindInput = document.querySelector<HTMLSelectElement>('#object-kind')!;
const objectSizeInput = document.querySelector<HTMLInputElement>('#object-size')!;
const objectMassInput = document.querySelector<HTMLInputElement>('#object-mass')!;
function updateObject() {
  automation = null;
  graspTargetStaged = false;
  physics.releaseGrasp();
  setGraspStage('idle');
  settings.objectKind = objectKindInput.value as ObjectKind;
  settings.objectSize = Number(objectSizeInput.value);
  settings.objectMass = Number(objectMassInput.value);
  physics.rebuildObject();
  rebuildObjectMesh();
  document.querySelector<HTMLButtonElement>('#run-grasp-btn')!.disabled = settings.objectKind === 'none';
}
[objectKindInput, objectSizeInput, objectMassInput].forEach((el) => el.addEventListener('change', updateObject));
document.querySelector<HTMLInputElement>('#total-mass')!.addEventListener('change', (event) => {
  settings.totalMass = Number((event.target as HTMLInputElement).value);
  physics.reset();
});

document.querySelector<HTMLInputElement>('#show-mesh')!.addEventListener('change', (e) => { if (stlMesh) stlMesh.visible = (e.target as HTMLInputElement).checked; });
document.querySelector<HTMLInputElement>('#show-cables')!.addEventListener('change', (e) => cableLines.forEach((l) => l.visible = (e.target as HTMLInputElement).checked));
document.querySelector<HTMLInputElement>('#show-bodies')!.addEventListener('change', (e) => debugGroup.visible = (e.target as HTMLInputElement).checked);
document.querySelector<HTMLInputElement>('#show-grid')!.addEventListener('change', (e) => grid.visible = (e.target as HTMLInputElement).checked);

function resetSimulation() {
  graspTargetStaged = false;
  automation = null;
  setGraspStage('idle');
  settings.cableForces = [0, 0, 0];
  syncForceUI();
  physics.reset();
}
document.querySelector('#reset-btn')!.addEventListener('click', resetSimulation);
document.querySelector('#pause-btn')!.addEventListener('click', () => {
  paused = !paused;
  const btn = document.querySelector<HTMLButtonElement>('#pause-btn')!;
  btn.classList.toggle('active', paused);
  btn.querySelector('.pause-icon')!.textContent = paused ? '▶' : 'Ⅱ';
  btn.querySelector('span:last-child')!.textContent = paused ? '再開' : '一時停止';
});

document.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach((button) => button.addEventListener('click', () => {
  releaseGraspTarget();
  physics.releaseGrasp();
  physics.wrapProgress = 0;
  document.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('selected', b === button));
  automation = null;
  setGraspStage('idle');
  const preset = button.dataset.preset;
  if (preset === 'relax') settings.cableForces = [0, 0, 0];
  if (preset === 'pack-a') settings.cableForces = [18, 0, 0];
  if (preset === 'balanced') settings.cableForces = [9, 9, 9];
  if (preset === 'steer') settings.cableForces = [0, 14, 5];
  if (preset === 'whip') { automation = 'whip'; automationStart = performance.now(); }
  if (preset === 'grasp') runFullGraspSimulation();
  syncForceUI();
}));

document.querySelector('#capture-btn')!.addEventListener('click', () => {
  renderer.render(scene, camera);
  const link = document.createElement('a');
  link.download = `spirob-simulation-${Date.now()}.png`;
  link.href = renderer.domElement.toDataURL('image/png');
  link.click();
});

const dialog = document.querySelector<HTMLDialogElement>('#about-dialog')!;
document.querySelector('#help-btn')!.addEventListener('click', () => dialog.showModal());
dialog.querySelector('.dialog-close')!.addEventListener('click', () => dialog.close());

document.querySelector('#export-btn')!.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({ schema: 'spirob-calibration-v1', settings }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = 'spirob-calibration.json'; link.click();
  URL.revokeObjectURL(url);
});
document.querySelector('#import-btn')!.addEventListener('click', () => document.querySelector<HTMLInputElement>('#import-file')!.click());
document.querySelector<HTMLInputElement>('#import-file')!.addEventListener('change', async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0]; if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (data.schema !== 'spirob-calibration-v1') throw new Error('schema');
    Object.assign(settings, data.settings);
    objectKindInput.value = settings.objectKind;
    objectSizeInput.value = String(settings.objectSize);
    objectMassInput.value = String(settings.objectMass);
    document.querySelector<HTMLInputElement>('#total-mass')!.value = String(settings.totalMass);
    syncForceUI();
    physics.reset();
    rebuildObjectMesh();
  } catch { alert('SpiRob calibration v1形式のJSONを選択してください。'); }
});

// Test-object dragging on a horizontal plane.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0));
let draggingObject = false;
renderer.domElement.addEventListener('pointerdown', (event) => {
  if (!objectMesh || !physics.objectBody) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  if (raycaster.intersectObject(objectMesh).length) {
    automation = null; setGraspStage('idle');
    graspTargetStaged = false; physics.releaseGrasp();
    draggingObject = true; controls.enabled = false; dragPlane.constant = -physics.objectBody.position.y;
    physics.objectBody.type = CANNON.Body.KINEMATIC; physics.objectBody.updateMassProperties();
    renderer.domElement.setPointerCapture(event.pointerId);
  }
});
renderer.domElement.addEventListener('pointermove', (event) => {
  if (!draggingObject || !physics.objectBody) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(dragPlane, hit)) physics.objectBody.position.set(hit.x, hit.y, hit.z);
});
renderer.domElement.addEventListener('pointerup', () => {
  if (!draggingObject || !physics.objectBody) return;
  draggingObject = false; controls.enabled = true;
  physics.objectBody.type = CANNON.Body.DYNAMIC; physics.objectBody.mass = settings.objectMass; physics.objectBody.updateMassProperties();
  physics.objectBody.velocity.setZero(); physics.objectBody.angularVelocity.setZero();
});

function setGraspStage(stage: GraspStage) {
  graspStage = stage;
  const hud = document.querySelector<HTMLElement>('#grasp-hud')!;
  hud.classList.toggle('visible', stage !== 'idle');
  const displayStage = stage === 'holding' ? 'HOLDING' : stage.toUpperCase();
  document.querySelector('#grasp-stage-hud')!.textContent = displayStage;
  const order: GraspStage[] = ['packing', 'reaching', 'wrapping', 'grasping'];
  const current = stage === 'holding' ? order.length : order.indexOf(stage);
  document.querySelectorAll<HTMLElement>('[data-grasp-step]').forEach((item, index) => {
    item.classList.toggle('active', index === current);
    item.classList.toggle('completed', index < current || stage === 'holding');
  });
  const runButton = document.querySelector<HTMLButtonElement>('#run-grasp-btn')!;
  const runLabel = runButton.querySelector<HTMLElement>('b')!;
  const runDetail = runButton.querySelector<HTMLElement>('small')!;
  const stageLabels: Record<Exclude<GraspStage, 'idle'>, string> = {
    packing: '準備中', reaching: '対象へ接近中', wrapping: '巻き付き中', grasping: '把持中', holding: '把持完了 — 再実行',
  };
  runButton.classList.toggle('running', stage !== 'idle' && stage !== 'holding');
  runButton.classList.toggle('complete', stage === 'holding');
  runButton.setAttribute('aria-pressed', String(stage !== 'idle'));
  runLabel.textContent = stage === 'idle' ? '把持シミュレーション実行' : stageLabels[stage];
  runDetail.textContent = stage === 'idle' || stage === 'holding'
    ? 'PACK → REACH → WRAP → GRASP'
    : `${Math.max(1, current + 1)} / 4 · 自動シーケンス実行中`;
}

function runFullGraspSimulation() {
  if (settings.objectKind === 'none') return;
  if (paused) {
    paused = false;
    const pauseButton = document.querySelector<HTMLButtonElement>('#pause-btn')!;
    pauseButton.classList.remove('active');
    pauseButton.querySelector('.pause-icon')!.textContent = 'Ⅱ';
    pauseButton.querySelector('span:last-child')!.textContent = '一時停止';
  }
  graspTargetStaged = false;
  automation = null;
  settings.cableForces = [0, 0, 0];
  physics.reset();
  syncForceUI();
  setGraspStage('idle');
  beginAutoGrasp(performance.now());
  document.querySelectorAll('[data-preset]').forEach((button) => {
    button.classList.toggle('selected', (button as HTMLElement).dataset.preset === 'grasp');
  });
}

document.querySelector('#run-grasp-btn')!.addEventListener('click', runFullGraspSimulation);

function stageGraspTarget() {
  if (!physics.objectBody) return;
  graspTargetStaged = true;
  physics.objectBody.type = CANNON.Body.KINEMATIC;
  // Repeatable presentation fixture: this leaves roughly 70 mm of the distal
  // body available to follow the circumference of the default 52 mm sphere.
  physics.objectBody.position.set(-0.03, -0.06, 0);
  physics.objectBody.quaternion.set(0, 0, 0, 1);
  physics.objectBody.velocity.setZero();
  physics.objectBody.angularVelocity.setZero();
  physics.objectBody.updateMassProperties();
}

function releaseGraspTarget() {
  if (!graspTargetStaged || !physics.objectBody) return;
  graspTargetStaged = false;
  physics.objectBody.type = CANNON.Body.DYNAMIC;
  physics.objectBody.mass = settings.objectMass;
  physics.objectBody.velocity.setZero();
  physics.objectBody.angularVelocity.setZero();
  physics.objectBody.updateMassProperties();
  physics.objectBody.wakeUp();
}

function beginAutoGrasp(now: number) {
  if (!physics.objectBody) { setGraspStage('idle'); return; }
  stageGraspTarget();
  const root = physics.bodies[0];
  const worldTarget = physics.objectBody.position.vsub(root.position);
  const localTarget = root.quaternion.inverse().vmult(worldTarget);
  const desiredMomentY = -localTarget.z;
  const desiredMomentZ = localTarget.y;
  const phases = [0, (Math.PI * 2) / 3, (Math.PI * 4) / 3];
  let bestScore = -Infinity;
  graspPrimaryCable = 0;
  phases.forEach((phase, cable) => {
    const score = Math.sin(phase) * desiredMomentY - Math.cos(phase) * desiredMomentZ;
    if (score > bestScore) { bestScore = score; graspPrimaryCable = cable; }
  });
  if (settings.mount === 'horizontal') graspPrimaryCable = 0;
  automation = 'grasp';
  automationStart = now;
  setGraspStage('packing');
}

function runAutomation(now: number) {
  if (!automation) return;
  const t = (now - automationStart) / 1000;
  if (automation === 'whip') {
    physics.wrapProgress = 0;
    if (t < 0.45) settings.cableForces = [Math.min(24, t / 0.45 * 24), 0, 0];
    else if (t < 0.52) settings.cableForces = [24 * (1 - (t - 0.45) / 0.07), 0, 0];
    else if (t < 1.4) settings.cableForces = [0, 0, 0];
    else { automationStart = now; }
  } else if (automation === 'grasp') {
    const command = autoGraspCommand(t, graspPrimaryCable);
    setGraspStage(command.phase);
    physics.wrapProgress = command.wrapProgress;
    settings.cableForces = command.cableForces;
    if (command.releaseTarget) releaseGraspTarget();
  }
  syncForceUI();
}

const chart = document.querySelector<HTMLCanvasElement>('#velocity-chart')!;
const chartContext = chart.getContext('2d')!;
const velocityHistory = Array(100).fill(0) as number[];
let telemetryAccumulator = 0;
function updateTelemetry(dt: number) {
  telemetryAccumulator += dt;
  if (telemetryAccumulator < 0.05) return;
  telemetryAccumulator = 0;
  const tip = physics.bodies.at(-1)!;
  const root = physics.bodies[0];
  const speed = physics.tipSpeed();
  document.querySelector('#tip-speed')!.textContent = speed.toFixed(3);
  document.querySelector('#tip-x')!.textContent = ((tip.position.x - root.position.x) * 1000).toFixed(1);
  document.querySelector('#tip-y')!.textContent = ((tip.position.y - root.position.y) * 1000).toFixed(1);
  document.querySelector('#tip-z')!.textContent = ((tip.position.z - root.position.z) * 1000).toFixed(1);
  document.querySelector('#contacts')!.textContent = String(physics.contactCount);
  const quality = Math.round(physics.graspQuality * 100);
  const state = graspStage === 'holding'
    ? (physics.isGrasping ? 'HOLDING' : 'SEEKING')
    : physics.isGrasping ? 'CLOSED' : graspStage === 'idle' ? 'IDLE' : graspStage.toUpperCase();
  const graspState = document.querySelector<HTMLElement>('#grasp-state')!;
  graspState.textContent = state;
  graspState.classList.toggle('holding', physics.isGrasping);
  document.querySelector<HTMLOutputElement>('#grasp-quality')!.value = `${quality}%`;
  document.querySelector<HTMLElement>('#quality-fill')!.style.width = `${quality}%`;
  document.querySelector('#grasp-force')!.textContent = `${physics.graspForce.toFixed(1)} N`;
  document.querySelector('#contact-units')!.textContent = physics.contactingUnits.length
    ? physics.contactingUnits.map((unit) => unit + 1).join(', ')
    : '—';
  document.querySelector('#object-position')!.textContent = physics.objectBody
    ? [physics.objectBody.position.x, physics.objectBody.position.y, physics.objectBody.position.z]
      .map((value) => (value * 1000).toFixed(0)).join(' / ')
    : '—';
  const min = Math.floor(physics.simulatedTime / 60).toString().padStart(2, '0');
  const sec = Math.floor(physics.simulatedTime % 60).toString().padStart(2, '0');
  const ms = Math.floor((physics.simulatedTime % 1) * 1000).toString().padStart(3, '0');
  document.querySelector('#timecode')!.textContent = `${min}:${sec}.${ms}`;
  velocityHistory.push(speed); velocityHistory.shift();
  chartContext.clearRect(0, 0, chart.width, chart.height);
  const gradient = chartContext.createLinearGradient(0, 0, 0, chart.height);
  gradient.addColorStop(0, 'rgba(65,240,190,.32)'); gradient.addColorStop(1, 'rgba(65,240,190,0)');
  chartContext.beginPath();
  velocityHistory.forEach((v, i) => {
    const x = i / (velocityHistory.length - 1) * chart.width;
    const y = chart.height - Math.min(1, v / 2.5) * (chart.height - 5);
    if (i === 0) chartContext.moveTo(x, y); else chartContext.lineTo(x, y);
  });
  chartContext.lineTo(chart.width, chart.height); chartContext.lineTo(0, chart.height); chartContext.closePath();
  chartContext.fillStyle = gradient; chartContext.fill();
  chartContext.beginPath();
  velocityHistory.forEach((v, i) => {
    const x = i / (velocityHistory.length - 1) * chart.width;
    const y = chart.height - Math.min(1, v / 2.5) * (chart.height - 5);
    if (i === 0) chartContext.moveTo(x, y); else chartContext.lineTo(x, y);
  });
  chartContext.strokeStyle = '#49e6b6'; chartContext.lineWidth = 1.5; chartContext.stroke();
}

function resize() {
  const { clientWidth, clientHeight } = viewport;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / Math.max(1, clientHeight);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

let lastTime = performance.now();
function animate(now: number) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.04, (now - lastTime) / 1000);
  lastTime = now;
  if (!paused) { runAutomation(now); physics.step(dt); }
  controls.update();
  updateSceneObjects();
  updateTelemetry(dt);
  renderer.render(scene, camera);
}
resize();
syncForceUI();
requestAnimationFrame(animate);
