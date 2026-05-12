import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const jobId = new URLSearchParams(location.search).get("job_id");
const baseUrl = `/jobs/${jobId}/files/`;

const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const downloadLink = document.getElementById("download-link");
const controlsHint = document.getElementById("controls-hint");

if (jobId) {
  downloadLink.href = `/download/${jobId}`;
}

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111827);
scene.fog = new THREE.Fog(0x111827, 15, 60);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 500);
camera.position.set(0, 1.6, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById("canvas-container").appendChild(renderer.domElement);

// Lighting — even ambient for interior
const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xfff5e0, 0.6);
dirLight.position.set(5, 8, 5);
dirLight.castShadow = true;
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0xe0f0ff, 0.3);
fillLight.position.set(-5, 4, -5);
scene.add(fillLight);

// Orbit controls — good default for room inspection
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.1;
controls.maxDistance = 50;
controls.screenSpacePanning = true;

// Load model
function loadModel() {
  const mtlLoader = new MTLLoader();
  mtlLoader.setPath(baseUrl);

  mtlLoader.load(
    "model.mtl",
    (materials) => {
      materials.preload();
      const objLoader = new OBJLoader();
      objLoader.setMaterials(materials);
      objLoader.setPath(baseUrl);
      objLoader.load(
        "model.obj",
        (object) => onModelLoaded(object),
        (xhr) => {
          if (xhr.total > 0) {
            const pct = Math.round((xhr.loaded / xhr.total) * 100);
            loadingText.textContent = `Loading model... ${pct}%`;
          }
        },
        (err) => showLoadError(err)
      );
    },
    null,
    () => {
      // MTL failed — try loading OBJ alone with a default material
      const objLoader = new OBJLoader();
      objLoader.setPath(baseUrl);
      objLoader.load(
        "model.obj",
        (object) => {
          object.traverse((child) => {
            if (child.isMesh) {
              child.material = new THREE.MeshLambertMaterial({ color: 0xcccccc, side: THREE.DoubleSide });
            }
          });
          onModelLoaded(object);
        },
        null,
        (err) => showLoadError(err)
      );
    }
  );
}

function onModelLoaded(object) {
  // Apply double-sided material to all meshes (important for room interiors
  // where the camera is inside — faces may be backfacing)
  object.traverse((child) => {
    if (child.isMesh) {
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => { m.side = THREE.DoubleSide; });
      } else {
        child.material.side = THREE.DoubleSide;
      }
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  // Center and scale the room to a comfortable viewing size
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  // Normalize so the longest dimension = 10 units (approx 10 m room)
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = 10 / maxDim;
  object.scale.setScalar(scale);

  // Re-compute box after scaling
  object.position.sub(center.multiplyScalar(scale));

  scene.add(object);

  // Recompute bounding box of placed object
  const scaledBox = new THREE.Box3().setFromObject(object);
  const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
  const scaledSize = scaledBox.getSize(new THREE.Vector3());

  // Position camera at eye height inside the room (near one corner)
  const eyeHeight = scaledSize.y * 0.45; // ~45% up from floor
  camera.position.set(
    scaledCenter.x,
    scaledBox.min.y + eyeHeight,
    scaledCenter.z
  );

  // Look toward the center of the room at eye level
  controls.target.set(scaledCenter.x, scaledBox.min.y + eyeHeight, scaledCenter.z - 1);
  controls.update();

  // Adjust far plane for room scale
  camera.far = Math.max(size.x, size.z) * scale * 4;
  camera.updateProjectionMatrix();

  loadingOverlay.style.opacity = "0";
  setTimeout(() => { loadingOverlay.style.display = "none"; }, 400);

  // Fade hint after 5s
  setTimeout(() => { controlsHint.style.opacity = "0"; }, 5000);
}

function showLoadError(err) {
  loadingText.textContent = "Failed to load model. The job may still be processing.";
  console.error("Model load error:", err);
}

// Resize handler
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

animate();
loadModel();
