const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const context = canvas.getContext('2d');
const matchResultDiv = document.getElementById('matchResult');
const notMatchResultDiv = document.getElementById('notMatchResult');
const retryButton = document.getElementById('retry');

const MATCH_THRESHOLD = 0.495; 

let users = []; 
let datasetDescriptors = [];

async function loadModels() {
  const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models/';
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
}

async function loadUsers() {
  try {
    const response = await fetch('usersList.json');
    const data = await response.json();
    users = data;
    console.log('Users loaded:', users);
  } catch (error) {
    console.error('Error loading usersList.json:', error);
  }
}

async function loadDataset() {
  for (let user of users) {
    const imgPath = `dataset/${user.national_code}.jpeg`;
    try {
      const img = await faceapi.fetchImage(imgPath);
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (detection) {
        const fullName = `${user.name} ${user.last_name}`;
        datasetDescriptors.push({ descriptor: detection.descriptor, label: fullName, national_code: user.national_code });
      } else {
        console.log(`No face detected in ${imgPath}`);
      }
    } catch (error) {
      console.error(`Error loading image ${imgPath}:`, error);
    }
  }
  console.log("Dataset descriptors:", datasetDescriptors);
}

async function startVideo() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }
}

function findBestMatch(faceDescriptor) {
  let bestMatch = null;
  let smallestDistance = Infinity;
  for (let data of datasetDescriptors) {
    const distance = faceapi.euclideanDistance(faceDescriptor, data.descriptor);
    if (distance < smallestDistance) {
      smallestDistance = distance;
      bestMatch = data;
    }
  }
  return { bestMatch, smallestDistance };
}

async function detectFaceOnce() {
  await new Promise(resolve => setTimeout(resolve, 1000));
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();

  context.clearRect(0, 0, canvas.width, canvas.height);

  if (detection) {
    const resizedDetection = faceapi.resizeResults(detection, {
      width: video.videoWidth,
      height: video.videoHeight,
    });
  
    const box = resizedDetection.detection.box;
    context.beginPath();
    context.rect(box.x, box.y, box.width, box.height);
    context.lineWidth = 2;
    context.strokeStyle = "red";
    context.stroke();

    const { bestMatch, smallestDistance } = findBestMatch(detection.descriptor);
    if (bestMatch && smallestDistance < MATCH_THRESHOLD) {
      context.strokeStyle = "green";
      context.stroke();
      matchResultDiv.innerText = `Face match confirmed for ${bestMatch.label}.`;
      notMatchResultDiv.innerText = '';
      retryButton.style.display = "inline-block";
    } else {
      notMatchResultDiv.innerText = "No matching user found.";
      matchResultDiv.innerText = "";
      await detectFaceOnce();
    }

  } else {
    notMatchResultDiv.innerText = "No face detected. Please try again.";
    matchResultDiv.innerText = "";
    await detectFaceOnce();
  }

}

retryButton.addEventListener('click', async () => {
  retryButton.style.display = "none";
  matchResultDiv.innerText = "";
  await startVideo();
  await detectFaceOnce();
});

async function main() {
  await loadModels();
  await loadUsers();
  await loadDataset();
  await startVideo();
  await detectFaceOnce();
}

window.addEventListener('load', main);
