const video = document.querySelector("video");
const worker = new Worker("worker.js");
const yolo_classes = ["Paper", "Rock", "Scissors"];

let interval;
let boxes = [];
let busy = false;
let inferCount = 0;
let totalInferTime = 0;

window.navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((error) => {
    alert("You have to give browser the Webcam permission to run detection");
  });

video.addEventListener("play", () => {
  const canvas = document.querySelector("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  interval = setInterval(() => {
    console.log("interval");
    context.drawImage(video, 0, 0);
    draw_boxes(canvas, boxes);
    const input = prepare_input(canvas);
    if (!busy) {
      const startTime = performance.now(); // 记录开始时间
      worker.postMessage({ input, startTime }); // 将开始时间发送到 worker
      busy = true;
    }
  }, 30);
});

video.addEventListener("pause", () => {
  clearInterval(interval);
});

const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
playBtn.addEventListener("click", () => {
  video.play();
});
pauseBtn.addEventListener("click", () => {
  video.pause();
});

worker.onmessage = (event) => {
  const output = event.data;
  if (output.type === "modelLoaded") {
    document.getElementById("loading").style.display = "none";
    document.getElementById("btn-group").style.display = "block";
  } else if (output.type === "modelResult") {
    const endTime = performance.now(); // 记录结束时间
    const inferTime = endTime - output.startTime; // 计算执行时间
    inferCount++;
    totalInferTime += inferTime;
    const averageInferTime = parseInt(totalInferTime / inferCount);
    console.log(`Infer count: ${inferCount}`);
    console.log(`Average infer time: ${averageInferTime} ms`);

    const canvas = document.querySelector("canvas");
    boxes = process_output(output.result, canvas.width, canvas.height);
    busy = false;
  }
};

function prepare_input(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 640;
  const context = canvas.getContext("2d");
  context.drawImage(img, 0, 0, 640, 640);

  const data = context.getImageData(0, 0, 640, 640).data;
  const red = [],
    green = [],
    blue = [];
  for (let index = 0; index < data.length; index += 4) {
    red.push(data[index] / 255);
    green.push(data[index + 1] / 255);
    blue.push(data[index + 2] / 255);
  }
  return [...red, ...green, ...blue];
}

function process_output(output, img_width, img_height) {
  let boxes = [];
  for (let index = 0; index < 8400; index++) {
    const [class_id, prob] = [...Array(yolo_classes.length).keys()]
      .map((col) => [col, output[8400 * (col + 4) + index]])
      .reduce((accum, item) => (item[1] > accum[1] ? item : accum), [0, 0]);
    if (prob < 0.5) {
      continue;
    }
    const label = yolo_classes[class_id];
    const xc = output[index];
    const yc = output[8400 + index];
    const w = output[2 * 8400 + index];
    const h = output[3 * 8400 + index];
    const x1 = ((xc - w / 2) / 640) * img_width;
    const y1 = ((yc - h / 2) / 640) * img_height;
    const x2 = ((xc + w / 2) / 640) * img_width;
    const y2 = ((yc + h / 2) / 640) * img_height;
    boxes.push([x1, y1, x2, y2, label, prob]);
  }

  boxes = boxes.sort((box1, box2) => box2[5] - box1[5]);
  const result = [];
  while (boxes.length > 0) {
    result.push(boxes[0]);
    boxes = boxes.filter((box) => iou(boxes[0], box) < 0.7);
  }
  return result;
}

function iou(box1, box2) {
  return intersection(box1, box2) / union(box1, box2);
}

function union(box1, box2) {
  const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
  const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
  const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
  const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
  return box1_area + box2_area - intersection(box1, box2);
}

function intersection(box1, box2) {
  const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
  const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
  const x1 = Math.max(box1_x1, box2_x1);
  const y1 = Math.max(box1_y1, box2_y1);
  const x2 = Math.min(box1_x2, box2_x2);
  const y2 = Math.min(box1_y2, box2_y2);
  return (x2 - x1) * (y2 - y1);
}

function draw_boxes(canvas, boxes) {
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 3;
  ctx.font = "18px serif";
  boxes.forEach(([x1, y1, x2, y2, label]) => {
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "#00ff00";
    const width = ctx.measureText(label).width;
    ctx.fillRect(x1, y1, width + 10, 25);
    ctx.fillStyle = "#000000";
    ctx.fillText(label, x1, y1 + 18);
  });

  // 绘制 Infer count 和 Average infer time
  ctx.font = "16px Arial";
  ctx.fillStyle = "black";
  ctx.fillText(`Infer count: ${inferCount}`, 10, 20);
  ctx.fillText(
    `Average infer time: ${
      inferCount ? parseInt(totalInferTime / inferCount) : 0
    } ms`,
    10,
    40,
  );
}
