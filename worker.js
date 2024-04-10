importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js",
);

let model = null;
ort.InferenceSession.create("./rps_best_uint8.onnx", {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
}).then((res) => {
  model = res;
  console.log("model", model);
  postMessage({ type: "modelLoaded" });
});

async function run_model(input) {
  if (!model) {
    model = await model;
  }
  input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
  const outputs = await model.run({ images: input });
  return outputs["output0"].data;
}

onmessage = async (event) => {
  const { input, startTime } = event.data;
  const output = await run_model(input);
  postMessage({ type: "modelResult", result: output, startTime });
};
