importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js",
);

let model = null;

async function run_model(input) {
  if (!model) {
    model = await ort.InferenceSession.create("./rps_best_uint8.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
  const outputs = await model.run({ images: input });
  return outputs["output0"].data;
}

onmessage = async (event) => {
  const input = event.data;
  const output = await run_model(input);
  postMessage(output);
};
