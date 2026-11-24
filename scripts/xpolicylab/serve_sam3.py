"""Serve SAM3 segmentation over HTTP for roboclock.policies.vision_llm_primitives.

  python scripts/xpolicylab/serve_sam3.py $MODELS_DIR/sam3 --port 8232 --gpu 6

The weights are facebook/sam3 (Hugging Face, gated: request access, then
`hf download facebook/sam3 --local-dir $MODELS_DIR/sam3`). The SAM License applies.
transformers >= 5 is needed (the vLLM venv has it).

Endpoints (JSON in, JSON out; images and masks are base64 PNG):

  GET  /health   -> {"ok": true, "point_prompts": bool}
  POST /segment  {"image_png_b64", "prompts": [text, ...], "min_score"?: 0.3, "max_instances"?: 10}
                 -> {"results": {text: [{"mask_png_b64", "score", "box": [x0, y0, x1, y1]}, ...]}}
                 Text-prompted instance segmentation (Sam3Model), the request and reply format of
                 robotcs_icl's skills/sam3_anchor.py.
  POST /point    {"image_png_b64", "points": [[x, y], ...], "labels"?: [1, ...]}
                 -> {"masks": [{"mask_png_b64", "score"}, ...]} best first
                 Point-prompted masks (Sam3TrackerModel), when the checkpoint provides the tracker.

One request runs at a time on the GPU.
"""

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image


def png_b64(mask):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(text):
    return Image.open(io.BytesIO(base64.b64decode(text))).convert("RGB")


class Segmenter:
    def __init__(self, path, device):
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        # A weight missing from the checkpoint would be left at its random initialisation.
        self.model, info = Sam3Model.from_pretrained(path, torch_dtype=self.dtype, output_loading_info=True)
        if info["missing_keys"]:
            raise RuntimeError(f"{path} lacks {len(info['missing_keys'])} SAM3 weights, e.g. {info['missing_keys'][:3]}")
        self.model = self.model.to(device).eval()
        self.processor = Sam3Processor.from_pretrained(path)
        self.tracker = self.tracker_processor = None
        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            tracker, info = Sam3TrackerModel.from_pretrained(path, torch_dtype=self.dtype, output_loading_info=True)
            if info["missing_keys"]:
                raise RuntimeError(f"the checkpoint lacks {len(info['missing_keys'])} tracker weights")
            self.tracker = tracker.to(device).eval()
            self.tracker_processor = Sam3TrackerProcessor.from_pretrained(path)
        except Exception as error:  # the checkpoint may hold the detector only
            print(f"[sam3] point prompts unavailable: {error}", flush=True)
        self.lock = threading.Lock()

    @torch.no_grad()
    def segment(self, image, prompts, min_score=0.3, max_instances=10):
        results = {}
        with self.lock:
            for prompt in prompts:
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
                outputs = self.model(**inputs)
                found = self.processor.post_process_instance_segmentation(
                    outputs, threshold=min_score, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
                )[0]
                order = torch.argsort(found["scores"], descending=True)[:max_instances]
                results[prompt] = [
                    {
                        "mask_png_b64": png_b64(found["masks"][k].cpu().numpy() > 0),
                        "score": float(found["scores"][k]),
                        "box": [float(x) for x in found["boxes"][k].tolist()],
                    }
                    for k in order.tolist()
                ]
        return {"results": results}

    @torch.no_grad()
    def point(self, image, points, labels=None):
        if self.tracker is None:
            raise ValueError("this checkpoint has no point-prompt tracker")
        labels = labels or [1] * len(points)
        with self.lock:
            inputs = self.tracker_processor(
                images=image, input_points=[[points]], input_labels=[[labels]], return_tensors="pt"
            ).to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            outputs = self.tracker(**inputs, multimask_output=True)
            masks = self.tracker_processor.post_process_masks(outputs.pred_masks.float().cpu(), inputs["original_sizes"])[0][0]
            scores = outputs.iou_scores.float().cpu()[0, 0]
        order = torch.argsort(scores, descending=True).tolist()
        return {"masks": [{"mask_png_b64": png_b64(masks[k].numpy()), "score": float(scores[k])} for k in order]}


def handler(segmenter):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ok": True, "point_prompts": segmenter.tracker is not None})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                image = decode_image(request["image_png_b64"])
                if self.path == "/segment":
                    payload = segmenter.segment(
                        image, list(request["prompts"]), float(request.get("min_score", 0.3)),
                        int(request.get("max_instances", 10)),
                    )
                elif self.path == "/point":
                    payload = segmenter.point(image, request["points"], request.get("labels"))
                else:
                    return self.reply(404, {"error": "not found"})
                self.reply(200, payload)
            except (KeyError, ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Directory with the facebook/sam3 files")
    parser.add_argument("--port", type=int, default=8232)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpu", default="0", help="CUDA device index, or 'cpu'")
    args = parser.parse_args()
    device = "cpu" if args.gpu == "cpu" else f"cuda:{args.gpu}"
    segmenter = Segmenter(args.model, device)
    print(f"[sam3] serving on http://{args.host}:{args.port} ({device})", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler(segmenter)).serve_forever()


if __name__ == "__main__":
    main()
