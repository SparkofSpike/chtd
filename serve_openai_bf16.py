"""Small OpenAI-compatible Qwen3 endpoint for the frozen AGH student route.

Generate a complete turn before emitting an SSE chunk. This preserves tool-call
JSON as one unit and makes training and serving use the same chat renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from distill.finetune import render_chat


_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_assistant_output(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert complete Qwen tool blocks into OpenAI tool calls, or keep raw text on error."""
    blocks = list(_TOOL_BLOCK.finditer(text))
    if not blocks:
        return text.strip(), []
    calls: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        try:
            payload = json.loads(block.group(1))
        except json.JSONDecodeError:
            return text.strip(), []
        name = payload.get("name") if isinstance(payload, dict) else None
        arguments = payload.get("arguments") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name or not isinstance(arguments, (str, dict)):
            return text.strip(), []
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(f"{index}\0{block.group(1)}".encode("utf-8")).hexdigest()
        calls.append({
            "id": "call_" + digest[:24],
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    prose = _TOOL_BLOCK.sub("", text).strip()
    return prose or None, calls


def stream_payload(response: dict[str, Any]) -> bytes:
    """Emit the parsed turn as one OpenAI SSE delta and a terminal chunk."""
    choice = response["choices"][0]
    message = choice["message"]
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [dict(call, index=index)
                               for index, call in enumerate(message["tool_calls"])]
    common = {key: response[key] for key in ("id", "object", "created", "model")}
    common["object"] = "chat.completion.chunk"
    first = dict(common, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
    last = dict(common, choices=[{"index": 0, "delta": {},
                                  "finish_reason": choice["finish_reason"]}])
    return ("data: " + json.dumps(first, ensure_ascii=False) + "\n\n" +
            "data: " + json.dumps(last, ensure_ascii=False) + "\n\n" +
            "data: [DONE]\n\n").encode("utf-8")


def serve(model_path: str, model_id: str, host: str, port: int,
          adapter_path: str | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        # A client that goes away without closing — a killed harness process, a dropped Tailscale
        # link — leaves a socket this handler would otherwise read from forever, and a plain
        # `HTTPServer` serves one request at a time. Measured on the training box: one such socket
        # held the only worker thread and the endpoint stopped answering entirely, with the log
        # silent and three concurrent probes all timing out. The timeout bounds the wait; the
        # threading server keeps one stranded connection from taking the whole endpoint with it.
        # Only socket reads and writes are timed — the model runs on CPU time of its own.
        timeout = 300

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                free, total = torch.cuda.mem_get_info()
                self._send(200, {"model": model_id, "adapter": adapter_path,
                                 "gpu_free_gb": round(free / 1024**3, 2),
                                 "gpu_total_gb": round(total / 1024**3, 2)})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4_000_000:
                    raise ValueError("request body size out of range")
                request = json.loads(self.rfile.read(length))
                messages = request["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a nonempty list")
                tools = request.get("tools")
                if tools is not None and not isinstance(tools, list):
                    raise ValueError("tools must be a list")
                prompt = render_chat(tokenizer, messages, tools, add_generation_prompt=True)
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                max_new = request.get("max_tokens") or 1024
                max_new = min(max(1, int(max_new)), 1024)
                with inference_lock, torch.inference_mode():
                    output = model.generate(
                        **encoded, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0, encoded["input_ids"].shape[1]:].clone()
                text = tokenizer.decode(generated, skip_special_tokens=True)
                # Release the KV cache and activation buffers this request allocated. The
                # caching allocator keeps a block for every sequence length served and does
                # not return them to the driver; measured 2026-10-11, ~100 requests filled
                # the 47 GB card and the server began failing allocations while still
                # answering HTTP 200. `encoded` stays: the response builder reads its shape
                # for `prompt_tokens`, and releasing it here cost every request an
                # UnboundLocalError that a curl to /dev/null does not show.
                content, calls = parse_assistant_output(text)
                response = {
                    "id": "chatcmpl-" + hashlib.sha256(generated.cpu().numpy().tobytes()).hexdigest()[:24],
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                                                      **({"tool_calls": calls} if calls else {})},
                                 "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": int(encoded["input_ids"].shape[1]),
                              "completion_tokens": int(generated.shape[0]),
                              "total_tokens": int(output.shape[1])},
                }
                if request.get("stream"):
                    body = stream_payload(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._send(200, response)
                # Release this request's KV cache and activation buffers. The caching
                # allocator keeps a block for every sequence length served and does not
                # return them to the driver: measured on 2026-10-11, ~100 requests filled
                # 44 of 47 GB and allocations began failing while the server still answered
                # HTTP 200. The release sits after the response is written because
                # `output` and `encoded` supply the usage counts.
                del output, encoded
                torch.cuda.empty_cache()
            except (KeyError, TypeError, ValueError) as error:
                self._send(400, {"error": str(error)[:250]})

    print(f"serving {model_id} on {host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="qwen3-4b-instruct-2507")
    parser.add_argument("--adapter-path", default=None,
                        help="optional PEFT LoRA adapter; omit for the unadapted base")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.model_path, args.model_id, args.host, args.port, args.adapter_path)


if __name__ == "__main__":
    main()
