# Changelog

## 0.1.0 - first public release

- License: GNU AGPL-3.0. Commercial licenses are available (see COMMERCIAL-LICENSE.md).
- An engine for Ternary-Bonsai-2-2B on the NVIDIA Quadro K420. It generates 20.8 tokens/s with the exact head and
  23 tokens/s with the fast head. It reads prompts at 26.5 tokens/s. The context is 24,576 tokens in 1 GB.
- With the exact head, the output is token-for-token identical to llama.cpp (`kindi verify`).
- `kindi convert` makes an exact conversion from the PTQ1_0 GGUF.
- An OpenAI-compatible server with chat completions, streams, tool calls, an optional API key and live progress in
  `/health`.
- Thinking mode: the server sends the thinking as `reasoning_content` (in a stream, as `delta.reasoning_content`).
  `thinking_budget_tokens` sets a limit. At the limit, the server closes the thinking and the model gives its answer.
- When `max_tokens` stops a reply in the middle of a tool call, the server returns that text. The client can then
  see that the reply is not complete.
- A prompt cache: checkpoints of the recurrent state, parked conversations in RAM and saved openings on disk.
- An optional fast (shortlist) output head. You can change the head at run time (`kindi head`).
- A Docker image with the CUDA 10.2 / CuPy 9.6 stack that Kepler needs.
- When a client disconnects (timeout, Ctrl-C), the server stops the request at the next token. This is also true
  during a long prompt read.
