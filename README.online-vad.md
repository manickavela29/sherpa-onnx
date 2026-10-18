# Online VAD (Silero) — design notes & how to test

This documents the optional, independent Silero VAD pipeline added on
branch `feat/online-silero-vad-endpoint` (PR
[manickavela29/sherpa-onnx#1](https://github.com/manickavela29/sherpa-onnx/pull/1)),
and specifically **why it's structured the way it is, compared to the
`refactor_vad` work already shipped internally in `ds-k2-sherpa-onnx`**
(Uniphore's private fork, which this public port draws on). If you've
read that fork's `RFC-VAD-PIPELINE-REFACTOR.md` and are wondering why
this port doesn't look the same, this is the explanation.

## What's the same

Both are solving the same problem: run a VAD independently of the ASR
decode loop (not inline, not at ASR's cadence), gate `IsReady()` on VAD
coverage, skip the encoder over VAD-confirmed silence, and stream
low-cadence pre-speech messages on a websocket server. Both land the
VAD-driven endpoint as an additional signal alongside blank-based rules.

## What's different, and why

### 1. One VAD model, not a pluggable multi-backend abstraction

`ds-k2-sherpa-onnx`'s `refactor_vad` built `OnlineVadModel` — an
abstract base with `OnlineFvadModel` / `OnlineSeotModel` /
`OnlineZseotModel` subclasses, because that fork actually has three
different VAD/endpoint sources (feature-VAD, semantic-EOT, zipformer-EOT)
that are mutually exclusive at runtime, selected by which model path is
configured.

The public repo has exactly one VAD backend relevant here: Silero. FVAD,
SEOT, and ZSEOT are Uniphore-proprietary models that don't exist in
public `sherpa-onnx` and aren't being contributed. With only one backend,
an abstract-base-plus-subclasses design buys nothing — it would be
indirection with a single implementation behind it. This port has no
`OnlineVadModel` equivalent; `OnlineStream` talks to the VAD directly.

### 2. Reusing `VoiceActivityDetector`/`SileroVadModel`, not a new batched `OnlineSileroVad`

`ds-k2-sherpa-onnx` had to write `OnlineSileroVad` + `OnlineVadBuffer`
from scratch: a batched-across-streams ONNX wrapper with explicit
LSTM-state stacking/unstacking (`StackStates`/`UnStackStates`), because
**nothing like it existed yet** in that fork's online path, and because
Silero was being run on the *same* forward-pass budget as the much
heavier SEOT/ZSEOT zipformer encoder — batching many streams into one
ONNX call mattered there for throughput.

The public repo already ships `VoiceActivityDetector` (wrapping
`SileroVadModel`), used today by the standalone VAD+offline-ASR examples.
It already does per-stream windowing, hangover/min-silence-duration
state, and segment bookkeeping — all correctness-sensitive logic that's
already written, reviewed, and shipped. This port gives each
`OnlineStream` its own `VoiceActivityDetector` instance and calls it
directly. No batching, no custom ring buffer, no state-stacking code to
get right from scratch. Silero alone is cheap enough that one instance
per stream is not a meaningful throughput concern — the thing `refactor_vad`
was optimizing for (amortizing one more model's cost against a heavy
zipformer SEOT/ZSEOT encoder) doesn't exist here.

### 3. The gate/skip logic is centralized, not embedded in one decoder impl

`refactor_vad`'s RFC "Files of interest" table lists only
`online-recognizer-transducer-impl.h` as touched for `RunVadStreams`
(coverage gate, batch skip, endpoint reconciliation) — reasonable there,
since that fork's production path is exclusively zipformer2 transducer.

Public `sherpa-onnx` has five-plus online decoder impls (transducer, ctc,
paraformer, transducer-nemo, parakeet-unified, plus RKNN/QNN variants)
that all implement `IsReady()`/`DecodeStreams()` independently. Embedding
VAD-specific logic inside one of them would mean only that impl's users
get the feature, and would require duplicating (and keeping in sync) the
gate/skip algorithm per impl. Instead, `OnlineRecognizerImpl` gained two
one-line virtual accessors:

```cpp
virtual int32_t GetChunkSizeForVad() const { return 0; }   // 0 = unsupported
virtual int32_t GetChunkShiftForVad() const { return 0; }
```

and the actual gate/skip algorithm lives once, in
`OnlineRecognizer::IsReady()`/`MaybeSkipVadSilence()` — the layer that
already wraps every impl uniformly. Four impls (transducer, ctc,
transducer-nemo, parakeet-unified) opt in with one-line overrides
returning an existing constant they already had (`model_->ChunkSize()` /
`ChunkShift()` etc.). Paraformer and the RKNN/QNN impls are deliberately
left at the `0` default (see "Deliberate scope cuts" in the PR
description) rather than guessed at.

### 4. VAD buffering lives inside `AcceptWaveform()`, not a separate ingestion path — **this is the one you asked about**

The alternative considered was a dedicated method (e.g.
`AcceptVadWaveform()`) that the websocket server would call explicitly,
instead of adding a branch inside the shared `AcceptWaveform()` every
caller already goes through. It was rejected for three concrete reasons,
each tied to how audio actually reaches an `OnlineStream` in this repo
today:

1. **`AcceptWaveform()` is already the only entry point for raw audio,
   and the websocket server already calls it per-incoming-message,
   decoupled from the ASR decode timer.** `OnlineWebsocketServer::OnMessage`
   posts `decoder_.AcceptWaveform(c)` to `io_work_` the instant a binary
   frame arrives (`online-websocket-server-impl.cc`) — it does not wait
   for the `timer_`-driven `ProcessConnections()` tick at all. A separate
   `AcceptVadWaveform()` would need its *own* wiring into `OnMessage` to
   get this same decoupling, duplicating plumbing that already exists.

2. **There is exactly one raw-audio queue per connection
   (`Connection::samples`), and it's drained destructively.**
   `OnlineWebsocketDecoder::AcceptWaveform` pops each buffered chunk
   exactly once (`c->samples.pop_front()`) and hands it to
   `OnlineStream::AcceptWaveform`. A second, separate VAD-ingestion method
   would need either its own second queue (fed by `OnMessage`, doubling
   the audio-copy/queueing code and giving it its own set of
   lifetime/ordering bugs to get right) or to tee the data at `OnMessage`
   time before either queue drains it. Branching inside the *existing*
   `AcceptWaveform()` means the VAD sees the exact same samples, in the
   exact same order, via the exact same already-tested code path — by
   construction, not by keeping two paths in sync.

3. **The previously-merged inline-VAD commit (`14bb2125`) already put the
   resampling-to-VAD-rate logic inside `AcceptWaveform()`.** Moving VAD
   buffering to a separate method while leaving resampling in
   `AcceptWaveform()` would split one coherent piece of logic ("what
   happens to incoming audio on its way to the VAD") across two places
   for no benefit — the deferred/inline switch is a four-line `if`
   inside that existing branch (append to a pending buffer vs. score
   immediately), not a structural change to where audio enters the VAD
   path.

The actual correctness-sensitive design decision here isn't *where*
buffering happens (inside vs. beside `AcceptWaveform`) — it's that
**deferred inference defaults to off**, and only the websocket server's
`OnlineWebsocketDecoder` constructor opts in
(`online_vad_config.defer_inference = true`, set automatically only when
a VAD model is configured). A pre-implementation design-review pass
caught that making deferred inference the default — regardless of which
method triggers it — would have silently broken VAD-driven endpointing
for every *other* `OnlineRecognizer`/`OnlineStream` caller in the repo
(`sherpa-onnx-vad-with-online-asr`, `sherpa-onnx-microphone`,
`sherpa-onnx.cc`, …), none of which know to call the new
`RunOnlineVad()` to drive inference. That hazard exists regardless of
whether the buffering branch lives inside `AcceptWaveform()` or in a
sibling method; keeping it inside `AcceptWaveform()` was the simpler of
two options that both needed the same inline-by-default fix.

### 5. The websocket VAD tick only scores + sends pre-speech messages; it never touches ASR decode state

`refactor_vad`'s `RunVadStreams` step 4 (batch skip) and its endpoint
decision both run *inside* the VAD tick, mutating
`num_processed_frames` directly from that thread. This port splits
that: `ProcessVadConnections` (the new VAD tick) only calls
`OnlineStream::RunOnlineVad()` (updates VAD-internal state under
`OnlineStream`'s own lock) and sends pre-speech empty messages. The
*skip* (`MaybeSkipVadSilence`, which does mutate
`num_processed_frames_`) only ever runs from `OnlineRecognizer::IsReady()`,
which is only ever called from the ASR/decode side (`ProcessConnections`/
`Decode()`, or directly by non-websocket callers). This keeps "who is
allowed to mutate ASR decode state" to a single thread-ownership
discipline (the existing `active_`-set-guarded decode path), rather than
introducing a second mutator. The VAD tick additionally excludes any
connection currently in the decoder's `active_` set from its snapshot —
cheap, and removes any timing ambiguity about which side "owns" a stream
at a given instant, even though (see the PR's design notes) the actual
field-level locking already makes this memory-safe either way.

## Building and testing

No executable links in the sandbox this was developed in — an
unrelated, pre-existing environment issue (`undefined reference to
std::__throw_bad_array_new_length()` when linking the prebuilt
onnxruntime 1.28.2 static lib against Ubuntu 20.04's GCC 9/10). The
`Dockerfile` in this directory uses Ubuntu 22.04 (GCC 11) specifically to
get a real, linkable build for testing.

### Build the image

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t sherpa-onnx-online-vad:test .
```

### Functional smoke test (default `CMD`)

Mount an ASR model + Silero VAD model + a test wav at `/models` (see
layout below) and a host directory to collect results at `/results`:

```bash
docker run --rm \
  -v /path/to/streaming-zipformer-model:/models/asr:ro \
  -v /path/to/silero_vad.onnx:/models/vad/silero_vad.onnx:ro \
  -v /path/to/test_wavs:/models/test_wavs:ro \
  -v "$(pwd)/vad_test_results:/results" \
  sherpa-onnx-online-vad:test
```

Expected `/models` layout:
```
/models/asr/tokens.txt
/models/asr/encoder.onnx
/models/asr/decoder.onnx
/models/asr/joiner.onnx
/models/vad/silero_vad.onnx
/models/test_wavs/0.wav
```

This runs every `*-test` unit-test binary the build produced, then (if
the models above are present) starts
`sherpa-onnx-online-websocket-server` with `--online-vad-model` set and
drives it with the built-in `sherpa-onnx-online-websocket-client` against
the test wav, saving server/client logs and a pass/fail summary under
`/results` (bind-mounted to `./vad_test_results` above).

### Running the server standalone (for manual testing or as a load-test target)

```bash
docker run --rm -p 6006:6006 \
  -v /path/to/streaming-zipformer-model:/models/asr:ro \
  -v /path/to/silero_vad.onnx:/models/vad/silero_vad.onnx:ro \
  -e ONLINE_VAD_MODEL=/models/vad/silero_vad.onnx \
  --entrypoint /opt/sherpa/start_websocket_server.sh \
  sherpa-onnx-online-vad:test
```

See `docker/start_websocket_server.sh` for every env var this accepts
(`NUM_WORK_THREADS`, `VAD_LOOP_INTERVAL_MS`, `ONLINE_VAD_THRESHOLD`, …).
Omit `ONLINE_VAD_MODEL` to run with VAD disabled (today's behavior,
unchanged).

For actual concurrent load testing (not just this functional smoke
test), point `ds-e2e-asr-testing`'s engine-mode streaming client at a
server started this way.
