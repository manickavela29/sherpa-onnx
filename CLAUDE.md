# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

sherpa-onnx: offline/streaming ASR, TTS, VAD, speaker diarization/ID, KWS, punctuation, speech enhancement, etc., running on ONNX Runtime (plus RKNN/QNN/Axera/Ascend/etc. backends). The core is C++; every other language binding wraps it.

## Build

CMake project; the root `CMakeLists.txt` options are `SHERPA_ONNX_*` (e.g. `SHERPA_ONNX_ENABLE_TESTS`, `SHERPA_ONNX_ENABLE_PYTHON`, `SHERPA_ONNX_ENABLE_GPU`, `SHERPA_ONNX_ENABLE_TTS`, `SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION`, `BUILD_SHARED_LIBS`). `build/` is the local build dir (tests and Python are OFF in it by default).

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TESTS=ON ..
make -j$(nproc)
```

- Cross-compile / platform builds are the top-level `build-*.sh` scripts (android, ios, wasm, aarch64, riscv64, rknn, ...).
- Python wheel: `python setup.py install` (or `pip install .`); `SHERPA_ONNX_CMAKE_ARGS` passes extra cmake flags.
- Tests are GoogleTest, built only with `SHERPA_ONNX_ENABLE_TESTS=ON`. Each `*-test.cc` becomes its own executable named after the file, registered with ctest. Add new test files to `sherpa_onnx_test_srcs` in `sherpa-onnx/csrc/CMakeLists.txt`.
  ```bash
  cd build && ctest -R circular-buffer-test --output-on-failure   # single test
  ./bin/circular-buffer-test --gtest_filter='Suite.Case'            # single case (check build/bin or build/sherpa-onnx/csrc)
  ```
- Python tests: `sherpa-onnx/python/tests/` (pytest style; some need downloaded models).
- Lint: cpplint 2.0.2 via `./scripts/check_style_cpplint.sh` (no arg = last commit, `1` = uncommitted changes, `2` = whole repo). Config in `CPPLINT.cfg`; CI is `.github/workflows/style_check.yaml`. Headers use `SHERPA_ONNX_CSRC_<NAME>_H_` guards and a `// sherpa-onnx/csrc/<file>` + copyright header comment.

## Architecture

- `sherpa-onnx/csrc/` — all core C++ (~600 files), plus the CLI executables (`sherpa-onnx.cc`, `sherpa-onnx-offline.cc`, `*-microphone*.cc`, websocket servers, ...). `csrc/README.md` lists the binaries.
- Language bindings are thin layers over core, and a new config field/feature usually has to be mirrored in each: `sherpa-onnx/c-api` (C API + `cxx-api`), `sherpa-onnx/python/csrc` (pybind11) and `python/sherpa_onnx`, `sherpa-onnx/jni` + `java-api` + `kotlin-api`, `rust`, `pascal-api`, and the wrappers in `swift-api-examples`, `dotnet-examples`, `go-api-examples`, `nodejs-addon-examples`, `flutter`, `dart-api-examples`, `harmony-os`, `wasm`. Each language has a `*-examples` dir that CI tests.
- **Model-dispatch pattern**: public facades (`OnlineRecognizer`, `OfflineRecognizer`, `OfflineTts`, ...) hold a `*Impl` created by `*Impl::Create(config)` in `*-impl.cc`, which inspects the config / model metadata to pick a concrete implementation (`offline-recognizer-<model>-impl.h`, `online-recognizer-{transducer,ctc,paraformer}-impl.h`). Each model family has its own `*-model.{h,cc}`, `*-model-config.{h,cc}` and `*-model-meta-data.h`. Adding a model = new model + config + impl + registration in the dispatcher + `sources` in `csrc/CMakeLists.txt` + `Config` plumbing in every binding.
- Configs are structs with `Register(ParseOptions*)`, `Validate()`, `ToString()`; CLI flags come from `Register`, and bindings expose the same fields.
- VAD: `VadModel` (`silero-vad-model`, `ten-vad-model`) behind `VoiceActivityDetector`, which segments speech using a `CircularBuffer`.
- `cmake/` holds per-dependency fetch scripts (onnxruntime, kaldi-native-fbank, etc.); dependencies are downloaded at configure time.
- `scripts/` contains per-model export/conversion scripts (not part of the build); `.github/workflows/` has the large CI matrix (per-platform builds, per-language tests, model export).

## Endpoint diagnostics (optional engine metrics)

`--endpoint-diagnostics` (default off, a field of `OnlineRecognizerConfig`, read by the online websocket server) adds endpoint fields to every JSON result so that clients and benchmarks can see which endpoint rule fired and when. With the flag off the JSON is byte-for-byte what it was before. Only transducer models implement it; other models never report an endpoint rule.

| field | meaning |
|---|---|
| `endpoint_rule` | `0` not an endpoint, `1` blank-based (rule1, rule2 or rule3). Same encoding as `scripts/test_stream_asr_client_engine.py` in `ds-e2e-asr-testing` (which also reserves `4` for VAD endpoints; this engine has none) |
| `endpoint_rule_id` | `0` none, `1` rule1 (2.4 s of trailing silence, nothing decoded needed), `2` rule2 (1.2 s of trailing silence after decoding something), `3` rule3 (utterance length, 20 s by default). The first rule that is activated in the order rule1, rule2, rule3 |
| `endpoint_time` | only when `endpoint_rule` is not 0: audio time in seconds, on the stream's clock (the clock of `start_time`), at which the endpoint was decided = `(frames of earlier segments + frames of this segment) x 10 ms` |
| `speech_end_time` | only when `endpoint_rule` is not 0: an **estimate** of where speech ended = `endpoint_time - num_trailing_blanks x 4 x 10 ms`, i.e. minus the trailing silence that triggered the rule; at least 0 |

Notes: the response for the end-of-stream (`is_eof`) is not an endpoint, so it reports `endpoint_rule` 0 unless an endpoint rule also fired in the same decoding step. The trailing silence is measured in decoder output frames (4 input frames of 10 ms), so times have 10 ms resolution; the engine decides an endpoint only after a decoding step, so the actual response arrives later than `endpoint_time`. `Endpoint::FiredRule()` (`sherpa-onnx/csrc/endpoint.h`) is the rule evaluation; `Endpoint::IsEndpoint()` is unchanged in behavior. The server reads the values through `OnlineRecognizer::GetEndpointInfo()` **before** `Reset()`, which restarts the segment counters. Tests: `endpoint-test.cc`, and the endpoint cases in `recognizer-result-json-test.cc`.

## Default engine config (EN streaming model, no VAD)

Whenever a task refers to "the base config" / "default config" for running the sherpa engine, use the values below. They come from `/mnt/efs/manickavela/asr_work/refactor_vad/ds-k2-sherpa-onnx/start_e2e_engine.sh` (defaults with `VAD` unset). Re-check that script if in doubt; it is the source of truth.

Binary: `/opt/sherpa/bin/sherpa-onnx-online-websocket-server`.

Model dir (EN global, v2025.07.1.0): `/mnt/efs/manickavela/asr_work/semantic_endpoint_v2/e2e_gpu_streaming_asr_model_en-global_v2025.07.1.0`. It holds `tokens.txt`, `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `bpe.model`, `bpe.vocab`, `model_info.json`, `fp16/encoder.onnx`, `fp32/encoder.onnx` and prebuilt TRT engine/timing cache files (sm86). `start_e2e_engine.sh` expects it mounted at `/opt/k2_sherpa/ai_model/asr/`, so mount it there (e.g. `-v <model dir>:/opt/k2_sherpa/ai_model/asr`) or, for a direct run, set `$MODEL` to the host path.

| Env var / flag | Default |
| --- | --- |
| `WS_SERVER_PORT` (`--port`) | 6006 |
| `MAX_BATCH_SIZE` | 50 |
| `NUM_THREADS` / `NUM_WORKER_THREADS` / `NUM_IO_THREADS` | 100 / 100 / 100 |
| `LOOP_INTERVAL_MS` | 50 |
| `PROVIDER` | `trt` (or `cuda`) |
| `SAMPLE_RATE` | 8000 |
| `BIASING_SCORE` (`--hotwords-score`) | 2.0 (hotwords file is an empty temp file) |
| `VAD` | empty (no fvad/seot/zseot) |
| `ASR_TRT_ENGINE_CACHE_PATH` | `/opt/k2_sherpa/ai_model/asr/trt-cache` (only for `PROVIDER=trt`; see the TRT cache rule below) |

Fixed flags: `--model-type=zipformer2 --decoding-method=modified_beam_search --max-active-paths=10 --end-tail-padding=1.6 --warm-up=10`.

- **Endpointing:** use sherpa-onnx's built-in defaults. Do not pass `ENDPOINT_CONFIG`, `ENDPOINT_RULES` or any `--rule*` / `--tentative-silence-dur` flags unless asked.
- **Silero VAD (optional, commented out by default):** uncomment the `--silero-vad-online-*` lines in the command below only when asked. The current `start_e2e_engine.sh` makes Silero mandatory, so a no-Silero run needs that script changed.
- With `PROVIDER=trt`: `PRECISION=fp32`, encoder is `asr/fp32/encoder.onnx` if present (else `asr/encoder.onnx`), plus `--trt_engine_cache_enable=1 --trt_timing_cache_enable=1` with both paths set to the cache dir above.
- **TRT cache folder (Claude must do this when running with `PROVIDER=trt`):** before starting the server, create a `trt-cache` folder and point `ASR_TRT_ENGINE_CACHE_PATH` at it, so compiled engines persist across restarts instead of being rebuilt every run: `mkdir -p <model dir>/trt-cache && export ASR_TRT_ENGINE_CACHE_PATH=<model dir>/trt-cache` (inside the container that is `/opt/k2_sherpa/ai_model/asr/trt-cache`, which `start_e2e_engine.sh` also creates itself). For a direct run, pass `--trt_engine_cache_enable=1 --trt_engine_cache_path=<that folder> --trt_timing_cache_enable=1 --trt_timing_cache_path=<that folder>`. The first start is slow while TRT builds the engine. Don't use it with `PROVIDER=cuda`.
- BPE hotwords: if `/workspace/sherpa-onnx/scripts/export_bpe_vocab.py` exists, it converts `bpe.model` to `bpe.vocab` and adds `--bpe-vocab=.../asr/bpe.vocab --modeling_unit=bpe`.
- Run it with `bash start_e2e_engine.sh` (override any value via env var, e.g. `PROVIDER=cuda WS_SERVER_PORT=6010 bash start_e2e_engine.sh`).

Base command (Silero lines commented; `$MODEL` is the asr model dir above, `$ENCODER` is the TRT/fp32 encoder resolved as described):

```bash
/opt/sherpa/bin/sherpa-onnx-online-websocket-server \
    --port=6006 \
    --tokens=$MODEL/tokens.txt --encoder=$ENCODER \
    --decoder=$MODEL/decoder.onnx --joiner=$MODEL/joiner.onnx \
    --max-batch-size=50 --loop-interval-ms=50 \
    --num_threads=100 --num-work-threads=100 --num-io-threads=100 \
    --sample-rate=8000 --provider=trt \
    --model-type=zipformer2 --decoding-method=modified_beam_search --max-active-paths=10 \
    --hotwords-file=$EMPTY_FILE --hotwords-score=2.0 \
    --end-tail-padding=1.6 --warm-up=10
    # --silero-vad-online-model=/opt/k2_sherpa/ai_model/silero/silero_vad.onnx \
    # --silero-vad-online-threshold=0.5 \
    # --silero-vad-online-window-size=256 \
    # --silero-vad-online-min-speech-duration=0.03
```
