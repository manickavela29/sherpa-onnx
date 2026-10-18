#!/bin/bash
# Starts sherpa-onnx-online-websocket-server for load/manual testing of the
# online-VAD feature, with env-var knobs mirroring ds-k2-sherpa-onnx's
# start_e2e_engine.sh pattern. Not a production entrypoint -- just enough
# to drive the server the same way across local runs, docker compose, and
# ad-hoc `docker run` invocations.
#
# Expects an ASR model directory mounted at $MODEL_DIR (tokens.txt,
# encoder.onnx, decoder.onnx, joiner.onnx) and, if ONLINE_VAD_MODEL is
# set, a Silero VAD onnx at that path. See README.online-vad.md for
# where to get small test models.
set -euo pipefail

PORT=${PORT:-6006}
NUM_IO_THREADS=${NUM_IO_THREADS:-1}
NUM_WORK_THREADS=${NUM_WORK_THREADS:-3}
NUM_VAD_THREADS=${NUM_VAD_THREADS:-1}
LOOP_INTERVAL_MS=${LOOP_INTERVAL_MS:-10}
VAD_LOOP_INTERVAL_MS=${VAD_LOOP_INTERVAL_MS:-20}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
SAMPLE_RATE=${SAMPLE_RATE:-16000}
DECODING_METHOD=${DECODING_METHOD:-"greedy_search"}
MODEL_DIR=${MODEL_DIR:-"/models/asr"}
LOG_FILE=${LOG_FILE:-"/tmp/online-websocket-server.log"}

# Empty (default) = VAD-driven endpointing disabled; set to a Silero VAD
# onnx path (e.g. /models/vad/silero_vad.onnx) to enable it.
ONLINE_VAD_MODEL=${ONLINE_VAD_MODEL:-""}
ONLINE_VAD_THRESHOLD=${ONLINE_VAD_THRESHOLD:-0.5}
ONLINE_VAD_MIN_SILENCE_DURATION=${ONLINE_VAD_MIN_SILENCE_DURATION:-0.5}
ONLINE_VAD_MIN_SPEECH_DURATION=${ONLINE_VAD_MIN_SPEECH_DURATION:-0.25}
ONLINE_VAD_WINDOW_SIZE=${ONLINE_VAD_WINDOW_SIZE:-512}
ONLINE_VAD_SAMPLE_RATE=${ONLINE_VAD_SAMPLE_RATE:-16000}

for f in tokens.txt encoder.onnx decoder.onnx joiner.onnx; do
  if [ ! -f "$MODEL_DIR/$f" ]; then
    echo "Missing $MODEL_DIR/$f -- mount an ASR model directory at $MODEL_DIR" >&2
    exit 1
  fi
done

VAD_ARGS=()
if [ -n "$ONLINE_VAD_MODEL" ]; then
  if [ ! -f "$ONLINE_VAD_MODEL" ]; then
    echo "ONLINE_VAD_MODEL=$ONLINE_VAD_MODEL does not exist" >&2
    exit 1
  fi
  echo "Online VAD enabled: $ONLINE_VAD_MODEL"
  VAD_ARGS=(
    "--online-vad-model=$ONLINE_VAD_MODEL"
    "--online-vad-threshold=$ONLINE_VAD_THRESHOLD"
    "--online-vad-min-silence-duration=$ONLINE_VAD_MIN_SILENCE_DURATION"
    "--online-vad-min-speech-duration=$ONLINE_VAD_MIN_SPEECH_DURATION"
    "--online-vad-window-size=$ONLINE_VAD_WINDOW_SIZE"
    "--online-vad-sample-rate=$ONLINE_VAD_SAMPLE_RATE"
    "--vad-loop-interval-ms=$VAD_LOOP_INTERVAL_MS"
    "--num-vad-threads=$NUM_VAD_THREADS"
  )
else
  echo "Online VAD disabled (ONLINE_VAD_MODEL not set) -- blank-rule endpointing only"
fi

exec sherpa-onnx-online-websocket-server \
  --port="$PORT" \
  --num-io-threads="$NUM_IO_THREADS" \
  --num-work-threads="$NUM_WORK_THREADS" \
  --loop-interval-ms="$LOOP_INTERVAL_MS" \
  --max-batch-size="$MAX_BATCH_SIZE" \
  --input-sample-rate="$SAMPLE_RATE" \
  --log-file="$LOG_FILE" \
  --tokens="$MODEL_DIR/tokens.txt" \
  --encoder="$MODEL_DIR/encoder.onnx" \
  --decoder="$MODEL_DIR/decoder.onnx" \
  --joiner="$MODEL_DIR/joiner.onnx" \
  --model-type=zipformer2 \
  --decoding-method="$DECODING_METHOD" \
  --provider=cpu \
  "${VAD_ARGS[@]}"
