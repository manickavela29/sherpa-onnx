#!/bin/bash
# Functional smoke test for the online-VAD feature, run as the image's
# default CMD. Two tiers:
#   1. Existing unit test binaries (if SHERPA_ONNX_ENABLE_TESTS built
#      any) -- a cheap regression signal unrelated to VAD.
#   2. If an ASR model + Silero VAD model are mounted at /models (see
#      README.online-vad.md for the expected layout), starts
#      sherpa-onnx-online-websocket-server with --online-vad-model set
#      and drives it with the built-in sherpa-onnx-online-websocket-client
#      against a real wav file, capturing every JSON response (which
#      includes endpoint_rule/endpoint_delay/speech_end_time) to a log
#      for inspection.
#
# This is a functional check, not a load test -- for load testing with
# many concurrent connections, use ds-e2e-asr-testing's
# scripts/test_stream_asr_client_engine.py against a server started the
# same way (see docker/start_websocket_server.sh and
# README.online-vad.md).
set -uo pipefail

BIN_DIR=/opt/sherpa/bin
RESULTS_DIR=${RESULTS_DIR:-/results}
mkdir -p "$RESULTS_DIR"

echo "===== 1. Unit tests ====="
test_fail=0
for t in "$BIN_DIR"/*-test; do
  [ -x "$t" ] || continue
  name=$(basename "$t")
  echo "--- $name ---"
  if "$t" > "$RESULTS_DIR/$name.log" 2>&1; then
    echo "PASS: $name"
  else
    echo "FAIL: $name (see $RESULTS_DIR/$name.log)"
    test_fail=1
  fi
done

echo
echo "===== 2. Online-VAD functional smoke test ====="
MODEL_DIR=${MODEL_DIR:-/models/asr}
VAD_MODEL=${ONLINE_VAD_MODEL:-/models/vad/silero_vad.onnx}
WAV_FILE=${SMOKE_TEST_WAV:-/models/test_wavs/0.wav}

if [ ! -f "$MODEL_DIR/tokens.txt" ] || [ ! -f "$VAD_MODEL" ] || [ ! -f "$WAV_FILE" ]; then
  echo "Skipping: expected an ASR model at \$MODEL_DIR ($MODEL_DIR), a Silero"
  echo "VAD model at \$ONLINE_VAD_MODEL ($VAD_MODEL), and a test wav at"
  echo "\$SMOKE_TEST_WAV ($WAV_FILE). Mount /models (see README.online-vad.md)"
  echo "to run this part."
  exit $test_fail
fi

export MODEL_DIR ONLINE_VAD_MODEL="$VAD_MODEL" PORT=6006 LOG_FILE="$RESULTS_DIR/server.log"
/opt/sherpa/start_websocket_server.sh &
server_pid=$!
trap 'kill $server_pid 2>/dev/null' EXIT

for _ in $(seq 1 30); do
  if (exec 3<>/dev/tcp/127.0.0.1/6006) 2>/dev/null; then
    exec 3>&- 3<&-
    break
  fi
  sleep 0.5
done

echo "Sending $WAV_FILE through sherpa-onnx-online-websocket-client ..."
"$BIN_DIR/sherpa-onnx-online-websocket-client" \
  --server-ip=127.0.0.1 --server-port=6006 \
  --samples-per-message=1600 --seconds-per-message=0.1 \
  "$WAV_FILE" > "$RESULTS_DIR/client.log" 2>&1

echo
echo "JSON responses received (endpoint_rule/endpoint_delay/speech_end_time included):"
grep -o '{.*}' "$RESULTS_DIR/client.log" | tail -5

if grep -q '"endpoint_rule"' "$RESULTS_DIR/client.log"; then
  echo "PASS: responses carry the new endpoint_rule field"
else
  echo "FAIL: no endpoint_rule field seen in any response"
  test_fail=1
fi

kill "$server_pid" 2>/dev/null
wait "$server_pid" 2>/dev/null
trap - EXIT

echo
echo "Results saved under $RESULTS_DIR"
exit $test_fail
