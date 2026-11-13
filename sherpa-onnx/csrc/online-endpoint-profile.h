// sherpa-onnx/csrc/online-endpoint-profile.h
//
// Copyright (c)  2026
//
// Flag-gated endpoint-latency tracer. Disabled unless the environment variable
// SHERPA_EP_PROFILE=<output path> is set; when disabled every call site costs
// one predictable branch on a constant bool.
//
// Output: tab separated, one event per line:
//   t_ns  sid  event  a  b  c
// t_ns is wall-clock (CLOCK_REALTIME) nanoseconds, so it can be joined with
// a client on the same host (or NTP-synced hosts). sid is a per-process
// stream id, 0 for process-wide gauges. a/b/c meaning depends on the event;
// see the table in scripts/ep_latency_breakdown.py (harness repo) and the
// comments at the call sites.
#ifndef SHERPA_ONNX_CSRC_ONLINE_ENDPOINT_PROFILE_H_
#define SHERPA_ONNX_CSRC_ONLINE_ENDPOINT_PROFILE_H_

#include <cstdint>

namespace sherpa_onnx {

namespace epprof {

// a/b/c are documented per event.
enum class Ev : int32_t {
  kConnOpen = 0,       // a=client remote port
  kMsgRx = 1,          // network thread got an audio msg. a=num samples
  kAudioAccept = 2,    // samples handed to the stream. a=n, b=cumulative n
                       //   (input sample rate)
  kInputFinished = 3,  // "Done" processed. a=tail padding samples,
                       //   b=cumulative samples incl. padding
  kVadTick = 4,        // VAD loop tick with work. a=streams queued,
                       //   b=ms since previous tick
  kVadBatch = 5,       // VadDecode(). a=batch size, b=remaining queue,
                       //   c=duration ns
  kVadInfer = 6,       // per stream VadProcessStep. a=windows scored,
                       //   b=duration ns, c=total windows ready
  kVadFlip = 7,        // speech state change. a=new is_speech,
                       //   b=window index, c=audio position in VAD samples
  kReadyBlockBegin = 8,  // IsReady blocked on VAD coverage. a=deficit frames
  kReadyBlockEnd = 9,    // unblocked. a=blocked duration ns
  kVadSkip = 10,         // MaybeSkipVadSilence. a=frames skipped
  kAsrTick = 11,         // ProcessConnections tick. a=ms since previous tick,
                         //   b=ready queue size, c=num connections
  kReadyQueued = 12,     // stream pushed to ready queue. a=queue size
  kDecodeBatch = 13,     // Decode() batch. a=batch, b=remaining, c=dur ns
  kEndpointEval = 14,    // after DecodeStreams. a=is_endpoint, b=rule,
                         //   c=bit0 vad_had_speech, bit1 vad_is_speech,
                         //   bit2 vad_endpoint
  kResultPost = 15,      // result queued for send. a=is_final, b=rule,
                         //   c=endpoint_delay in ms
  kWsSend = 16,          // result actually sent. a=is_final
  kDoneSend = 17,        // "Done!" sent
  kStreamCreate = 18,    // GetOrCreateConnection new stream. a=ns waiting for
                         //   decoder mutex_, b=ns inside CreateStream()
};

bool Enabled();
void Log(Ev ev, int64_t sid, int64_t a = 0, int64_t b = 0, int64_t c = 0);
int64_t NowNs();  // wall-clock ns; use to time sections, only when Enabled()

}  // namespace epprof

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_ENDPOINT_PROFILE_H_
