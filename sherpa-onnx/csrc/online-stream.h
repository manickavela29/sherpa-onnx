// sherpa-onnx/csrc/online-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_ONLINE_STREAM_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldi-decoder/csrc/faster-decoder.h"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/online-paraformer-decoder.h"
#include "sherpa-onnx/csrc/online-stream-state.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct TransducerKeywordResult;
class OnlineStream {
 public:
  explicit OnlineStream(const FeatureExtractorConfig &config = {},
                        ContextGraphPtr context_graph = nullptr);

  virtual ~OnlineStream();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /**
   * InputFinished() tells the class you won't be providing any
   * more waveform.  This will help flush out the last frame or two
   * of features, in the case where snip-edges == false; it also
   * affects the return value of IsLastFrame().
   */
  void InputFinished() const;

  int32_t NumFramesReady() const;

  /** Note: IsLastFrame() will only ever return true if you have called
   * InputFinished() (and this frame is the last frame).
   */
  bool IsLastFrame(int32_t frame) const;

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in row major)
   */
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;

  void Reset();

  int32_t FeatureDim() const;

  // Return a reference to the number of processed frames so far
  // before subsampling..
  // Initially, it is 0. It is always less than NumFramesReady().
  //
  // The returned reference is valid as long as this object is alive.
  int32_t &GetNumProcessedFrames();  // It's reset after calling Reset()

  int32_t GetNumFramesSinceStart() const;

  int32_t &GetCurrentSegment();

  void SetResult(const OnlineTransducerDecoderResult &r);
  OnlineTransducerDecoderResult &GetResult();

  void SetKeywordResult(const TransducerKeywordResult &r);
  TransducerKeywordResult &GetKeywordResult(bool remove_duplicates = false);

  void SetCtcResult(const OnlineCtcDecoderResult &r);
  OnlineCtcDecoderResult &GetCtcResult();

  void SetParaformerResult(const OnlineParaformerDecoderResult &r);
  OnlineParaformerDecoderResult &GetParaformerResult();

  void SetStates(std::vector<Ort::Value> states);
  std::vector<Ort::Value> &GetStates();

  void SetQnnStates(std::vector<OnlineStreamStateTensor> states);
  std::vector<OnlineStreamStateTensor> &GetQnnStates();

  void SetQnnResult(const OnlineTransducerDecoderResultNoOrt &r);
  OnlineTransducerDecoderResultNoOrt &GetQnnResult();

  void SetNeMoDecoderStates(std::vector<Ort::Value> decoder_states);
  void SetNeMoDecoderOut(Ort::Value decoder_out);

  std::vector<Ort::Value> &GetNeMoDecoderStates();
  Ort::Value &GetNeMoDecoderOut();

  /**
   * Get the context graph corresponding to this stream.
   *
   * @return Return the context graph for this stream.
   */
  const ContextGraphPtr &GetContextGraph() const;

  // for online ctc decoder
  void SetFasterDecoder(std::unique_ptr<kaldi_decoder::FasterDecoder> decoder);
  kaldi_decoder::FasterDecoder *GetFasterDecoder() const;
  int32_t &GetFasterDecoderProcessedFrames();

  // for streaming paraformer
  std::vector<float> &GetParaformerFeatCache();
  std::vector<float> &GetParaformerEncoderOutCache();
  std::vector<float> &GetParaformerAlphaCache();

  // Generic per-stream option mechanism (key-value string pairs).
  void SetOption(const std::string &key, const std::string &value);
  bool HasOption(const std::string &key) const;

  // Returns the value for the given key, or an empty string if the key
  // does not exist. No exception is thrown for missing keys.
  const std::string &GetOption(const std::string &key) const;
  int32_t GetOptionInt(const std::string &key, int32_t default_value = 0) const;
  float GetOptionFloat(const std::string &key,
                       float default_value = 0.0f) const;

  // ---- Independent, per-stream Silero VAD (optional) ----
  //
  // Unrelated to the ASR feature pipeline: this VAD has its own ONNX
  // session, its own resampling of the waveform passed to
  // AcceptWaveform(), and its own speech/silence state. Call once, right
  // after creating the stream, only if you want VAD-driven endpointing.

  // Creates the online VAD for this stream from `vad_config`
  // (`vad_config.silero_vad.model` must be non-empty). `asr_frame_shift_ms`
  // is the ASR feature pipeline's frame shift, used to convert VAD-scored
  // sample counts into ASR-frame-equivalent units for the coverage gate
  // and batch-skip. `defer_inference`, when true, makes AcceptWaveform()
  // only buffer samples for the VAD -- actual Silero inference only runs
  // when something calls RunOnlineVad() explicitly (e.g. a dedicated
  // timer). When false (the default for every caller except the
  // websocket server), AcceptWaveform() runs VAD inference inline, same
  // as before this flag existed.
  void InitOnlineVad(const VadModelConfig &vad_config, float asr_frame_shift_ms,
                     bool defer_inference = false);

  // True if InitOnlineVad() was called on this stream.
  bool HasOnlineVad() const;

  // True if the online VAD currently considers this stream to contain
  // speech (sticky through its own min-silence-duration hangover).
  bool IsOnlineVadSpeechDetected() const;

  // True once this stream has seen speech and the online VAD has since
  // observed enough trailing silence (its min_silence_duration) to end
  // it. This is the VAD-driven endpoint signal. Cleared by
  // ResetOnlineVad().
  bool IsOnlineVadEndpoint() const;

  // Clears the "seen speech this segment" bookkeeping so a new endpoint
  // requires fresh speech first. Does not drop the VAD's own internal
  // buffering/resampler state, which keeps running continuously across
  // segments (mirrors how Reset() does not reset the feature extractor).
  void ResetOnlineVad();

  // When InitOnlineVad(..., defer_inference=true) was used, actually runs
  // Silero inference on whatever samples AcceptWaveform() has buffered
  // since the last call, updating the VAD's speech/silence state and
  // coverage counters. A no-op if nothing is pending, if deferred
  // inference wasn't requested (inline mode already did this inside
  // AcceptWaveform), or if no online VAD was configured.
  void RunOnlineVad();

  // Total ASR-feature-frame-equivalent audio the online VAD has actually
  // scored so far, in the same absolute frame space as
  // GetNumFramesSinceStart() + GetNumProcessedFrames() (i.e. never reset
  // by Reset()/ResetOnlineVad()). 0 if no online VAD is configured.
  int64_t GetOnlineVadFramesCoveredAbsolute() const;

  // True + the absolute frame (same space as above) where the VAD's
  // current confirmed-silence run began, if the VAD is presently silent
  // and a silence run has actually started this stream. False (output
  // untouched) otherwise.
  bool GetOnlineVadConfirmedSilenceStartAbsolute(int64_t *out_frame) const;

  // True once InputFinished() has been called on this stream.
  bool IsInputFinished() const;

  // Which endpoint rule last fired for this stream's current segment: 0
  // none, 1 blank-based (rule1/2/3, undistinguished), 4 VAD-based. Reset
  // to 0 by OnlineRecognizer::Reset(). Returned reference is valid as
  // long as this object is alive.
  int32_t &GetEndpointRule();

  // Seconds between the VAD-based endpoint firing and the VAD's last
  // scored speech frame; only meaningful when GetEndpointRule() == 4.
  float &GetEndpointDelay();

  // Stream time (seconds) at which the VAD last scored speech for the
  // current segment; only meaningful when GetEndpointRule() == 4.
  float &GetSpeechEndTime();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_STREAM_H_
