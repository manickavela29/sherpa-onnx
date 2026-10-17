// sherpa-onnx/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-stream.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transducer-keyword-decoder.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

namespace sherpa_onnx {

class OnlineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : feat_extractor_(config), context_graph_(std::move(context_graph)) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);

    if (online_vad_) {
      const float *vad_waveform = waveform;
      int32_t vad_n = n;
      std::vector<float> resampled;

      if (sampling_rate != online_vad_sample_rate_) {
        if (!online_vad_resampler_ ||
            online_vad_resampler_->GetInputSamplingRate() != sampling_rate) {
          float min_freq =
              std::min<int32_t>(sampling_rate, online_vad_sample_rate_);
          float lowpass_cutoff = 0.99f * 0.5f * min_freq;
          online_vad_resampler_ = std::make_unique<LinearResample>(
              sampling_rate, online_vad_sample_rate_, lowpass_cutoff, 6);
        }
        online_vad_resampler_->Resample(waveform, n, false, &resampled);
        vad_waveform = resampled.data();
        vad_n = static_cast<int32_t>(resampled.size());
      }

      if (vad_n > 0) {
        if (defer_online_vad_inference_) {
          online_vad_pending_samples_.insert(online_vad_pending_samples_.end(),
                                             vad_waveform,
                                             vad_waveform + vad_n);
        } else {
          ScoreOnlineVad(vad_waveform, vad_n);
        }
      }
    }
  }

  void InputFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_extractor_.InputFinished();
    input_finished_ = true;
    if (online_vad_) {
      DrainOnlineVadLocked();
      online_vad_->Flush();
    }
  }

  int32_t NumFramesReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feat_extractor_.IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    // we don't reset the feature extractor
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  void InitOnlineVad(const VadModelConfig &vad_config, float asr_frame_shift_ms,
                     bool defer_inference) {
    std::lock_guard<std::mutex> lock(mutex_);
    online_vad_sample_rate_ = vad_config.sample_rate;
    online_vad_ = std::make_unique<VoiceActivityDetector>(vad_config);
    asr_frame_shift_ms_ = asr_frame_shift_ms;
    defer_online_vad_inference_ = defer_inference;
  }

  bool HasOnlineVad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return online_vad_ != nullptr;
  }

  bool IsOnlineVadSpeechDetected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return online_vad_ && online_vad_->IsSpeechDetected();
  }

  bool IsOnlineVadEndpoint() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return online_vad_ && online_vad_had_speech_ &&
           !online_vad_->IsSpeechDetected();
  }

  void ResetOnlineVad() {
    std::lock_guard<std::mutex> lock(mutex_);
    online_vad_had_speech_ = false;
  }

  // Assumes mutex_ is already held by the caller. Runs Silero inference on
  // `data`/`n` and updates the coverage/speech-state counters.
  void ScoreOnlineVad(const float *data, int32_t n) {
    bool was_speech = online_vad_->IsSpeechDetected();
    online_vad_->AcceptWaveform(data, n);
    online_vad_samples_scored_ += n;
    if (online_vad_->IsSpeechDetected()) {
      online_vad_had_speech_ = true;
      online_vad_last_speech_sample_ = online_vad_samples_scored_;
    } else if (was_speech) {
      online_vad_silence_start_sample_ = online_vad_samples_scored_;
    }
  }

  // Assumes mutex_ is already held by the caller.
  void DrainOnlineVadLocked() {
    if (online_vad_pending_samples_.empty()) return;
    std::vector<float> pending;
    std::swap(pending, online_vad_pending_samples_);
    ScoreOnlineVad(pending.data(), static_cast<int32_t>(pending.size()));
  }

  void RunOnlineVad() {
    std::vector<float> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!online_vad_ || online_vad_pending_samples_.empty()) return;
      std::swap(pending, online_vad_pending_samples_);
    }

    // Runs the actual Silero inference without mutex_ held, so
    // AcceptWaveform() on another thread can keep buffering new samples
    // concurrently. online_vad_ itself has no internal locking, so this
    // relies on RunOnlineVad() never being called concurrently with
    // itself for the same stream (the caller -- a single dedicated VAD
    // timer thread -- is responsible for that).
    bool was_speech = online_vad_->IsSpeechDetected();
    online_vad_->AcceptWaveform(pending.data(),
                                static_cast<int32_t>(pending.size()));

    std::lock_guard<std::mutex> lock(mutex_);
    online_vad_samples_scored_ += static_cast<int64_t>(pending.size());
    if (online_vad_->IsSpeechDetected()) {
      online_vad_had_speech_ = true;
      online_vad_last_speech_sample_ = online_vad_samples_scored_;
    } else if (was_speech) {
      online_vad_silence_start_sample_ = online_vad_samples_scored_;
    }
  }

  int64_t GetOnlineVadFramesCoveredAbsolute() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!online_vad_ || online_vad_sample_rate_ <= 0 ||
        asr_frame_shift_ms_ <= 0.0f) {
      return 0;
    }
    double frame_shift_samples =
        online_vad_sample_rate_ * asr_frame_shift_ms_ / 1000.0;
    return static_cast<int64_t>(online_vad_samples_scored_ /
                                frame_shift_samples);
  }

  bool GetOnlineVadConfirmedSilenceStartAbsolute(int64_t *out_frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!online_vad_ || online_vad_->IsSpeechDetected() ||
        online_vad_silence_start_sample_ < 0 || asr_frame_shift_ms_ <= 0.0f) {
      return false;
    }
    double frame_shift_samples =
        online_vad_sample_rate_ * asr_frame_shift_ms_ / 1000.0;
    *out_frame = static_cast<int64_t>(online_vad_silence_start_sample_ /
                                      frame_shift_samples);
    return true;
  }

  bool IsInputFinished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return input_finished_;
  }

  int32_t &GetEndpointRule() {
    std::lock_guard<std::mutex> lock(mutex_);
    return endpoint_rule_;
  }

  float &GetEndpointDelay() {
    std::lock_guard<std::mutex> lock(mutex_);
    return endpoint_delay_;
  }

  float &GetSpeechEndTime() {
    std::lock_guard<std::mutex> lock(mutex_);
    return speech_end_time_;
  }

  int32_t &GetNumProcessedFrames() {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_processed_frames_;
  }

  int32_t GetNumFramesSinceStart() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return start_frame_index_;
  }

  int32_t &GetCurrentSegment() {
    std::lock_guard<std::mutex> lock(mutex_);
    return segment_;
  }

  void SetResult(const OnlineTransducerDecoderResult &r) { result_ = r; }

  OnlineTransducerDecoderResult &GetResult() { return result_; }

  void SetKeywordResult(const TransducerKeywordResult &r) {
    keyword_result_ = r;
  }
  TransducerKeywordResult &GetKeywordResult(bool remove_duplicates) {
    if (remove_duplicates) {
      if (!prev_keyword_result_.timestamps.empty() &&
          !keyword_result_.timestamps.empty() &&
          keyword_result_.timestamps[0] <=
              prev_keyword_result_.timestamps.back()) {
        return empty_keyword_result_;
      } else {
        prev_keyword_result_ = keyword_result_;
      }
      return keyword_result_;
    } else {
      return keyword_result_;
    }
  }

  OnlineCtcDecoderResult &GetCtcResult() { return ctc_result_; }

  void SetCtcResult(const OnlineCtcDecoderResult &r) { ctc_result_ = r; }

  void SetParaformerResult(const OnlineParaformerDecoderResult &r) {
    paraformer_result_ = r;
  }

  OnlineParaformerDecoderResult &GetParaformerResult() {
    return paraformer_result_;
  }

  int32_t FeatureDim() const { return feat_extractor_.FeatureDim(); }

  void SetStates(std::vector<Ort::Value> states) {
    states_ = std::move(states);
  }

  std::vector<Ort::Value> &GetStates() { return states_; }

  void SetQnnStates(std::vector<OnlineStreamStateTensor> states) {
    qnn_states_ = std::move(states);
  }

  std::vector<OnlineStreamStateTensor> &GetQnnStates() { return qnn_states_; }

  void SetQnnResult(const OnlineTransducerDecoderResultNoOrt &r) {
    qnn_result_ = r;
  }

  OnlineTransducerDecoderResultNoOrt &GetQnnResult() { return qnn_result_; }

  void SetNeMoDecoderStates(std::vector<Ort::Value> decoder_states) {
    nemo_decoder_states_ = std::move(decoder_states);
  }

  void SetNeMoDecoderOut(Ort::Value decoder_out) {
    nemo_decoder_out_ = std::move(decoder_out);
  }

  std::vector<Ort::Value> &GetNeMoDecoderStates() {
    return nemo_decoder_states_;
  }

  Ort::Value &GetNeMoDecoderOut() { return nemo_decoder_out_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

  std::vector<float> &GetParaformerFeatCache() {
    return paraformer_feat_cache_;
  }

  std::vector<float> &GetParaformerEncoderOutCache() {
    return paraformer_encoder_out_cache_;
  }

  std::vector<float> &GetParaformerAlphaCache() {
    return paraformer_alpha_cache_;
  }

  void SetOption(const std::string &key, const std::string &value) {
    options_[key] = value;
  }

  bool HasOption(const std::string &key) const {
    return options_.count(key) != 0;
  }

  const std::string &GetOption(const std::string &key) const {
    auto it = options_.find(key);
    if (it != options_.end()) {
      return it->second;
    }
    static const std::string kEmpty;
    return kEmpty;
  }

  int32_t GetOptionInt(const std::string &key, int32_t default_value) const {
    auto it = options_.find(key);
    if (it != options_.end()) {
      return ToIntOrDefault(it->second, default_value);
    }
    return default_value;
  }

  float GetOptionFloat(const std::string &key, float default_value) const {
    auto it = options_.find(key);
    if (it != options_.end()) {
      return ToFloatOrDefault(it->second, default_value);
    }
    return default_value;
  }

  void SetFasterDecoder(std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
    faster_decoder_ = std::move(decoder);
  }

  kaldi_decoder::FasterDecoder *GetFasterDecoder() const {
    return faster_decoder_.get();
  }

  int32_t &GetFasterDecoderProcessedFrames() {
    return faster_decoder_processed_frames_;
  }

 private:
  FeatureExtractor feat_extractor_;
  mutable std::mutex mutex_;
  /// For contextual-biasing
  ContextGraphPtr context_graph_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;     // never reset
  int32_t segment_ = 0;
  OnlineTransducerDecoderResult result_;
  TransducerKeywordResult prev_keyword_result_;
  TransducerKeywordResult keyword_result_;
  TransducerKeywordResult empty_keyword_result_;
  OnlineCtcDecoderResult ctc_result_;
  std::vector<Ort::Value> states_;  // states for transducer or ctc models
  std::vector<OnlineStreamStateTensor> qnn_states_;

  // states for nemo transducer models
  std::vector<Ort::Value> nemo_decoder_states_;

  // decoder out for nemo transducer models
  Ort::Value nemo_decoder_out_{nullptr};

  std::vector<float> paraformer_feat_cache_;
  std::vector<float> paraformer_encoder_out_cache_;
  std::vector<float> paraformer_alpha_cache_;
  OnlineParaformerDecoderResult paraformer_result_;
  OnlineTransducerDecoderResultNoOrt qnn_result_;
  std::unordered_map<std::string, std::string> options_;
  std::unique_ptr<kaldi_decoder::FasterDecoder> faster_decoder_;
  int32_t faster_decoder_processed_frames_ = 0;

  // Independent, per-stream Silero VAD (optional; null unless
  // InitOnlineVad() was called).
  std::unique_ptr<VoiceActivityDetector> online_vad_;
  std::unique_ptr<LinearResample> online_vad_resampler_;
  int32_t online_vad_sample_rate_ = 16000;
  bool online_vad_had_speech_ = false;

  // When true, AcceptWaveform() only buffers samples into
  // online_vad_pending_samples_; actual Silero inference happens only
  // when RunOnlineVad() is called. When false (the default), inference
  // runs inline inside AcceptWaveform(), as before this flag existed.
  bool defer_online_vad_inference_ = false;
  std::vector<float> online_vad_pending_samples_;

  // All three counters below are in online_vad_'s own sample-rate domain
  // and are never reset by Reset()/ResetOnlineVad() -- only by stream
  // construction -- so they stay comparable, across segments, to
  // GetNumFramesSinceStart() + GetNumProcessedFrames() (itself monotonic
  // across Reset()).
  int64_t online_vad_samples_scored_ = 0;
  int64_t online_vad_silence_start_sample_ = -1;
  int64_t online_vad_last_speech_sample_ = -1;

  // ASR feature pipeline's frame shift, used to convert the VAD sample
  // counters above into ASR-frame-equivalent units.
  float asr_frame_shift_ms_ = 10.0f;

  bool input_finished_ = false;

  // Which endpoint rule last fired: 0 none, 1 blank-based, 4 VAD-based.
  // Set by OnlineRecognizer::IsEndpoint(), cleared by
  // OnlineRecognizer::Reset().
  int32_t endpoint_rule_ = 0;
  float endpoint_delay_ = 0.0f;
  float speech_end_time_ = 0.0f;
};


OnlineStream::OnlineStream(const FeatureExtractorConfig &config /*= {}*/,
                           ContextGraphPtr context_graph /*= nullptr */)
    : impl_(std::make_unique<Impl>(config, std::move(context_graph))) {}

OnlineStream::~OnlineStream() = default;

void OnlineStream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                  int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void OnlineStream::InputFinished() const { impl_->InputFinished(); }

int32_t OnlineStream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool OnlineStream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> OnlineStream::GetFrames(int32_t frame_index,
                                           int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void OnlineStream::Reset() { impl_->Reset(); }

int32_t OnlineStream::FeatureDim() const { return impl_->FeatureDim(); }

int32_t &OnlineStream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

int32_t OnlineStream::GetNumFramesSinceStart() const {
  return impl_->GetNumFramesSinceStart();
}

int32_t &OnlineStream::GetCurrentSegment() {
  return impl_->GetCurrentSegment();
}

void OnlineStream::SetResult(const OnlineTransducerDecoderResult &r) {
  impl_->SetResult(r);
}

OnlineTransducerDecoderResult &OnlineStream::GetResult() {
  return impl_->GetResult();
}

void OnlineStream::SetKeywordResult(const TransducerKeywordResult &r) {
  impl_->SetKeywordResult(r);
}

TransducerKeywordResult &OnlineStream::GetKeywordResult(
    bool remove_duplicates /*=false*/) {
  return impl_->GetKeywordResult(remove_duplicates);
}

OnlineCtcDecoderResult &OnlineStream::GetCtcResult() {
  return impl_->GetCtcResult();
}

void OnlineStream::SetCtcResult(const OnlineCtcDecoderResult &r) {
  impl_->SetCtcResult(r);
}

void OnlineStream::SetParaformerResult(const OnlineParaformerDecoderResult &r) {
  impl_->SetParaformerResult(r);
}

OnlineParaformerDecoderResult &OnlineStream::GetParaformerResult() {
  return impl_->GetParaformerResult();
}

void OnlineStream::SetStates(std::vector<Ort::Value> states) {
  impl_->SetStates(std::move(states));
}

std::vector<Ort::Value> &OnlineStream::GetStates() {
  return impl_->GetStates();
}

void OnlineStream::SetQnnStates(std::vector<OnlineStreamStateTensor> states) {
  impl_->SetQnnStates(std::move(states));
}

std::vector<OnlineStreamStateTensor> &OnlineStream::GetQnnStates() {
  return impl_->GetQnnStates();
}

void OnlineStream::SetQnnResult(const OnlineTransducerDecoderResultNoOrt &r) {
  impl_->SetQnnResult(r);
}

OnlineTransducerDecoderResultNoOrt &OnlineStream::GetQnnResult() {
  return impl_->GetQnnResult();
}

void OnlineStream::SetNeMoDecoderStates(
    std::vector<Ort::Value> decoder_states) {
  return impl_->SetNeMoDecoderStates(std::move(decoder_states));
}

void OnlineStream::SetNeMoDecoderOut(Ort::Value decoder_out) {
  return impl_->SetNeMoDecoderOut(std::move(decoder_out));
}

std::vector<Ort::Value> &OnlineStream::GetNeMoDecoderStates() {
  return impl_->GetNeMoDecoderStates();
}

Ort::Value &OnlineStream::GetNeMoDecoderOut() {
  return impl_->GetNeMoDecoderOut();
}

const ContextGraphPtr &OnlineStream::GetContextGraph() const {
  return impl_->GetContextGraph();
}

void OnlineStream::SetFasterDecoder(
    std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
  impl_->SetFasterDecoder(std::move(decoder));
}

kaldi_decoder::FasterDecoder *OnlineStream::GetFasterDecoder() const {
  return impl_->GetFasterDecoder();
}

int32_t &OnlineStream::GetFasterDecoderProcessedFrames() {
  return impl_->GetFasterDecoderProcessedFrames();
}

std::vector<float> &OnlineStream::GetParaformerFeatCache() {
  return impl_->GetParaformerFeatCache();
}

std::vector<float> &OnlineStream::GetParaformerEncoderOutCache() {
  return impl_->GetParaformerEncoderOutCache();
}

std::vector<float> &OnlineStream::GetParaformerAlphaCache() {
  return impl_->GetParaformerAlphaCache();
}

void OnlineStream::SetOption(const std::string &key, const std::string &value) {
  impl_->SetOption(key, value);
}

bool OnlineStream::HasOption(const std::string &key) const {
  return impl_->HasOption(key);
}

const std::string &OnlineStream::GetOption(const std::string &key) const {
  return impl_->GetOption(key);
}

int32_t OnlineStream::GetOptionInt(const std::string &key,
                                   int32_t default_value) const {
  return impl_->GetOptionInt(key, default_value);
}

float OnlineStream::GetOptionFloat(const std::string &key,
                                   float default_value) const {
  return impl_->GetOptionFloat(key, default_value);
}

void OnlineStream::InitOnlineVad(const VadModelConfig &vad_config,
                                 float asr_frame_shift_ms,
                                 bool defer_inference /*= false*/) {
  impl_->InitOnlineVad(vad_config, asr_frame_shift_ms, defer_inference);
}

bool OnlineStream::HasOnlineVad() const { return impl_->HasOnlineVad(); }

bool OnlineStream::IsOnlineVadSpeechDetected() const {
  return impl_->IsOnlineVadSpeechDetected();
}

bool OnlineStream::IsOnlineVadEndpoint() const {
  return impl_->IsOnlineVadEndpoint();
}

void OnlineStream::ResetOnlineVad() { impl_->ResetOnlineVad(); }

void OnlineStream::RunOnlineVad() { impl_->RunOnlineVad(); }

int64_t OnlineStream::GetOnlineVadFramesCoveredAbsolute() const {
  return impl_->GetOnlineVadFramesCoveredAbsolute();
}

bool OnlineStream::GetOnlineVadConfirmedSilenceStartAbsolute(
    int64_t *out_frame) const {
  return impl_->GetOnlineVadConfirmedSilenceStartAbsolute(out_frame);
}

bool OnlineStream::IsInputFinished() const {
  return impl_->IsInputFinished();
}

int32_t &OnlineStream::GetEndpointRule() { return impl_->GetEndpointRule(); }

float &OnlineStream::GetEndpointDelay() { return impl_->GetEndpointDelay(); }

float &OnlineStream::GetSpeechEndTime() { return impl_->GetSpeechEndTime(); }

}  // namespace sherpa_onnx
