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
#include "sherpa-onnx/csrc/online-vad-buffer.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transducer-keyword-decoder.h"
#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

class OnlineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : feat_extractor_(config), context_graph_(std::move(context_graph)) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);

    if (!vad_model_) return;

    // Append a *copy* of the incoming waveform (resampled to the VAD's
    // own sample rate) to an independent buffer. No inference happens
    // here -- that only happens inside VadProcessStep(), called by
    // OnlineRecognizer::VadProcess() on its own schedule, never from
    // AcceptWaveform() itself. This is the key structural difference
    // from this port's other branch (feat/online-silero-vad-endpoint),
    // which scores inline-by-default inside AcceptWaveform.
    const float *vad_waveform = waveform;
    int32_t vad_n = n;
    std::vector<float> resampled;
    if (sampling_rate != vad_sample_rate_) {
      if (!vad_resampler_ ||
          vad_resampler_->GetInputSamplingRate() != sampling_rate) {
        float min_freq = std::min<int32_t>(sampling_rate, vad_sample_rate_);
        float lowpass_cutoff = 0.99f * 0.5f * min_freq;
        vad_resampler_ = std::make_unique<LinearResample>(
            sampling_rate, vad_sample_rate_, lowpass_cutoff, 6);
      }
      vad_resampler_->Resample(waveform, n, false, &resampled);
      vad_waveform = resampled.data();
      vad_n = static_cast<int32_t>(resampled.size());
    }
    if (vad_n > 0) {
      vad_audio_copy_.insert(vad_audio_copy_.end(), vad_waveform,
                             vad_waveform + vad_n);
    }
  }

  void InputFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_extractor_.InputFinished();
    input_finished_ = true;
    if (vad_model_) {
      // Score whatever's left (there usually won't be a full window's
      // worth, so this is often a no-op beyond the carry buffer).
      RunVadProcessStepLocked();
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

  void InitVad(const VadModelConfig &vad_config, float asr_frame_shift_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    vad_sample_rate_ = vad_config.sample_rate;
    vad_model_ = VadModel::Create(vad_config);
    asr_frame_shift_ms_ = asr_frame_shift_ms;
    vad_window_shift_seconds_ =
        vad_model_->WindowShift() / static_cast<float>(vad_sample_rate_);
  }

  bool HasVad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vad_model_ != nullptr;
  }

  // Assumes mutex_ is already held by the caller.
  // KNOWN UNRESOLVED ISSUE (found via load testing, not fixed): under
  // sustained concurrent load (observed at --concurrency=10, ~90+ calls
  // in via ds-e2e-asr-testing's engine client against the real websocket
  // server), the process eventually crashes with glibc's "corrupted
  // double-linked list" (SIGSEGV / exit 139) -- a heap-corruption
  // signature. Two targeted fixes were tried and did NOT resolve it:
  // (1) switching the window-stepping loop below from size_t/unsigned to
  // signed int32_t arithmetic (removes a possible underflow-driven OOB
  // read, but didn't stop the crash), and (2) bounding vad_predictions_'s
  // growth so CircularBuffer::Resize() isn't repeatedly called at
  // ever-larger sizes (see the Pop() call at the end of this function --
  // also didn't stop it). The crash was not reproduced on this port's
  // other branch (feat/online-silero-vad-endpoint) under the identical
  // load test, which reuses the existing VoiceActivityDetector instead
  // of calling VadModel::IsSpeech() directly here. Root cause not
  // isolated further (no debugger/ASAN available in the environment
  // this was developed in) -- flagging explicitly rather than silently
  // leaving a load-bearing crash undocumented. Do not treat this branch
  // as production-ready; it exists for architectural comparison with
  // feat/online-silero-vad-endpoint, not as a merge candidate as-is.
  void RunVadProcessStepLocked() {
    if (!vad_model_ || vad_audio_copy_.empty()) return;

    vad_window_carry_.insert(vad_window_carry_.end(), vad_audio_copy_.begin(),
                             vad_audio_copy_.end());
    vad_audio_copy_.clear();

    int32_t window_size = vad_model_->WindowSize();
    int32_t window_shift = vad_model_->WindowShift();
    // Defensive: never let a bad/zero/negative stride or a stride larger
    // than the window drive the loop below out of bounds. Should not
    // happen in practice (both come from the fixed, already-validated
    // VadModelConfig), but previously this loop used size_t (unsigned)
    // arithmetic for `carry.size() - pos`, which would silently
    // underflow to a huge value instead of going negative if pos ever
    // slipped past carry.size() -- turning one bad iteration into an
    // out-of-bounds read handed straight to the ONNX model and
    // corrupting the heap (observed as a "corrupted double-linked list"
    // crash under sustained load on the sibling ds-k2-sherpa-onnx-style
    // branch, which has this same loop). Signed arithmetic with an
    // explicit guard removes the whole class of risk regardless of
    // whether this exact path was the trigger.
    if (window_size <= 0 || window_shift <= 0 || window_shift > window_size) {
      return;
    }

    int32_t carry_size = static_cast<int32_t>(vad_window_carry_.size());
    int32_t pos = 0;
    while (carry_size - pos >= window_size) {
      bool is_speech =
          vad_model_->IsSpeech(vad_window_carry_.data() + pos, window_size);
      vad_predictions_.Push(is_speech ? 1.0f : 0.0f);

      if (is_speech) {
        vad_had_speech_ = true;
        vad_last_speech_frame_ = vad_predictions_.NumFramesReady();
      } else if (vad_is_speech_) {
        vad_silence_start_frame_ = vad_predictions_.NumFramesReady();
      }
      vad_is_speech_ = is_speech;

      pos += window_shift;
    }
    if (pos > 0) {
      vad_window_carry_.erase(vad_window_carry_.begin(),
                              vad_window_carry_.begin() + pos);
    }

    // Bound memory: only NumTrailingSilence()'s small lookback (a few
    // hundred ms worth of windows) and the most recent prediction are
    // ever read: GetVadFramesCoveredAbsolute()/GetVadConfirmedSilence-
    // StartAbsolute() only need NumFramesReady() (a monotonic counter,
    // Pop() doesn't touch it -- same absolute-indexing convention as
    // FeatureExtractor's own fbank ring, see features.cc). Without this,
    // a long-running stream accumulates one prediction per window for
    // its *entire* lifetime, forcing CircularBuffer::Resize() to ever-
    // larger allocations and was the likely trigger for a heap
    // corruption crash found under sustained load testing (never popping
    // is a pattern this port's other branch, feat/online-silero-vad-
    // endpoint, doesn't have -- it tracks scalar counters, not a
    // growing buffer, which is consistent with only this branch
    // crashing in that test).
    constexpr int32_t kMaxRetainedPredictions = 1000;  // ~32s @ 32ms/window
    constexpr int32_t kTrimToPredictions = 200;        // ~6.4s, generous
    if (vad_predictions_.Size() > kMaxRetainedPredictions) {
      vad_predictions_.Pop(vad_predictions_.Size() - kTrimToPredictions);
    }
  }

  void VadProcessStep() {
    std::lock_guard<std::mutex> lock(mutex_);
    RunVadProcessStepLocked();
  }

  int64_t GetVadFramesCoveredAbsolute() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!vad_model_ || asr_frame_shift_ms_ <= 0.0f) return 0;
    float seconds_covered =
        vad_predictions_.NumFramesReady() * vad_window_shift_seconds_;
    return static_cast<int64_t>(seconds_covered * 1000.0f /
                                asr_frame_shift_ms_);
  }

  bool GetVadConfirmedSilenceStartAbsolute(int64_t *out_frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!vad_model_ || vad_is_speech_ || vad_silence_start_frame_ < 0 ||
        asr_frame_shift_ms_ <= 0.0f) {
      return false;
    }
    float seconds = vad_silence_start_frame_ * vad_window_shift_seconds_;
    *out_frame = static_cast<int64_t>(seconds * 1000.0f / asr_frame_shift_ms_);
    return true;
  }

  bool IsVadSpeechDetected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vad_model_ && vad_is_speech_;
  }

  bool IsVadEndpoint() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vad_model_ && vad_had_speech_ && !vad_is_speech_;
  }

  void ResetVad() {
    std::lock_guard<std::mutex> lock(mutex_);
    vad_had_speech_ = false;
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

  // Independent VAD pipeline (ds-k2-sherpa-onnx refactor_vad style;
  // optional, null unless InitVad() was called).
  std::unique_ptr<VadModel> vad_model_;
  std::unique_ptr<LinearResample> vad_resampler_;
  int32_t vad_sample_rate_ = 16000;

  // "Copy of audios": raw samples waiting for the next VadProcessStep(),
  // fed only from AcceptWaveform(); never read by the ASR path.
  std::vector<float> vad_audio_copy_;
  std::vector<float> vad_window_carry_;  // leftover sub-window remainder

  OnlineVadBuffer vad_predictions_;  // one entry per Silero window
  bool vad_had_speech_ = false;
  bool vad_is_speech_ = false;
  int64_t vad_silence_start_frame_ = -1;  // in vad_predictions_ frame units
  int64_t vad_last_speech_frame_ = -1;
  float vad_window_shift_seconds_ = 0.032f;

  float asr_frame_shift_ms_ = 10.0f;
  bool input_finished_ = false;

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

void OnlineStream::InitVad(const VadModelConfig &vad_config,
                           float asr_frame_shift_ms) {
  impl_->InitVad(vad_config, asr_frame_shift_ms);
}

bool OnlineStream::HasVad() const { return impl_->HasVad(); }

void OnlineStream::VadProcessStep() { impl_->VadProcessStep(); }

int64_t OnlineStream::GetVadFramesCoveredAbsolute() const {
  return impl_->GetVadFramesCoveredAbsolute();
}

bool OnlineStream::GetVadConfirmedSilenceStartAbsolute(
    int64_t *out_frame) const {
  return impl_->GetVadConfirmedSilenceStartAbsolute(out_frame);
}

bool OnlineStream::IsVadSpeechDetected() const {
  return impl_->IsVadSpeechDetected();
}

bool OnlineStream::IsVadEndpoint() const { return impl_->IsVadEndpoint(); }

void OnlineStream::ResetVad() { impl_->ResetVad(); }

bool OnlineStream::IsInputFinished() const {
  return impl_->IsInputFinished();
}

int32_t &OnlineStream::GetEndpointRule() { return impl_->GetEndpointRule(); }

float &OnlineStream::GetEndpointDelay() { return impl_->GetEndpointDelay(); }

float &OnlineStream::GetSpeechEndTime() { return impl_->GetSpeechEndTime(); }

}  // namespace sherpa_onnx
