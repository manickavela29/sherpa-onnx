// sherpa-onnx/csrc/online-websocket-server-impl.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-websocket-server-impl.h"
#include "sherpa-onnx/csrc/macros.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/online-endpoint-profile.h"

namespace sherpa_onnx {

void OnlineWebsocketDecoderConfig::Register(ParseOptions *po) {
  recognizer_config.Register(po);

  po->Register("loop-interval-ms", &loop_interval_ms,
               "It determines how often the decoder loop runs. ");

  po->Register("max-batch-size", &max_batch_size,
               "Max batch size for recognition.");

  po->Register("end-tail-padding", &end_tail_padding,
               "It determines the length of tail_padding at the end of audio.");

  po->Register("input-sample-rate", &input_sample_rate,
               "The sample rate of the input audio from the client. "
               "If it differs from the model's expected sample rate, "
               "the server will resample internally.");

  po->Register("vad-loop-interval-ms", &vad_loop_interval_ms,
               "How often the independent VAD tick (VadProcess) runs. "
               "Only meaningful when --online-vad-model is set.");

  po->Register("max-vad-batch-size", &max_vad_batch_size,
               "Max batch size for one VadProcess() call.");
}

void OnlineWebsocketDecoderConfig::Validate() const {
  recognizer_config.Validate();
  SHERPA_ONNX_CHECK_GT(loop_interval_ms, 0);
  SHERPA_ONNX_CHECK_GT(max_batch_size, 0);
  SHERPA_ONNX_CHECK_GT(end_tail_padding, 0);
  SHERPA_ONNX_CHECK_GT(input_sample_rate, 0);
  SHERPA_ONNX_CHECK_GT(vad_loop_interval_ms, 0);
  SHERPA_ONNX_CHECK_GT(max_vad_batch_size, 0);
}

void OnlineWebsocketServerConfig::Register(sherpa_onnx::ParseOptions *po) {
  decoder_config.Register(po);

  po->Register("log-file", &log_file,
               "Path to the log file. Logs are "
               "appended to this file");
}

void OnlineWebsocketServerConfig::Validate() const {
  decoder_config.Validate();
}

OnlineWebsocketDecoder::OnlineWebsocketDecoder(OnlineWebsocketServer *server)
    : server_(server),
      config_(server->GetConfig().decoder_config),
      timer_(server->GetWorkContext()),
      vad_timer_(server->GetVadContext()) {
  recognizer_ = std::make_unique<OnlineRecognizer>(config_.recognizer_config);
}

std::shared_ptr<Connection> OnlineWebsocketDecoder::GetOrCreateConnection(
    connection_hdl hdl) {
  const int64_t prof_t0 = epprof::Enabled() ? epprof::NowNs() : 0;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = connections_.find(hdl);
  if (it != connections_.end()) {
    return it->second;
  } else {
    // create a new connection
    const int64_t prof_t1 = epprof::Enabled() ? epprof::NowNs() : 0;
    std::shared_ptr<OnlineStream> s = recognizer_->CreateStream();
    if (epprof::Enabled()) {
      epprof::Log(epprof::Ev::kStreamCreate, s->Id(), prof_t1 - prof_t0,
                  epprof::NowNs() - prof_t1);
    }
    auto c = std::make_shared<Connection>(hdl, s);
    connections_.insert({hdl, c});
    return c;
  }
}

void OnlineWebsocketDecoder::AcceptWaveform(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  float sample_rate = config_.input_sample_rate;
  while (!c->samples.empty()) {
    const auto &s = c->samples.front();
    c->s->AcceptWaveform(sample_rate, s.data(), s.size());
    c->samples.pop_front();
  }
}

void OnlineWebsocketDecoder::InputFinished(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);

  float sample_rate = config_.input_sample_rate;

  while (!c->samples.empty()) {
    const auto &s = c->samples.front();
    c->s->AcceptWaveform(sample_rate, s.data(), s.size());
    c->samples.pop_front();
  }

  std::vector<float> tail_padding(
      static_cast<int64_t>(config_.end_tail_padding * sample_rate));

  epprof::Log(epprof::Ev::kInputFinished, c->s->Id(),
              static_cast<int64_t>(tail_padding.size()));

  c->s->AcceptWaveform(sample_rate, tail_padding.data(), tail_padding.size());

  c->s->InputFinished();
  c->eof = true;
}

void OnlineWebsocketDecoder::Warmup() const {
  recognizer_->WarmpUpRecognizer(config_.recognizer_config.model_config.warm_up,
                                 config_.max_batch_size);
}

void OnlineWebsocketDecoder::Run() {
  timer_.expires_after(std::chrono::milliseconds(config_.loop_interval_ms));

  timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessConnections(ec); });
}

void OnlineWebsocketDecoder::RunVad() {
  if (!config_.recognizer_config.online_vad_config.IsEnabled()) {
    // No online VAD configured; never arm vad_timer_ at all.
    return;
  }

  vad_timer_.expires_after(
      std::chrono::milliseconds(config_.vad_loop_interval_ms));

  vad_timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessVadConnections(ec); });
}

void OnlineWebsocketDecoder::ProcessConnections(const asio::error_code &ec) {
  if (ec) {
    SHERPA_ONNX_LOG(FATAL) << "The decoder loop is aborted!";
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (epprof::Enabled()) {
    static int64_t last_tick_ns = 0;  // guarded by mutex_
    int64_t now = epprof::NowNs();
    epprof::Log(epprof::Ev::kAsrTick, 0,
                last_tick_ns ? (now - last_tick_ns) / 1000000 : 0,
                static_cast<int64_t>(ready_connections_.size()),
                static_cast<int64_t>(connections_.size()));
    last_tick_ns = now;
  }
  std::vector<connection_hdl> to_remove;
  for (auto &p : connections_) {
    auto hdl = p.first;
    auto c = p.second;

    // The order of `if` below matters!
    if (!server_->Contains(hdl)) {
      // If the connection is disconnected, we stop processing it
      to_remove.push_back(hdl);
      continue;
    }

    if (active_.count(hdl)) {
      // Another thread is decoding this stream, so skip it
      continue;
    }

    // IsReady() can advance the stream's processed frames (silence skip), so
    // evaluate it once: a second call could flip to false after this
    // stream was already accepted.
    const bool ready = recognizer_->IsReady(c->s.get());
    if (!ready && !c->eof) {
      // this stream has not enough frames to decode, so skip it
      continue;
    }

    if (!ready && c->eof) {
      // We won't receive samples from the client, so send a Done! to client

      asio::post(server_->GetWorkContext(),
                 [this, hdl = c->hdl, sid = c->s->Id()]() {
                   server_->Send(hdl, "Done!");
                   epprof::Log(epprof::Ev::kDoneSend, sid);
                 });

      to_remove.push_back(hdl);
      continue;
    }

    // TODO(fangun): If the connection is timed out, we need to also
    // add it to `to_remove`

    // this stream has enough frames and is currently not processed by any
    // threads, so put it into the ready queue
    ready_connections_.push_back(c);
    epprof::Log(epprof::Ev::kReadyQueued, c->s->Id(),
                static_cast<int64_t>(ready_connections_.size()));

    // In `Decode()`, it will remove hdl from `active_`
    active_.insert(c->hdl);
  }

  for (auto hdl : to_remove) {
    connections_.erase(hdl);
  }

  if (!ready_connections_.empty()) {
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }

  // Schedule another call
  timer_.expires_after(std::chrono::milliseconds(config_.loop_interval_ms));

  timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessConnections(ec); });
}

void OnlineWebsocketDecoder::Decode() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (ready_connections_.empty()) {
    // There are no connections that are ready for decoding,
    // so we return directly
    return;
  }

  std::vector<std::shared_ptr<Connection>> c_vec;
  std::vector<OnlineStream *> s_vec;
  while (!ready_connections_.empty() &&
         static_cast<int32_t>(s_vec.size()) < config_.max_batch_size) {
    auto c = ready_connections_.front();
    ready_connections_.pop_front();

    if (!recognizer_->IsDecodable(c->s.get())) {
      // Never hand DecodeStreams() a stream without a full chunk; it would
      // abort in GetFrames(). The next ASR tick re-evaluates it.
      SHERPA_ONNX_LOGE("Stream %d queued without enough frames, deferred",
                       static_cast<int>(c->s->Id()));
      active_.erase(c->hdl);
      continue;
    }

    c_vec.push_back(c);
    s_vec.push_back(c->s.get());
  }

  if (s_vec.empty()) return;

  if (!ready_connections_.empty()) {
    // there are too many ready connections but this thread can only handle
    // max_batch_size connections at a time, so we schedule another call
    // to Decode() and let other threads to process the ready connections
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }

  const bool prof = epprof::Enabled();
  const int64_t prof_t0 = prof ? epprof::NowNs() : 0;
  const int64_t prof_remaining = static_cast<int64_t>(ready_connections_.size());
  lock.unlock();
  recognizer_->DecodeStreams(s_vec.data(), s_vec.size());
  lock.lock();
  if (prof) {
    epprof::Log(epprof::Ev::kDecodeBatch, 0,
                static_cast<int64_t>(s_vec.size()), prof_remaining,
                epprof::NowNs() - prof_t0);
  }

  for (auto c : c_vec) {
    // IsEndpoint() must run before GetResult(): it's what sets the
    // stream's endpoint_rule/endpoint_delay/speech_end_time fields when
    // it fires, and GetResult() is what copies them onto the returned
    // result.
    bool is_endpoint = recognizer_->IsEndpoint(c->s.get());
    auto result = recognizer_->GetResult(c->s.get());
    if (prof) {
      int64_t bits = (c->s->HasVad() ? 1 : 0) |
                     (c->s->IsVadSpeechDetected() ? 2 : 0) |
                     (c->s->HasVad() && c->s->IsVadEndpoint() ? 4 : 0);
      epprof::Log(epprof::Ev::kEndpointEval, c->s->Id(), is_endpoint ? 1 : 0,
                  result.endpoint_rule, bits);
    }
    if (is_endpoint) {
      result.is_final = true;
      recognizer_->Reset(c->s.get());
    }

    if (!recognizer_->IsReady(c->s.get()) && c->eof) {
      result.is_final = true;
      result.is_eof = true;
    }

    epprof::Log(epprof::Ev::kResultPost, c->s->Id(), result.is_final ? 1 : 0,
                result.endpoint_rule,
                static_cast<int64_t>(result.endpoint_delay * 1000));
    asio::post(server_->GetConnectionContext(),
               [this, hdl = c->hdl, sid = c->s->Id(),
                is_final = result.is_final, str = result.AsJsonString()]() {
                 server_->Send(hdl, str);
                 epprof::Log(epprof::Ev::kWsSend, sid, is_final ? 1 : 0);
               });
    active_.erase(c->hdl);
  }
}

void OnlineWebsocketDecoder::ScheduleVadFlushes(
    const std::vector<std::shared_ptr<Connection>> &cs) {
  // Called from VadDecode() without vad_mutex_ held.
  std::lock_guard<std::mutex> lock(mutex_);
  bool any = false;
  for (const auto &c : cs) {
    if (!c->s->IsVadFlushPending()) continue;
    if (!connections_.count(c->hdl) || active_.count(c->hdl)) {
      // gone, or being decoded right now: the ASR tick will pick it up
      continue;
    }
    if (!recognizer_->IsReady(c->s.get())) continue;
    ready_connections_.push_back(c);
    epprof::Log(epprof::Ev::kReadyQueued, c->s->Id(),
                static_cast<int64_t>(ready_connections_.size()));
    // In `Decode()`, it will remove hdl from `active_`
    active_.insert(c->hdl);
    any = true;
  }
  if (any) {
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }
}

void OnlineWebsocketDecoder::ProcessVadConnections(const asio::error_code &ec) {
  if (ec) {
    SHERPA_ONNX_LOG(FATAL) << "The VAD loop is aborted!";
  }

  // `connections_` is protected by `mutex_` (see its declaration above),
  // not `vad_mutex_`. Iterating it here under vad_mutex_ alone raced with
  // ProcessConnections()'s `connections_.erase(hdl)` under mutex_ on
  // another thread -- confirmed via AddressSanitizer as a heap-use-after-
  // free (a Connection destroyed by the ASR decode-loop thread while this
  // VAD-loop thread was still mid-iteration/copying its shared_ptr out of
  // the same std::map). This was the actual root cause of the crash found
  // under load -- not a memory leak, not memory pressure: a lock taken
  // on the wrong mutex for the data it was protecting. Snapshot under the
  // correct lock first, then do the vad_ready_connections_/vad_active_
  // bookkeeping under vad_mutex_ as before.
  std::vector<std::pair<connection_hdl, std::shared_ptr<Connection>>>
      snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.reserve(connections_.size());
    for (auto &p : connections_) {
      snapshot.emplace_back(p.first, p.second);
    }
  }

  std::lock_guard<std::mutex> lock(vad_mutex_);
  for (auto &p : snapshot) {
    auto hdl = p.first;
    auto c = p.second;

    if (!server_->Contains(hdl)) continue;
    if (vad_active_.count(hdl)) continue;
    if (c->eof) continue;
    // Nothing to score yet.
    if (!recognizer_->VadHasWindow(c->s.get())) continue;

    vad_ready_connections_.push_back(c);
    // In VadDecode(), it will remove hdl from vad_active_.
    vad_active_.insert(hdl);
  }

  if (!vad_ready_connections_.empty()) {
    if (epprof::Enabled()) {
      static int64_t last_vad_tick_ns = 0;  // guarded by vad_mutex_
      int64_t now = epprof::NowNs();
      epprof::Log(epprof::Ev::kVadTick, 0,
                  static_cast<int64_t>(vad_ready_connections_.size()),
                  last_vad_tick_ns ? (now - last_vad_tick_ns) / 1000000 : 0);
      last_vad_tick_ns = now;
    }
    asio::post(server_->GetVadContext(), [this]() { VadDecode(); });
  }

  // Schedule another call
  vad_timer_.expires_after(
      std::chrono::milliseconds(config_.vad_loop_interval_ms));

  vad_timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessVadConnections(ec); });
}

void OnlineWebsocketDecoder::VadDecode() {
  std::unique_lock<std::mutex> lock(vad_mutex_);
  if (vad_ready_connections_.empty()) return;

  std::vector<std::shared_ptr<Connection>> c_vec;
  std::vector<OnlineStream *> s_vec;
  while (!vad_ready_connections_.empty() &&
         static_cast<int32_t>(s_vec.size()) < config_.max_vad_batch_size) {
    auto c = vad_ready_connections_.front();
    vad_ready_connections_.pop_front();
    c_vec.push_back(c);
    s_vec.push_back(c->s.get());
  }

  if (!vad_ready_connections_.empty()) {
    asio::post(server_->GetVadContext(), [this]() { VadDecode(); });
  }

  const int64_t prof_t0 = epprof::Enabled() ? epprof::NowNs() : 0;
  const int64_t prof_remaining =
      static_cast<int64_t>(vad_ready_connections_.size());
  lock.unlock();
  // The actual VadProcess call: parallel to, and scheduled independently
  // of, DecodeStreams(). Operates purely on each stream's copied-audio
  // buffer and its own OnlineVadBuffer -- never touches num_processed_frames_
  // or any other ASR decode state directly.
  recognizer_->VadProcess(s_vec.data(), s_vec.size());
  // Decode flush chunks now, not at the next ASR tick.
  ScheduleVadFlushes(c_vec);
  lock.lock();
  if (epprof::Enabled()) {
    epprof::Log(epprof::Ev::kVadBatch, 0, static_cast<int64_t>(s_vec.size()),
                prof_remaining, epprof::NowNs() - prof_t0);
  }

  for (auto &c : c_vec) {
    vad_active_.erase(c->hdl);
  }
}

OnlineWebsocketServer::OnlineWebsocketServer(
    asio::io_context &io_conn, asio::io_context &io_work,
    asio::io_context &io_vad, const OnlineWebsocketServerConfig &config)
    : config_(config),
      io_conn_(io_conn),
      io_work_(io_work),
      io_vad_(io_vad),
      log_(OpenOutputFile(config.log_file, std::ios::app)),
      tee_(std::cout, log_),
      decoder_(this) {
  SetupLog();

  server_.init_asio(&io_conn_);

  server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

  server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

  server_.set_message_handler(
      [this](connection_hdl hdl, server::message_ptr msg) {
        OnMessage(hdl, msg);
      });
}

void OnlineWebsocketServer::Run(uint16_t port) {
  server_.set_reuse_addr(true);
  server_.listen(asio::ip::tcp::v4(), port);
  server_.start_accept();
  auto recognizer_config = config_.decoder_config.recognizer_config;
  int32_t warm_up = recognizer_config.model_config.warm_up;
  const std::string &model_type = recognizer_config.model_config.model_type;
  if (0 < warm_up && warm_up < 100) {
    if (model_type == "zipformer2") {
      decoder_.Warmup();
      SHERPA_ONNX_LOGE("Warm up completed : %d times.", warm_up);
    } else {
      SHERPA_ONNX_LOGE("Only Zipformer2 has warmup support for now.");
      SHERPA_ONNX_LOGE("Given: %s", model_type.c_str());
      SHERPA_ONNX_EXIT(0);
    }
  } else if (warm_up == 0) {
    SHERPA_ONNX_LOGE("Starting without warmup!");
  } else {
    SHERPA_ONNX_LOGE("Invalid Warm up Value!. Expected 0 < warm_up < 100");
    SHERPA_ONNX_EXIT(0);
  }
  decoder_.Run();
  decoder_.RunVad();
}

void OnlineWebsocketServer::SetupLog() {
  server_.clear_access_channels(websocketpp::log::alevel::all);
  // server_.set_access_channels(websocketpp::log::alevel::connect);
  // server_.set_access_channels(websocketpp::log::alevel::disconnect);

  // So that it also prints to std::cout and std::cerr
  server_.get_alog().set_ostream(&tee_);
  server_.get_elog().set_ostream(&tee_);
}

void OnlineWebsocketServer::Send(connection_hdl hdl, const std::string &text) {
  websocketpp::lib::error_code ec;
  if (!Contains(hdl)) {
    return;
  }

  server_.send(hdl, text, websocketpp::frame::opcode::text, ec);
  if (ec) {
    server_.get_alog().write(websocketpp::log::alevel::app, ec.message());
  }
}

void OnlineWebsocketServer::OnOpen(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.insert(hdl);

  std::ostringstream os;
  os << "New connection: "
     << server_.get_con_from_hdl(hdl)->get_remote_endpoint() << ". "
     << "Number of active connections: " << connections_.size() << ".\n";
  SHERPA_ONNX_LOG(INFO) << os.str();
}

void OnlineWebsocketServer::OnClose(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.erase(hdl);

  SHERPA_ONNX_LOG(INFO) << "Number of active connections: "
                        << connections_.size() << "\n";
}

bool OnlineWebsocketServer::Contains(connection_hdl hdl) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.count(hdl);
}

void OnlineWebsocketServer::OnMessage(connection_hdl hdl,
                                      server::message_ptr msg) {
  auto c = decoder_.GetOrCreateConnection(hdl);

  const std::string &payload = msg->get_payload();

  if (epprof::Enabled()) {
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!c->prof_open_logged) {
      c->prof_open_logged = true;
      // remote endpoint looks like "127.0.0.1:54321" or "[::1]:54321"
      std::string ep = server_.get_con_from_hdl(hdl)->get_remote_endpoint();
      auto pos = ep.rfind(':');
      int64_t port = pos == std::string::npos ? 0 : std::atoll(ep.c_str() + pos + 1);
      epprof::Log(epprof::Ev::kConnOpen, c->s->Id(), port);
    }
    if (msg->get_opcode() == websocketpp::frame::opcode::binary) {
      epprof::Log(epprof::Ev::kMsgRx, c->s->Id(),
                  static_cast<int64_t>(payload.size() / sizeof(float)));
    }
  }

  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text:
      if (payload == "Done") {
        asio::post(io_work_, [this, c]() { decoder_.InputFinished(c); });
      }
      break;
    case websocketpp::frame::opcode::binary: {
      auto p = reinterpret_cast<const float *>(payload.data());
      int32_t num_samples = payload.size() / sizeof(float);
      std::vector<float> samples(p, p + num_samples);

      {
        std::lock_guard<std::mutex> lock(c->mutex);
        c->samples.push_back(std::move(samples));
      }

      asio::post(io_work_, [this, c]() { decoder_.AcceptWaveform(c); });
      break;
    }
    default:
      break;
  }
}

void OnlineWebsocketServer::Close(connection_hdl hdl,
                                  websocketpp::close::status::value code,
                                  const std::string &reason) {
  auto con = server_.get_con_from_hdl(hdl);

  std::ostringstream os;
  os << "Closing " << con->get_remote_endpoint() << " with reason: " << reason
     << "\n";

  websocketpp::lib::error_code ec;
  server_.close(hdl, code, reason, ec);
  if (ec) {
    os << "Failed to close" << con->get_remote_endpoint() << ". "
       << ec.message() << "\n";
  }
  server_.get_alog().write(websocketpp::log::alevel::app, os.str());
}

}  // namespace sherpa_onnx
