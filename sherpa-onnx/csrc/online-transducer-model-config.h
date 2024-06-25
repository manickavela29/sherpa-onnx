// sherpa-onnx/csrc/online-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/provider-config.h"

namespace sherpa_onnx {

struct OnlineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;

  std::string provider = "cpu";
  // provider handles overriding from OnlineModelconfig
  ProviderConfig encoder_config;
  ProviderConfig decoder_config;
  ProviderConfig joiner_config;

  OnlineTransducerModelConfig(const std::string &provider="cpu")
      : provider(provider),
        encoder_config(provider,"encoder_"),
        decoder_config(provider,"decoder_"),
        joiner_config(provider,"joiner_") {
          SHERPA_ONNX_LOGE("Provider at trnasducer : %s\n",provider.c_str());
        }

  OnlineTransducerModelConfig(const std::string &encoder,
                              const std::string &decoder,
                              const std::string &joiner,
                              const std::string &provider="cpu")
      : encoder(encoder), decoder(decoder), joiner(joiner),
        encoder_config(provider,"encoder_"),
        decoder_config(provider,"decoder_"),
        joiner_config(provider,"joiner_") {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
