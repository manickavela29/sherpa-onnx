// sherpa-onnx/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-transducer-model-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineTransducerModelConfig::Register(ParseOptions *po) {
  // SHERPA_ONNX_LOGE("Printing confs in transducer %s : %s",encoder_config.ToString().c_str());
  encoder_config.Register(po);
  decoder_config.Register(po);
  joiner_config.Register(po);

  po->Register("encoder", &encoder, "Path to encoder.onnx");
  po->Register("decoder", &decoder, "Path to decoder.onnx");
  po->Register("joiner", &joiner, "Path to joiner.onnx");
    std::ostringstream os;
  po->PrintConfig(os);
  SHERPA_ONNX_LOGE("Args : %s",os.str().c_str());
}

bool OnlineTransducerModelConfig::Validate() const {
  SHERPA_ONNX_LOGE("Provider at transducer : %s\n",provider.c_str());
  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("transducer encoder: '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("transducer decoder: '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (!FileExists(joiner)) {
    SHERPA_ONNX_LOGE("joiner: '%s' does not exist", joiner.c_str());
    return false;
  }

  return true;
}

std::string OnlineTransducerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineTransducerModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "joiner=\"" << joiner << "\", ";;
  os << "encoder_config=\"" << encoder_config.ToString() << "\", ";
  os << "decoder_config=\"" << decoder_config.ToString() << "\", ";
  os << "joiner_config=\"" << joiner_config.ToString() << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
