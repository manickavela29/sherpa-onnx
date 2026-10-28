// sherpa-onnx/csrc/endpoint-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/endpoint.h"

#include "gtest/gtest.h"

namespace sherpa_onnx {

// Default rules: rule1 = 2.4 s of trailing silence even if nothing was
// decoded, rule2 = 1.2 s of trailing silence after decoding something,
// rule3 = an utterance of 20 s. Frames are 10 ms. The cases stay away from the
// exact thresholds, where float rounding of frames x 0.01f decides the result.
TEST(Endpoint, FiredRuleWithDefaultConfig) {
  Endpoint endpoint{EndpointConfig()};
  const float frame_shift = 0.01;

  // 5 s of audio, the last 3 s silent: rule1 (checked first) fires
  EXPECT_EQ(endpoint.FiredRule(500, 300, frame_shift), 1);

  // 5 s of audio, the last 1.25 s silent: only rule2 fires
  EXPECT_EQ(endpoint.FiredRule(500, 125, frame_shift), 2);

  // 25 s of audio, the last 0.5 s silent: only the length rule3 fires
  EXPECT_EQ(endpoint.FiredRule(2500, 50, frame_shift), 3);

  // 3 s of audio, the last 0.5 s silent: nothing fires
  EXPECT_EQ(endpoint.FiredRule(300, 50, frame_shift), 0);

  // 1.25 s of audio and all of it silent: rule2 needs non-silence, rule1 needs
  // 2.4 s of silence
  EXPECT_EQ(endpoint.FiredRule(125, 125, frame_shift), 0);

  // several rules could fire: the first one in the order rule1, rule2, rule3
  EXPECT_EQ(endpoint.FiredRule(3000, 300, frame_shift), 1);
}

TEST(Endpoint, IsEndpointAgreesWithFiredRule) {
  Endpoint endpoint{EndpointConfig()};
  const float frame_shift = 0.01;

  for (int32_t frames : {100, 300, 500, 2500}) {
    for (int32_t silence : {0, 50, 125, 250, 300}) {
      if (silence > frames) {
        continue;
      }
      EXPECT_EQ(endpoint.IsEndpoint(frames, silence, frame_shift),
                endpoint.FiredRule(frames, silence, frame_shift) != 0)
          << frames << " " << silence;
    }
  }
}

}  // namespace sherpa_onnx
