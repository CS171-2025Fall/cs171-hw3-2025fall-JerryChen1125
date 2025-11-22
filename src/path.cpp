#include "rdr/path.h"

#include "rdr/bsdf.h"
#include "rdr/light.h"
#include "rdr/scene.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 * It is your turn!
 * =====================================================================
 */

Vec3f Path::estimate() const {
  // This is left as the next assignment
  return Vec3f(0.0f);
}

bool Path::verify() const {
  bool result = true;
  if (this->length() == 0) return result;

  result &= interactions[0].wo == -ray0.direction;
  AssertAllNormalized(interactions[0].wo);
  AssertAllNormalized(ray0.direction);
  for (size_t i = 0; i < interactions.size() - 1; ++i) {
    auto interaction = interactions[i];
    auto primitive   = interaction.primitive;
    auto bsdf        = interaction.bsdf;

    result &= primitive != nullptr;
    result &= bsdf != nullptr;
    result &= AllClose(interaction.wi, -interactions[i + 1].wo);
    AssertAllNormalized(interaction.wi);
  }

  return result;
}

std::string Path::toString() const {
  // https://graphics.stanford.edu/courses/cs348b-01/course29.hanrahan.pdf
  std::ostringstream ss;
  ss << "Path["
     << "E" << ToString(ray0.origin);

  if (!interactions.empty()) ss << " -> ";
  for (size_t i = 0; i < interactions.size(); ++i) {
    switch (interactions[i].type) {
      case ESurfaceInteractionType::EDiffuse:
        ss << "D";
        break;
      case ESurfaceInteractionType::EGlossy:
        ss << "G";
        break;
      case ESurfaceInteractionType::ESpecular:
        ss << "S";
        break;
      case ESurfaceInteractionType::EInfLight:
        ss << "IL";
        break;
      default:
        ss << "N";
        break;
    }

    ss << "[p" << ToString(interactions[i].p) << ", ";
    ss << "n" << ToString(interactions[i].normal) << ", ";
    ss << "wi" << ToString(interactions[i].wi) << "]";
    if (i < interactions.size() - 1) ss << " -> ";
  }

  ss << "]";
  return ss.str();
}

/* ===================================================================== *
 * Our Debug class (for performance lol)
 * =====================================================================
 */

PathImmediate &PathImmediate::addInteraction(
    const SurfaceInteraction &next_interaction) {
  assert(next_interaction.isValid());
  is_terminated = next_interaction.isLight();

  // TODO: optimize for logic?
  if (cached_length == 0) {
    if (is_terminated)
      cached_result =
          next_interaction.light->Le(next_interaction, next_interaction.wo);
  } else {
    if (is_terminated) {
      const Vec3f &brdf = interaction.isSpecular()
                            ? interaction.bsdf_cache
                            : interaction.bsdf->evaluate(interaction);
      const Vec3f &Le =
          next_interaction.light->Le(next_interaction, next_interaction.wo);
      const Float &pdf =
          toPdfMeasure(next_interaction, interaction, EMeasure::ESolidAngle);
      if (!IsAllValid(pdf)) goto next;

      // Monte Carlo estimator
      cached_result =
          Le * brdf * abs(interaction.cosThetaI()) / pdf * cached_throughput;

      AssertAllValid(cached_result);
      AssertAllNonNegative(cached_result);
    } else {
      const auto *bsdf = interaction.bsdf;

      const Vec3f &brdf = interaction.isSpecular()
                            ? interaction.bsdf_cache
                            : bsdf->evaluate(interaction);
      const Float &pdf =
          toPdfMeasure(next_interaction, interaction, EMeasure::ESolidAngle);
      const Float &cos_term =
          std::abs(Dot(interaction.wi, interaction.shading.n));

      cached_throughput *= brdf * cos_term / pdf;

      AssertAllValid(cached_throughput);
      AssertAllNonNegative(cached_throughput);
    }
  }

next:
  interaction = next_interaction;
  ++cached_length;
  return *this;
}

RDR_NAMESPACE_END
