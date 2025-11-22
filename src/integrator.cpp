#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      // Anti-aliasing: support multi-ray sampling per pixel. If `spp` is a
      // perfect square, use a stratified `sqrt(spp) x sqrt(spp)` grid inside
      // the pixel for more uniform coverage; otherwise fall back to random
      // jitter provided by `Sampler::getPixelSample`.
      int sqrt_spp        = (int)std::lround(std::sqrt((double)spp));
      bool use_stratified = (sqrt_spp * sqrt_spp == spp && sqrt_spp > 0);

      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.
        Vec2f pixel_sample;
        if (use_stratified) {
          int sx = sample % sqrt_spp;
          int sy = sample / sqrt_spp;
          // jitter inside subcell
          Vec2f jitter = sampler.get2D();
          pixel_sample = Cast<Float>(Vec2i(dx, dy)) +
                         Vec2f((sx + jitter.x) / (Float)sqrt_spp,
                             (sy + jitter.y) / (Float)sqrt_spp);
        } else {
          // random sub-pixel sample
          pixel_sample = sampler.getPixelSample();
        }

        // Generate a differential ray through the sampled position
        DifferentialRay ray =
            camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      // Sample the BSDF to get the new incoming direction `wi` (delta
      // BSDFs should set interaction.wi). We ignore the returned value here
      // because we only need the new direction to continue the path.
      Float pdf = 0.0F;
      if (interaction.bsdf) {
        interaction.bsdf->sample(interaction, sampler, &pdf);
      }

      // Spawn a new ray along the sampled direction. Note: spawnRay
      // returns a `Ray`, and `DifferentialRay` has an assignment from
      // `Ray`, which clears differentials — that's acceptable for
      // specular bounces.
      Ray new_ray = interaction.spawnRay(interaction.wi);
      ray         = new_ray;
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);
  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);
  // auto test_ray       = DifferentialRay(interaction.p, light_dir);
  // TODO(HW3): Test for occlusion
  //
  // You should test if there is any intersection between interaction.p and
  // point_light_position using scene->intersect. If so, return an occluded
  // color. (or Vec3f color(0, 0, 0) to be specific)
  //
  // You may find the following variables useful:
  //
  // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
  //    This function tests whether the ray intersects with any geometry in the
  //    scene. And if so, it returns true and fills the interaction with the
  //    intersection information.
  //
  //    You can use iteraction.p to get the intersection position.
  //
  // Create a shadow ray from the interaction point to the point light and
  // test for occlusion. Use spawnRayTo to correctly set the ray origin and
  // the maximum travel distance to just before the light position.
  Ray test_ray = interaction.spawnRayTo(point_light_position);
  SurfaceInteraction shadow_it;
  if (scene->intersect(test_ray, shadow_it)) {
    // Occluded
    return color;
  }

  // Not occluded, compute the contribution using perfect diffuse diffuse model
  // Perform a quick and dirty check to determine whether the BSDF is ideal
  // diffuse by RTTI
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

  if (bsdf != nullptr && is_ideal_diffuse) {
    // TODO(HW3): Compute the contribution
    //
    // You can use bsdf->evaluate(interaction) * cos_theta to approximate the
    // albedo. In this homework, we do not need to consider a
    // radiometry-accurate model, so a simple phong-shading-like model is can be
    // used to determine the value of color.

    // The angle between light direction and surface normal
    Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);

    // Set incoming direction for BSDF evaluation
    interaction.wi = light_dir;

    // Evaluate BSDF (albedo) and apply cosine and inverse-square falloff
    Vec3f albedo      = bsdf->evaluate(interaction);
    Vec3f attenuation = point_light_flux / (dist_to_light * dist_to_light);
    color             = albedo * cos_theta * attenuation;
  }

  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  return Vec3f(0.0f);
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  return Vec3f(0.0f);
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  return Vec3f(0.0f);
}

RDR_NAMESPACE_END
