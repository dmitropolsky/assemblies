#include "brain.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace nemo {
namespace {

float BinomQuantile(uint32_t k, float p, float percent) {
  double pi = std::pow(1.0 - p, k);
  double mul = (1.0 * p) / (1.0 - p);
  double total_p = pi;
  uint32_t i = 0;
  while (total_p < percent) {
    pi *= ((k - i) * mul) / (i + 1);
    total_p += pi;
    ++i;
  }
  return i;
}

template<typename Trng>
float TruncatedNorm(float a, Trng& rng) {
  if (a <= 0.0f) {
    std::normal_distribution<float> norm(0.0f, 1.0f);
    for (;;) {
      const float x = norm(rng);
      if (x >= a) return x;
    }
  } else {
    // Exponential accept-reject algorithm from Robert,
    // https://arxiv.org/pdf/0907.4010.pdf
    const float alpha = (a + std::sqrt(a * a + 4)) * 0.5f;
    std::exponential_distribution<float> d(alpha);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (;;) {
      const float z = a + d(rng);
      const float dz = z - alpha;
      const float rho = std::exp(-0.5f * dz * dz);
      if (u(rng) < rho) return z;
    }
  }
}

template<typename Trng>
std::vector<Synapse> GenerateSynapses(uint32_t support, float p, Trng& rng) {
  std::vector<Synapse> synapses;
  // Sample from geometric(p) distribution by sampling from
  // floor(log(U[0, 1])/log(1-p)).
  std::uniform_real_distribution<float> u(0.0, 1.0);
  const float scale = 1.0f / std::log(1 - p);
  uint32_t last = std::floor(std::log(u(rng)) * scale);
  synapses.reserve(support * p * 1.05);
  while (last < support) {
    synapses.push_back({last, 1.0f});
    last += 1 + std::floor(std::log(u(rng)) * scale);
  }
  return synapses;
}

void SelectTopK(std::vector<Synapse>& activations, uint32_t k) {
  std::nth_element(activations.begin(), activations.begin() + k - 1,
                   activations.end(),
                   [](const Synapse& a, const Synapse& b) {
                     if (a.weight != b.weight) return a.weight > b.weight;
                     return a.neuron < b.neuron;
                   });
  activations.resize(k);
}

}  // namespace

Brain::Brain(float p, float beta, float max_weight, uint32_t seed)
    : rng_(seed), p_(p), beta_(beta), learn_rate_(1.0f + beta_),
      max_weight_(max_weight), areas_(1, Area(0, 0, 0)),
      fibers_(1, Fiber(0, 0)), incoming_fibers_(1), outgoing_fibers_(1),
      area_name_(1, "INVALID") {}

Area& Brain::AddArea(const std::string& name, uint32_t n, uint32_t k,
                     bool recurrent, bool is_explicit) {
  uint32_t area_i = areas_.size();
  areas_.push_back(Area(area_i, n, k));
  if (area_by_name_.find(name) != area_by_name_.end()) {
    fprintf(stderr, "Duplicate area name %s\n", name.c_str());
  }
  if (is_explicit) {
    areas_[area_i].support = n;
  }
  area_by_name_[name] = area_i;
  area_name_.push_back(name);
  incoming_fibers_.push_back({});
  outgoing_fibers_.push_back({});
  if (recurrent) {
    AddFiber(name, name);
  }
  return areas_.back();
}

void Brain::AddStimulus(const std::string& name, uint32_t k) {
  AddArea(name, k, k, /*recurrent=*/false, /*is_explicit=*/true);
  ActivateArea(name, 0);
}

void Brain::AddFiber(const std::string& from, const std::string& to,
                     bool bidirectional) {
  const Area& area_from = GetArea(from);
  const Area& area_to = GetArea(to);
  uint32_t fiber_i = fibers_.size();
  Fiber fiber(area_from.index, area_to.index);
  incoming_fibers_[area_to.index].push_back(fiber_i);
  outgoing_fibers_[area_from.index].push_back(fiber_i);
  for (uint32_t i = 0; i < area_from.support; ++i) {
    std::vector<Synapse> synapses =
        GenerateSynapses(area_to.support, p_, rng_);
    fiber.outgoing_synapses.emplace_back(std::move(synapses));
  }
  fibers_.emplace_back(std::move(fiber));
  if (bidirectional) {
    AddFiber(to, from);
  }
}

Area& Brain::GetArea(const std::string& name) {
  std::map<std::string, uint32_t>::iterator it = area_by_name_.find(name);
  if (it != area_by_name_.end()) {
    return areas_[it->second];
  }
  fprintf(stderr, "Invalid area name %s\n", name.c_str());
  return areas_[0];
}

const Area& Brain::GetArea(const std::string& name) const {
  std::map<std::string, uint32_t>::const_iterator it = area_by_name_.find(name);
  if (it != area_by_name_.end()) {
    return areas_[it->second];
  }
  fprintf(stderr, "Invalid area name %s\n", name.c_str());
  return areas_[0];
}

Fiber& Brain::GetFiber(const std::string& from, const std::string& to) {
  const Area& from_area = GetArea(from);
  const Area& to_area = GetArea(to);
  for (auto fiber_i : outgoing_fibers_[from_area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    if (fiber.to_area == to_area.index) {
      return fiber;
    }
  }
  fprintf(stderr, "No fiber found from %s to %s\n", from.c_str(), to.c_str());
  return fibers_[0];
}

const Fiber& Brain::GetFiber(const std::string& from,
                             const std::string& to) const{
  const Area& from_area = GetArea(from);
  const Area& to_area = GetArea(to);
  for (auto fiber_i : outgoing_fibers_[from_area.index]) {
    const Fiber& fiber = fibers_[fiber_i];
    if (fiber.to_area == to_area.index) {
      return fiber;
    }
  }
  fprintf(stderr, "No fiber found from %s to %s\n", from.c_str(), to.c_str());
  return fibers_[0];
}

void Brain::InhibitAll() {
  for (Fiber& fiber : fibers_) {
    fiber.is_active = false;
  }
}

void Brain::InhibitFiber(const std::string& from, const std::string& to) {
  GetFiber(from, to).is_active = false;
}

void Brain::ActivateFiber(const std::string& from, const std::string& to) {
  GetFiber(from, to).is_active = true;
}

void Brain::ActivateArea(const std::string& name, uint32_t assembly_index) {
  if (log_level_ > 0) {
    printf("Activating %s assembly %u\n", name.c_str(), assembly_index);
  }
  Area& area = GetArea(name);
  uint32_t offset = assembly_index * area.k;
  if (offset + area.k > area.support) {
    fprintf(stderr, "[Area %s] Could not activate assembly index %u "
            "(not enough support: %u vs %u)\n", name.c_str(), assembly_index,
            area.support, offset + area.k);
    return;
  }
  area.activated.resize(area.k);
  for (uint32_t i = 0; i < area.k; ++i) {
    area.activated[i] = offset + i;
  }
  area.is_fixed = true;
}

void Brain::SimulateOneStep(bool update_plasticity) {
  if (log_level_ > 0) {
    if (step_ == 0 && log_level_ > 2) {
      LogGraphStats();
    }
    printf("Step %u%s\n", step_, update_plasticity ? "" : " (readout)");
  }
  std::vector<std::vector<uint32_t>> new_activated(areas_.size());
  for (uint32_t area_i = 0; area_i < areas_.size(); ++area_i) {
    Area& to_area = areas_[area_i];
    uint32_t total_activated = 0;
    for (uint32_t fiber_i : incoming_fibers_[to_area.index]) {
      const Fiber& fiber = fibers_[fiber_i];
      const uint32_t num_activated = areas_[fiber.from_area].activated.size();
      if (!fiber.is_active || num_activated == 0) continue;
      if (log_level_ > 0) {
        printf("%s%s", total_activated == 0 ? "Projecting " : ",",
               area_name_[fiber.from_area].c_str());
      }
      total_activated += num_activated;
    }
    if (total_activated == 0) {
      continue;
    }
    if (log_level_ > 0) {
      printf(" into %s\n", area_name_[area_i].c_str());
    }
    if (!to_area.is_fixed) {
      std::vector<Synapse> activations;
      if (to_area.support > 0) {
        ComputeKnownActivations(to_area, activations);
        SelectTopK(activations, to_area.k);
      }
      if (activations.empty() ||
          activations[to_area.k - 1].weight < total_activated) {
        GenerateNewCandidates(to_area, total_activated, activations);
        SelectTopK(activations, to_area.k);
      }
      if (log_level_ > 1) {
        printf("[Area %s] Cutoff weight for best %d activations: %f\n",
               area_name_[area_i].c_str(), to_area.k,
               activations[to_area.k - 1].weight);
      }
      new_activated[area_i].resize(to_area.k);
      const uint32_t K = to_area.support;
      uint32_t num_new = 0;
      uint32_t total_from_activated = 0;
      uint32_t total_from_non_activated = 0;
      for (uint32_t i = 0; i < to_area.k; ++i) {
        const Synapse& s = activations[i];
        if (s.neuron >= K) {
          new_activated[area_i][i] = K + num_new;
          ConnectNewNeuron(to_area, std::round(s.weight),
                           total_from_non_activated);
          total_from_activated += std::round(s.weight);
          num_new++;
        } else {
          new_activated[area_i][i] = s.neuron;
        }
      }
      if (log_level_ > 1) {
        printf("[Area %s] Num new activations: %u, "
               "new synapses (from activated / from non-activated): %u / %u\n",
               area_name_[area_i].c_str(), num_new, total_from_activated,
               total_from_non_activated);
      }
      std::sort(new_activated[area_i].begin(), new_activated[area_i].end());
    } else {
      new_activated[area_i] = to_area.activated;
    }
    if (update_plasticity) {
      UpdatePlasticity(to_area, new_activated[area_i]);
    }
  }
  for (uint32_t area_i = 0; area_i < areas_.size(); ++area_i) {
    Area& area = areas_[area_i];
    if (!area.is_fixed) {
      std::swap(area.activated, new_activated[area_i]);
    }
  }
  if (log_level_ > 2) {
    LogGraphStats();
  }
  if (update_plasticity) {
    ++step_;
  }
}

void Brain::InitProjection(const ProjectMap& graph) {
  InhibitAll();
  for (const auto& [from, edges] : graph) {
    for (const auto& to : edges) {
      ActivateFiber(from, to);
    }
  }
}

void Brain::Project(const ProjectMap& graph, uint32_t num_steps,
                    bool update_plasticity) {
  InitProjection(graph);
  for (uint32_t i = 0; i < num_steps; ++i) {
    SimulateOneStep(update_plasticity);
  }
}

void Brain::ComputeKnownActivations(const Area& to_area,
                                    std::vector<Synapse>& activations) {
  activations.resize(to_area.support);
  for (uint32_t i = 0; i < activations.size(); ++i) {
    activations[i].neuron = i;
    activations[i].weight = 0;
  }
  for (uint32_t fiber_i : incoming_fibers_[to_area.index]) {
    const Fiber& fiber = fibers_[fiber_i];
    if (!fiber.is_active) continue;
    const Area& from_area = areas_[fiber.from_area];
    for (uint32_t from_neuron : from_area.activated) {
      const auto& synapses = fiber.outgoing_synapses[from_neuron];
      for (size_t i = 0; i < synapses.size(); ++i) {
        activations[synapses[i].neuron].weight += synapses[i].weight;
      }
    }
  }
}

void Brain::GenerateNewCandidates(const Area& to_area, uint32_t total_k,
                                  std::vector<Synapse>& activations) {
  // Compute the total number of neurons firing into this area.
  const uint32_t remaining_neurons = to_area.n - to_area.support;
  if (remaining_neurons <= 2 * to_area.k) {
    // Generate number of synapses for all remaining neurons directly from the
    // binomial(total_k, p_) distribution.
    std::binomial_distribution<> binom(total_k, p_);
    for (uint32_t i = 0; i < remaining_neurons; ++i) {
      activations.push_back({to_area.support + i, binom(rng_) * 1.0f});
    }
  } else {
    // Generate top k number of synapses from the tail of the normal
    // distribution that approximates the binomial(total_k, p_) distribution.
    // TODO(szabadka): For the normal approximation to work, the mean should be
    // at least 9. Find a better approximation if this does not hold.
    const float percent =
        (remaining_neurons - to_area.k) * 1.0f / remaining_neurons;
    const float cutoff = BinomQuantile(total_k, p_, percent);
    const float mu = total_k * p_;
    const float stddev = std::sqrt(total_k * p_ * (1.0f - p_));
    const float a = (cutoff - mu) / stddev;
    if (log_level_ > 1) {
      printf("[Area %s] Generating candidates: percent=%f cutoff=%.0f "
             "mu=%f stddev=%f a=%f\n", area_name_[to_area.index].c_str(),
             percent, cutoff, mu, stddev, a);
    }
    float max_d = 0;
    float min_d = total_k;
    for (uint32_t i = 0; i < to_area.k; ++i) {
      const float x = TruncatedNorm(a, rng_);
      const float d = std::min<float>(total_k, std::round(x * stddev + mu));
      max_d = std::max(d, max_d);
      min_d = std::min(d, min_d);
      activations.push_back({to_area.support + i, d});
    }
    if (log_level_ > 1) {
      printf("[Area %s] Range of %d new candidate connections: %.0f .. %.0f\n",
             area_name_[to_area.index].c_str(), to_area.k, min_d, max_d);
    }
  }
}

void Brain::ConnectNewNeuron(Area& area,
                             uint32_t num_synapses_from_activated,
                             uint32_t& total_synapses_from_non_activated) {
  ChooseSynapsesFromActivated(area, num_synapses_from_activated);
  ChooseSynapsesFromNonActivated(area, total_synapses_from_non_activated);
  ChooseOutgoingSynapses(area);
  ++area.support;
}

void Brain::ChooseSynapsesFromActivated(const Area& area,
                                        uint32_t num_synapses) {
  const uint32_t neuron = area.support;
  uint32_t total_k = 0;
  std::vector<uint32_t> offsets;
  const auto& incoming_fibers = incoming_fibers_[area.index];
  for (uint32_t fiber_i : incoming_fibers) {
    const Fiber& fiber = fibers_[fiber_i];
    const Area& from_area = areas_[fiber.from_area];
    const uint32_t from_k = from_area.activated.size();
    offsets.push_back(total_k);
    if (fiber.is_active) {
      total_k += from_k;
    }
  }
  offsets.push_back(total_k);
  std::uniform_int_distribution<> u(0, total_k - 1);
  std::vector<uint8_t> selected(total_k);
  for (uint32_t j = 0; j < num_synapses; ++j) {
    uint32_t next_i;
    while (selected[next_i = u(rng_)]) {}
    selected[next_i] = 1;
    auto it = std::upper_bound(offsets.begin(), offsets.end(), next_i);
    const uint32_t fiber_i = (it - offsets.begin()) - 1;
    Fiber& fiber = fibers_[incoming_fibers[fiber_i]];
    const Area& from_area = areas_[fiber.from_area];
    uint32_t from = from_area.activated[next_i - offsets[fiber_i]];
    fiber.outgoing_synapses[from].push_back({neuron, 1.0f});
  }
}

void Brain::ChooseSynapsesFromNonActivated(const Area& area,
                                           uint32_t& total_synapses) {
  const uint32_t neuron = area.support;
  for (uint32_t fiber_i : incoming_fibers_[area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    const Area& from_area = areas_[fiber.from_area];
    std::vector<uint8_t> selected(from_area.support);
    size_t num_activated = fiber.is_active ? from_area.activated.size() : 0;
    if (fiber.is_active) {
      for (uint32_t i : from_area.activated) {
        selected[i] = 1;
      }
    }
    if (from_area.support <= 2 * num_activated) {
      std::binomial_distribution<> binom(1, p_);
      for (size_t from = 0; from < from_area.support; ++from) {
        if (!selected[from] && binom(rng_)) {
          fiber.outgoing_synapses[from].push_back({neuron, 1.0f});
          ++total_synapses;
        }
      }
    } else {
      uint32_t population = from_area.support - num_activated;
      std::binomial_distribution<> binom(population, p_);
      std::uniform_int_distribution<> u(0, population - 1);
      size_t num_synapses = binom(rng_);
      for (size_t i = 0; i < num_synapses; ++i) {
        for (;;) {
          uint32_t from = u(rng_);
          if (selected[from]) {
            continue;
          }
          selected[from] = 1;
          fiber.outgoing_synapses[from].push_back({neuron, 1.0f});
          ++total_synapses;
          break;
        }
      }
    }
  }
}

void Brain::ChooseOutgoingSynapses(const Area& area) {
  for (uint32_t fiber_i : outgoing_fibers_[area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    const Area& to_area = areas_[fiber.to_area];
    uint32_t support = to_area.support;
    if (area.index == to_area.index) ++support;
    std::vector<Synapse> synapses = GenerateSynapses(support, p_, rng_);
    fiber.outgoing_synapses.emplace_back(std::move(synapses));
  }
}

void Brain::UpdatePlasticity(Area& to_area,
                             const std::vector<uint32_t>& new_activated) {
  std::vector<uint8_t> is_new_activated(to_area.support);
  for (uint32_t neuron : new_activated) {
    is_new_activated[neuron] = 1;
  }
  for (uint32_t fiber_i : incoming_fibers_[to_area.index]) {
    Fiber& fiber = fibers_[fiber_i];
    if (!fiber.is_active) continue;
    const Area& from_area = areas_[fiber.from_area];
    for (uint32_t from_neuron : from_area.activated) {
      auto& synapses = fiber.outgoing_synapses[from_neuron];
      for (size_t j = 0; j < synapses.size(); ++j) {
        if (is_new_activated[synapses[j].neuron]) {
          synapses[j].weight =
              std::min(synapses[j].weight * learn_rate_, max_weight_);
        }
      }
    }
  }
}

void Brain::ReadAssembly(const std::string& name,
                         size_t& index, size_t& overlap) {
  const Area& area = GetArea(name);
  const size_t num_assemblies = area.n / area.k;
  std::vector<size_t> overlaps(num_assemblies);
  for (auto neuron : area.activated) {
    ++overlaps[neuron / area.k];
  }
  index = std::max_element(overlaps.begin(), overlaps.end()) - overlaps.begin();
  overlap = overlaps[index];
}

void Brain::LogActivated(const std::string& area_name) {
  const Area& area = GetArea(area_name);
  printf("[%s] activated: ", area_name.c_str());
  for (auto n : area.activated) printf(" %u", n);
  printf("\n");
}

void Brain::LogGraphStats() {
  printf("Graph Stats after %u update steps\n", step_);
  for (const auto& area : areas_) {
    if (area.support == 0) continue;
    printf("Area %d [%s] has %d neurons\n",
           area.index, area_name_[area.index].c_str(), area.support);
    if (log_level_ > 3) {
      std::set<uint32_t> tmp(area.activated.begin(), area.activated.end());
      printf("   %s active:", area_name_[area.index].c_str());
      for (auto n : tmp) printf(" %u", n);
      printf("\n");
    }
  }
  const float kThresLow = std::pow(learn_rate_, 10);
  for (const Fiber& fiber : fibers_) {
    if (fiber.outgoing_synapses.empty()) continue;
    size_t num_synapses = 0;
    size_t num_low_weights = 0;
    size_t num_mid_weights = 0;
    size_t num_sat_weights = 0;
    float max_w = 0.0;
    for (uint32_t i = 0; i < fiber.outgoing_synapses.size(); ++i) {
      const auto& synapses = fiber.outgoing_synapses[i];
      num_synapses += synapses.size();
      for (size_t j = 0; j < synapses.size(); ++j) {
        const float w = synapses[j].weight;
        max_w = std::max(w, max_w);
        if (w < kThresLow) ++num_low_weights;
        else if (w < max_weight_) ++num_mid_weights;
        else ++num_sat_weights;
      }
    }
    printf("Fiber %s -> %s has %zu synapses (low/mid/sat: %zu/%zu/%zu), "
           "max w: %f\n", area_name_[fiber.from_area].c_str(),
           area_name_[fiber.to_area].c_str(), num_synapses, num_low_weights,
           num_mid_weights, num_sat_weights, max_w);
  }
}

}  // namespace nemo
