#ifndef NEMO_BRAIN_H_
#define NEMO_BRAIN_H_

#include <stdint.h>

#include <map>
#include <random>
#include <string>
#include <vector>

namespace nemo {

struct Synapse {
  uint32_t neuron;
  float weight;
};

struct Area {
  Area(uint32_t index, uint32_t n, uint32_t k) : index(index), n(n), k(k) {}

  uint32_t index;
  uint32_t n;
  uint32_t k;
  uint32_t support = 0;
  bool is_fixed = false;
  std::vector<uint32_t> activated;
};

struct Fiber {
  Fiber(uint32_t from, uint32_t to) : from_area(from), to_area(to) {}

  uint32_t from_area;
  uint32_t to_area;
  bool is_active = true;
  std::vector<std::vector<Synapse>> outgoing_synapses;
};

class Brain {
 public:
  Brain(float p, float beta, uint32_t seed);

  Area& AddArea(const std::string& name, uint32_t n, uint32_t k,
                bool recurrent = true);
  void AddStimulus(const std::string& name, uint32_t k);
  void AddFiber(const std::string& from, const std::string& to);

  Area& GetArea(const std::string& name);
  const Area& GetArea(const std::string& name) const;
  Fiber& GetFiber(const std::string& from, const std::string& to);

  void InhibitAll();
  void InhibitFiber(const std::string& from, const std::string& to);
  void ActivateFiber(const std::string& from, const std::string& to);
  void InitProjection(
      const std::map<std::string, std::vector<std::string>>& graph);

  void SimulateOneStep();

  void LogGraphStats();

 private:
  void ComputeKnownActivations(const Area& to_area,
                               std::vector<Synapse>& activations);
  void GenerateNewCandidates(const Area& to_area, uint32_t total_k,
                             std::vector<Synapse>& activations);
  void ConnectNewNeuron(Area& area,
                        uint32_t num_synapses_from_activated);
  void ChooseSynapsesFromActivated(const Area& area,
                                   uint32_t num_synapses);
  void ChooseSynapsesFromNonActivated(const Area& area);
  void ChooseOutgoingSynapses(const Area& area);
  void UpdatePlasticity(Area& to_area,
                        const std::vector<uint32_t>& new_activated);

  float p_;
  float beta_;
  std::mt19937 rng_;
  std::vector<Area> areas_;
  std::vector<Fiber> fibers_;
  std::vector<std::vector<uint32_t>> incoming_fibers_;
  std::vector<std::vector<uint32_t>> outgoing_fibers_;
  std::map<std::string, uint32_t> area_by_name_;
  std::vector<std::string> area_name_;
};

}  // namespace nemo

#endif // NEMO_BRAIN_H_
