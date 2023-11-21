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

  const uint32_t index;
  const uint32_t n;
  const uint32_t k;
  uint32_t support = 0;
  bool is_fixed = false;
  std::vector<uint32_t> activated;
};

struct Fiber {
  Fiber(uint32_t from, uint32_t to) : from_area(from), to_area(to) {}

  const uint32_t from_area;
  const uint32_t to_area;
  bool is_active = true;
  std::vector<std::vector<Synapse>> outgoing_synapses;
};

typedef std::map<std::string, std::vector<std::string>> ProjectMap;

class Brain {
 public:
  Brain(float p, float beta, float max_weight, uint32_t seed);

  Area& AddArea(const std::string& name, uint32_t n, uint32_t k,
                bool recurrent = true, bool is_explicit = false);
  void AddStimulus(const std::string& name, uint32_t k);
  void AddFiber(const std::string& from, const std::string& to,
                bool bidirectional = false);

  Area& GetArea(const std::string& name);
  const Area& GetArea(const std::string& name) const;
  Fiber& GetFiber(const std::string& from, const std::string& to);
  const Fiber& GetFiber(const std::string& from, const std::string& to) const;

  void InhibitAll();
  void InhibitFiber(const std::string& from, const std::string& to);
  void ActivateFiber(const std::string& from, const std::string& to);
  void InitProjection(const ProjectMap& graph);

  void ActivateArea(const std::string& name, uint32_t assembly_index);

  void SimulateOneStep(bool update_plasticity = true);
  void Project(const ProjectMap& graph, uint32_t num_steps,
               bool update_plasticity = true);

  void ReadAssembly(const std::string& name, size_t& index, size_t& overlap);

  void SetLogLevel(int log_level) { log_level_ = log_level; }
  void LogGraphStats();
  void LogActivated(const std::string& area_name);

 private:
  void ComputeKnownActivations(const Area& to_area,
                               std::vector<Synapse>& activations);
  void GenerateNewCandidates(const Area& to_area, uint32_t total_k,
                             std::vector<Synapse>& activations);
  void ConnectNewNeuron(Area& area,
                        uint32_t num_synapses_from_activated,
                        uint32_t& total_synapses_from_non_activated);
  void ChooseSynapsesFromActivated(const Area& area,
                                   uint32_t num_synapses);
  void ChooseSynapsesFromNonActivated(const Area& area,
                                      uint32_t& total_synapses);
  void ChooseOutgoingSynapses(const Area& area);
  void UpdatePlasticity(Area& to_area,
                        const std::vector<uint32_t>& new_activated);

 protected:
  std::mt19937 rng_;
  int log_level_ = 0;

  const float p_;
  const float beta_;
  const float learn_rate_;
  const float max_weight_;
  std::vector<Area> areas_;
  std::vector<Fiber> fibers_;
  std::vector<std::vector<uint32_t>> incoming_fibers_;
  std::vector<std::vector<uint32_t>> outgoing_fibers_;
  std::map<std::string, uint32_t> area_by_name_;
  std::vector<std::string> area_name_;
  uint32_t step_ = 0;
};

}  // namespace nemo

#endif // NEMO_BRAIN_H_
