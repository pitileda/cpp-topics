#include <cstdint>
#include <memory>

#include "provider.h"

class Service {
 public:
  Service(std::shared_ptr<DataProvider> provider) : provider_(provider) {}

  auto doubleData(int id) -> int64_t { return provider_->getData(id) * 2; }

 private:
  std::shared_ptr<DataProvider> provider_;
};