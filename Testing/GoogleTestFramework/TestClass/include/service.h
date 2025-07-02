#include <cstdint>
#include <memory>

#include "provider.h"

class Service {
 public:
  Service(std::shared_ptr<DataProvider> provider) : provider(provider) {}

  auto doubleData(int id) -> int64_t { return provider->getData(id); }

 private:
  std::shared_ptr<DataProvider> provider;
};