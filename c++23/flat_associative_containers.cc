import std;

int main() {
  std::flat_map<int, std::string> id_to_name{{1, "Ihor"}, {2, "Lena"}};
  for (const auto& [id, name] : id_to_name) {
    std::println("id: {}, name: {}", id, name);
  }

  for (auto i = 0; i < 1'000'000; ++i) {
    id_to_name[i] = "Ihor" + std::to_string(i);
  }

  std::string name;
  std::getline(std::cin, name);

  auto t1 = std::chrono::high_resolution_clock::now();
  for (auto i = 100'000; i < 900'000; ++i) {
    id_to_name[i] = name;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::println("flat_map time: {}",
               std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1));

  std::map<int, std::string> id_to_name_2{{1, "Ihor"}, {2, "Lena"}};
  for (const auto& [id, name] : id_to_name_2) {
    std::println("id: {}, name: {}", id, name);
  }

  for (auto i = 0; i < 1'000'000; ++i) {
    id_to_name_2[i] = "Ihor" + std::to_string(i);
  }

  std::getline(std::cin, name);

  auto t3 = std::chrono::high_resolution_clock::now();
  for (auto i = 100'000; i < 900'000; ++i) {
    id_to_name_2[i] = name;
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  std::println("map time: {}",
               std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3));

  return 0;
}