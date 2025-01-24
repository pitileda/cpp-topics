import std;

using namespace std;

std::expected<int, std::string> parse_int(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (const std::exception& e) {
    return std::unexpected("Invalid number: " + str);
  }
}

int main() {
  string x;
  getline(cin, x);
  auto result = parse_int(x);
  if (result) {
    std::cout << "Parsed number: " << *result << '\n';
  } else {
    std::cout << "Error: " << result.error() << '\n';
  }
  return 0;
}
