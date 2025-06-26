#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct Journal {
  std::string title_;
  std::vector<std::string> entries_;

  explicit Journal(const std::string& title) : title_(title) {}
  void add(const std::string& entry) {
    static int counter = 0;
    entries_.push_back(std::to_string(counter++) + ": " + entry + "\n");
  }
};

struct Persistency {
  static void save(const Journal& journal) {
    std::ios_base::openmode mode;
    if (std::filesystem::exists(filename)) {
      mode = std::ios::app;
    } else {
      mode = std::ios::out;
    }
    std::ofstream ofs(filename.c_str(), mode);
    for (auto entry : journal.entries_) {
      ofs << entry;
    }
  }

 private:
  static inline std::string filename = "JournalPersistency.log";
};

int main() {
  Journal j(std::string{"Ihor"});
  j.add(std::string("I am super"));
  j.add(std::string("I am the best"));

  Persistency::save(j);
  return 0;
}