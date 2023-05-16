#pragma once

#include <shared_mutex>
#include <unordered_map>

namespace ikov {
class TrieThreadSafe {
 private:
  struct TrieNode {
    bool is_end_;
    std::unordered_map<char, TrieNode*> children_;

    TrieNode() : is_end_(false), children_() {}
    ~TrieNode() {
      for (auto& [ch, node] : children_) {
        delete node;
      }
    };
  };

  mutable std::shared_mutex mutex_;
  TrieNode* root_;

 public:
  TrieThreadSafe();
  ~TrieThreadSafe();

  void insert(std::string word);
  bool search(std::string word) const;
  bool startWith(std::string prefix) const;
  void clear();
};
}  // namespace ikov
