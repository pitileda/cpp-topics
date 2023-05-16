#pragma once

#include <map>
#include <string>

namespace ikov {

class Trie {
 private:
  struct TrieNode {
    bool is_end_;
    std::map<char, TrieNode*> children_;

    ~TrieNode() {
      for (auto& [ch, node] : children_) {
        delete node;
      }
    }
  };

  TrieNode* root_;

 public:
  Trie(/* args */);
  ~Trie();

  void insert(std::string word);
  bool search(std::string word);
  bool startWith(std::string prefix);
};

}  // namespace ikov