#include "trie.h"

namespace ikov {

Trie::Trie() { root_ = new TrieNode; }

Trie::~Trie() { delete root_; }

/// @brief Insert word in the Trie structure
/// @param word
void Trie::insert(std::string word) {
  TrieNode* node = root_;
  for (auto& c : word) {
    if (!node->children_[c]) {
      node->children_[c] = new TrieNode;
    }
    node = node->children_[c];
  }
  node->is_end_ = true;
}

bool Trie::search(std::string word) {
  TrieNode* node = root_;
  for (auto& c : word) {
    if (!node->children_[c]) {
      return false;
    }
    node = node->children_[c];
  }
  return node->is_end_;
}

bool Trie::startWith(std::string prefix) {
  TrieNode* node = root_;
  for (auto& c : prefix) {
    if (!node->children_[c]) {
      return false;
    }
    node = node->children_[c];
  }
  return true;
}

}  // namespace ikov