#include "trie_threadsafe.h"

#include <mutex>

ikov::TrieThreadSafe::TrieThreadSafe() : root_(new TrieNode) {}

ikov::TrieThreadSafe::~TrieThreadSafe() { delete root_; }

void ikov::TrieThreadSafe::insert(std::string word) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  TrieNode* node = root_;
  for (const auto& c : word) {
    if (!node->children_[c]) {
      node->children_[c] = new TrieNode;
    }
    node = node->children_[c];
  }
  node->is_end_ = true;
}

bool ikov::TrieThreadSafe::search(std::string word) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const TrieNode* node = root_;
  for (const auto& c : word) {
    auto it = node->children_.find(c);
    if (it == node->children_.end()) {
      return false;
    }
    node = it->second;
  }
  return node->is_end_;
}

bool ikov::TrieThreadSafe::startWith(std::string prefix) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const TrieNode* node = root_;
  for (const auto& c : prefix) {
    auto it = node->children_.find(c);
    if (it == node->children_.end()) {
      return false;
    }
    node = it->second;
  }
  return true;
}

void ikov::TrieThreadSafe::clear() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  TrieNode* node = root_;
  node->children_.clear();
}