#include <string>
#include <memory>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include <queue>

#include <iostream>
#include <sstream>

namespace foo {
using namespace std;

struct PrefixCmp {
  bool operator()(const std::string& lhs, const std::string& rhs) {
    if (lhs == rhs) {
      return false;
    }
    if (lhs.size() == rhs.size()) {
      return lhs.compare(rhs);
    }
    return lhs.size() < rhs.size();
  }
};

using RangesIt = list<list<uint32_t> >::iterator;

struct PrefixIteratorCmp {
  bool operator()(const map<string, RangesIt>::iterator lhs,
                  const map<string, RangesIt>::iterator rhs) const {
    return &lhs->first != &rhs->first;
  }
};

string largest(vector<string> vs) {
  // list of ranges of indexes of words to iterate
  list<list<uint32_t> > ranges{};
  list<uint32_t> first;
  for (uint32_t i = 0; i < vs.size(); ++i) {
    first.push_back(i);
  }
  ranges.push_back(first);
  // preconditon is that we didn't find prefix, so
  // all words are for the empty "" prefix
  map<string, RangesIt> prefix_to_range{{"", ranges.begin()}};
  uint32_t position{0};
  RangesIt ranges_to_check;
  set<map<string, RangesIt>::iterator, PrefixIteratorCmp> prefixes_to_erase;

  do {
    ranges_to_check = ranges.end();
    map<string, RangesIt> new_prefixes;
    // iterate over all potential prefixes
    for (auto prefix_key = prefix_to_range.begin();
         prefix_key != prefix_to_range.end();
         ++prefix_key) {
      std::map<char, list<uint32_t> > matches;
      // iterate over all words for this prefix
      auto range_it_ = prefix_key->second;
      for (auto index_it = range_it_->begin(); index_it != range_it_->end();
           ++index_it) {
        auto matches_it = matches.end();
        if (position < vs[*index_it].size()) {
          matches_it = matches.find(vs[*index_it][position]);
        } else {
          continue;
        }
        if (matches_it != matches.end()) {
          // this char is common for at least two words
          prefixes_to_erase.insert(prefix_key);
          string s = prefix_key->first;
          s.push_back(vs[*index_it][position]);
          auto range_it_search = new_prefixes.find(s);
          if (range_it_search == new_prefixes.end()) {
            matches_it->second.push_back(*index_it);
            list<uint32_t> to_insert;
            to_insert.insert(to_insert.end(),
                             matches_it->second.begin(),
                             matches_it->second.end());
            RangesIt i = ranges.insert(ranges.end(), to_insert);
            new_prefixes.insert({s, i});
            ranges_to_check = i;
          } else {
            new_prefixes.at(s)->push_back(*index_it);
          }
        } else {
          matches[vs[*index_it][position]] = {*index_it};
        }
      }
    }
    for (const auto& item : new_prefixes) {
      prefix_to_range.insert(item);
    }
    for (const auto& item : prefixes_to_erase) {
      prefix_to_range.erase(item);
    }
    prefixes_to_erase.clear();

    position++;
    ranges.pop_front();
  } while (ranges.end() != ranges_to_check);

  using PrefixToRangeVal = pair<string, RangesIt>;
  return prefix_to_range.begin()->first;
}
}

struct lessStringCmp {
  bool operator()(const std::string& lhs, const std::string& rhs) {
    return lhs.size() < rhs.size();
  }
};

int main(int argc, char const* argv[]) {
  std::priority_queue<std::string, std::vector<std::string>, lessStringCmp> pq;
  pq.push({"Hello"});
  pq.push({"War"});
  std::cout << pq.top() << std::endl;
  pq.pop();
  std::cout << pq.top() << std::endl;

  std::cout << "The largest prefix is \"" << foo::largest({"foo"}) << " \".\n";

  std::cout << "The largest prefix is \"" << foo::largest({"foo", "bar"})
            << " \".\n";
  std::cout << "The largest prefix is \"" << foo::largest({"foo", "bar"})
            << " \".\n";

  std::cout << "The largest prefix is \""
            << foo::largest({"foo", "bar", "foot"}) << " \".\n";

  std::cout << "The largest prefix is \""
            << foo::largest({"foo", "barmen", "foot", "barmaley", "fee"})
            << " \".\n";
  std::cout << "The largest prefix is \""
            << foo::largest({"flower", "flow", "flight"}) << " \".\n";
  return 0;
}
