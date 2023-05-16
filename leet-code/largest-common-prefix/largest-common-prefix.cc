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

// struct PrefixNode {
//   char key;
//   set<shared_ptr<PrefixNode>> childs;
//   list<uint32_t> words;
  
//   using ChildPtr = set<shared_ptr<PrefixNode>>::iterator;
//   pair<uint32_t, ChildPtr> max_child{0, nullptr}; 
//   set<ChildPtr> to_delete;

//   PrefixNode() = default;
//   PrefixNode(const char& c_): key(c_){};

//   PrefixNode& addChild(const char& ch){
//     shared_ptr<PrefixNode> p = make_shared<PrefixNode>(ch);
//     childs.insert(p);
//     return *p;
//   }

//   ChildPtr insertChild(const char& ch, const uint32_t& word){
//     auto child_ptr = find(childs.begin(),childs.end(),
//       [ch](ChildPtr it){return (*it)->key == ch;});
//     if(child_ptr != childs.end()){
//       to_delete.erase(child_ptr);
//       (*child_ptr)->insertWord(word);
//       if( (max_child.first == 0) or 
//           (max_child.first < (*child_ptr)->words.size()) ){
//         max_child = {(*child_ptr)->words.size(), child_ptr};
//       }
//       return child_ptr;
//     }

//     shared_ptr<PrefixNode> p = make_shared<PrefixNode>(ch);
//     auto it = childs.insert(childs.end(), p);
//     to_delete.insert(it);
//     return --childs.end();
//   }

//   void cleanup(){
//     for (auto it = to_delete.begin(); it != to_delete.end(); ++it){
//       childs.erase(*it);
//     }
//     to_delete.clear();
//   }

//   using WordsPtr = list<uint32_t>::iterator;

//   WordsPtr insertWord(const uint32_t& word){
//     return words.insert(words.end(), word);
//   }

//   string str() const {
//     ostringstream oss;
//     oss << key;
//     if(childs.size() > 0){
//       oss << "\n";
//       for(auto it = childs.begin(); it != childs.end(); ++it){
//         oss << (*it)->str();
//       }
//     } else {
//       oss << " ";
//     }
//     return oss.str();
//   }
// };

struct PrefixCmp
{
  bool operator()(const std::string& lhs, const std::string& rhs){
    if (lhs == rhs)
    {
      return false;
    }
    if (lhs.size() == rhs.size()){
      return lhs.compare(rhs);
    }
    return lhs.size() < rhs.size();
  }
};

string largest(vector<string> vs){
  //list of ranges of indexes of words to iterate
  list<list<uint32_t>> ranges{};
  for(uint32_t i = 0; i < vs.size(); ++i){
    ranges.front().push_back(i);
  }
  using RangesIt = list<list<uint32_t>>::iterator;
  map<string, RangesIt> prefix_to_range{{"", ranges.begin()}};
  uint32_t position{0};
  RangesIt to_check;

  do{
    to_check = ranges.end();
    //iterate over all potential prefixes
    for(auto prefix_key = prefix_to_range.begin(); 
        prefix_key != prefix_to_range.end(); ++prefix_key){
      std::map<char, list<uint32_t>> matches;
      // iterate over all words for this prefix
      auto range_it_ = prefix_key->second;
      for(auto index_it = range_it_->begin();
          index_it != range_it_->end();
          ++index_it){
        auto matches_it = matches.find(vs[*index_it][position]);
        if(matches_it != matches.end()){
          string s = prefix_key->first;
          s.push_back(vs[*index_it][position]);
          auto range_it_search = prefix_to_range.find(s);
          if(range_it_search == prefix_to_range.end()){
            RangesIt i = ranges.insert(ranges.end(),{*index_it});
            prefix_to_range.insert({s, i});
            to_check = i;
          } else {
            prefix_to_range.at(s)->push_back(*index_it);
          }
        }
        matches_it->second.push_back(*index_it);
      }
    }

    position++;
    ranges.pop_front();
  } while (ranges.end() != to_check);
  
  using PrefixToRangeVal = pair<string, RangesIt>;
  return prefix_to_range.begin()->first;
}

}

struct lessStringCmp
{
  bool operator()(const std::string& lhs, const std::string& rhs){
    return lhs.size() < rhs.size();
  }
};


int main(int argc, char const *argv[])
{
  // foo::PrefixNode node('d');
  // node.addChild('c');
  // node.addChild('f');
  // node.addChild('g').addChild('a');
  // std::cout << node.str() << std::endl;

  // foo::PrefixNode bar('q');
  // auto ptr = bar.insertChild('a');
  // std::cout << (*ptr)->key;

  std::priority_queue<std::string, std::vector<std::string>, lessStringCmp> pq;
  pq.push({"Hello"});
  pq.push({"War"});
  std::cout << pq.top() << std::endl;
  pq.pop();
  std::cout << pq.top() << std::endl;

  std::cout << "The largest prefix of" << foo::largest({"foo", "bar"}) << '\n';
  return 0;
}