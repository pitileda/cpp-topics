#include <iostream>
#include <limits>
#include <map>

template <typename K, typename V>
class interval_map {
  std::map<K, V> m_map;
  V default_v;

 public:
  // constructor associates whole range of K with val by inserting (K_min, val)
  // into the map
  interval_map(V const& val) {
    m_map.insert(m_map.end(),
                 std::make_pair(std::numeric_limits<K>::lowest(), val));
    default_v = V{};
  }

  // Assign value val to interval [keyBegin, keyEnd).
  // Overwrite previous values in this interval.
  // Conforming to the C++ Standard Library conventions, the interval
  // includes keyBegin, but excludes keyEnd.
  // If !( keyBegin < keyEnd ), this designates an empty interval,
  // and assign must do nothing.
  void assign(K const& keyBegin, K const& keyEnd, V const& val) {
    if (!(keyBegin < keyEnd)) {
      return;
    }

    m_map[keyBegin] = val;
    m_map[keyEnd] = val;

    auto next_key_it = m_map.upper_bound(keyBegin);
    K next_key = next_key_it->first;

    while (next_key_it != m_map.end()) {
      next_key = next_key_it->first;
      if (next_key < keyEnd) {
        m_map.erase(next_key_it);
      } else {
        break;
      }
      next_key_it = m_map.upper_bound(keyEnd);
    }
  }

  // look-up of the value associated with key
  V const& operator[](K const& key) const {
    auto curr_key = m_map.find(key);
    auto next_key = m_map.upper_bound(key);

    if (curr_key == m_map.end()) {
      return next_key == m_map.end() ? default_v : next_key->second;
    }
    // check if next key differs
    if (curr_key->second == next_key->second) {
      return curr_key->second;
    } else {
      return next_key->second;
    }
  }
};

// Many solutions we receive are incorrect. Consider using a randomized test
// to discover the cases that your implementation does not handle correctly.
// We recommend to implement a test function that tests the functionality of
// the interval_map, for example using a map of unsigned int intervals to char.

int main() {
  interval_map<uint, char> im('A');
  auto printer = [](interval_map<uint, char> m) {
    for (unsigned int i = 0; i < 12; ++i) {
      const unsigned int k = i;
      std::cout << "i: " << k << " - " << m[k] << '\n';
    }
  };

  im.assign(0, 3, 'B');
  printer(im);
  // im.assign(0, 3, 'B');
  // // printer(im);
  // im.assign(3, 0, 'X');
  // // printer(im);
  im.assign(0, 1, 'W');
  printer(im);
  // im.assign(0, 1, 'R');
  // im.assign(2, 3, 'Y');
  // //    im.assign(3, 10, 'C');
  // //    im.assign(6, 8, 'D');
  // im.assign(0, 100, 'Z');
  // im.assign(9, 11, 'S');
  // //    im.assign(4, 7, 'I');

  // std::cout << im[199] << '\n';
  return 0;
}
