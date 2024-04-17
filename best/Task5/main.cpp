#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <optional>
#include <random>
#include <variant>

using namespace std;

enum class Direction { Up, Down, Left, Right };
struct Coord {
  size_t x_;
  size_t y_;
  explicit Coord(const size_t &x, const size_t &y) : x_(x), y_(y) {}
  bool operator==(const Coord &rhs) {
    return this->x_ == rhs.x_ && this->y_ == rhs.y_;
  }

  bool operator!=(const Coord &rhs) {
    return this->x_ != rhs.x_ || this->y_ != rhs.y_;
  }
};

Direction get_direction() {
  std::random_device d;
  std::mt19937 rng(d());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, 3);
  return Direction(dist(rng));
}

Coord get_coordinate(size_t m, size_t n) {
  std::random_device d;
  std::mt19937 rng(d());
  std::uniform_int_distribution<std::mt19937::result_type> dist_m(0, m - 1);
  std::uniform_int_distribution<std::mt19937::result_type> dist_n(0, n - 1);
  return Coord(dist_m(rng), dist_n(rng));
}

class Pool;
class Victim;
class Predator;

class Victim {
private:
  Coord coord_;
  Pool &pool_;

public:
  explicit Victim(const Coord &coord, Pool &pool)
      : coord_(coord), pool_(pool) {}
  Coord move();
  Coord get() const { return coord_; }
};

class Predator {
private:
  Coord coord_;
  Pool &pool_;

  void kill(const Coord &coord);

public:
  explicit Predator(const Coord &coord, Pool &pool)
      : coord_(coord), pool_(pool) {}
  Coord move();
  Coord get() const { return coord_; }
};

using VictimPtr = std::list<Victim>::iterator;
using PredatorPtr = std::list<Predator>::iterator;
using Cell = std::variant<VictimPtr, PredatorPtr>;
using Column = std::vector<std::optional<Cell>>;
using Cells = std::vector<Column>;

class Pool {

private:
  Cells cells_;
  std::list<Victim> victims_;
  std::list<Predator> predators_;

public:
  explicit Pool(const size_t &m, const size_t &n) : cells_(m, Column(n)) {}
  Pool(Pool &&) = delete;
  Pool(const Pool &) = delete;

  size_t get_victims() const { return victims_.size(); }

  size_t get_predators() const { return predators_.size(); }

  void set_victims(size_t number) {
    auto [m, n] = get_size();
    size_t attempt = 0;
    size_t attempts = 100;
    for (int var = 0; var < number; ++var) {
      Coord c = get_coordinate(m, n);
      do {
        c = get_coordinate(m, n);
        if (is_fish_at(c)) {
          attempt++;
          continue;
        }
        break;
      } while (attempt < attempts);
      add_fish(c, Victim(c, *this));
    }
  }

  void set_predators(size_t number) {
    auto [m, n] = get_size();
    size_t attempt = 0;
    size_t attempts = 100;
    for (int var = 0; var < number; ++var) {
      Coord c = get_coordinate(m, n);
      do {
        if (is_fish_at(c)) {
          attempt++;
          continue;
        }
        break;
      } while (attempt < attempts);
      add_fish(c, Predator(c, *this));
    }
  }

  void add_fish(const Coord &c, std::variant<Victim, Predator> fish) {
    std::visit(
        [&](auto &&f) {
          using FishType = std::decay_t<decltype(f)>;
          if constexpr (std::is_same_v<FishType, Victim>) {
            victims_.emplace_back(std::move(f));
            auto it = std::prev(victims_.end());
            cells_[c.x_][c.y_] = it;
          } else if constexpr (std::is_same_v<FishType, Predator>) {
            predators_.emplace_back(std::move(f));
            auto it = std::prev(predators_.end());
            cells_[c.x_][c.y_] = it;
          }
        },
        std::move(fish));
  }

  void remove_fish(const Coord &c) {
    auto &cell = cells_[c.x_][c.y_];
    if (cell.has_value()) {
      auto &fish_ptr = cell.value();
      std::visit(
          [&](auto &&fish) {
            using FishType = std::decay_t<decltype(fish)>;
            if constexpr (std::is_same_v<FishType, VictimPtr>) {
              victims_.erase(fish);
            } else if constexpr (std::is_same_v<FishType, PredatorPtr>) {
              predators_.erase(fish);
            }
          },
          fish_ptr);
      cell.reset();
    }
  }

  bool is_fish_at(const Coord &coord) {
    if (auto &cell = cells_[coord.x_][coord.y_]; cell.has_value()) {
      return true;
    }
    return false;
  }

  bool is_predator_at(const Coord &coord) {
    if (auto &cell = cells_[coord.x_][coord.y_]; cell.has_value()) {
      std::visit(
          [&](auto &&arg) {
            using FishType = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<FishType, VictimPtr>) {
              return false;
            } else if constexpr (std::is_same_v<FishType, PredatorPtr>) {
              return true;
            }
          },
          cell.value());
    }
    return false;
  }

  std::optional<Coord> nearest_victim_to(const Coord &coord) const {
    size_t dist = 0;
    size_t min_dist = 0;
    std::optional<Coord> res = std::nullopt;
    for (const auto &v : victims_) {
      dist = std::abs(static_cast<int>(v.get().x_ - coord.x_));
      dist += std::abs(static_cast<int>(v.get().y_ - coord.y_));
      if (min_dist == 0) {
        min_dist = dist;
        continue;
      }
      if (dist < min_dist) {
        res = v.get();
        min_dist = dist;
      }
    }
    return res;
  }

  bool victims_empty() const { return victims_.size() == 0 ? true : false; }

  std::pair<size_t, size_t> get_size() const {
    return {cells_.size(), cells_[0].size()};
  }

  void simulate(size_t n) {
    Coord move_to(0, 0);
    for (size_t i = 0; i < n; i++) {
      Coord from(0, 0);
      for (list<Victim>::iterator victim = victims_.begin();
           victim != victims_.end(); ++victim) {
        std::cout << "move V from" << victim->get().x_ << " "
                  << victim->get().y_ << "\n";
        print_cells();
        from = victim->get();
        move_to = victim->move();
        std::cout << "move V to" << move_to.x_ << " " << move_to.y_ << "\n";
        if (from != move_to) {
          cells_[from.x_][from.y_] = std::nullopt;
          cells_[move_to.x_][move_to.y_] = victim;
          print_cells();
        }
        std::cin.get();
      }

      for (auto predator = predators_.begin(); predator != predators_.end();
           ++predator) {
        std::cout << "move P from " << predator->get().x_ << " "
                  << predator->get().y_ << "\n";
        print_cells();
        from = predator->get();
        move_to = predator->move();
        std::cout << "move P to " << move_to.x_ << " " << move_to.y_ << "\n";
        if (from != move_to) {
          cells_[from.x_][from.y_] = std::nullopt;
          cells_[move_to.x_][move_to.y_] = predator;
          print_cells();
        }
        std::cin.get();
      }
    }
  }

  std::pair<size_t, size_t> get_fishes() const {
    return {victims_.size(), predators_.size()};
  }

  void print_cells() const {
    std::cout << "Cells:\n";
    std::cout << "[";
    for (auto &row : cells_) {
      for (auto &cell : row) {
        if (cell.has_value()) {
          std::visit(
              [&](auto &&arg) {
                using FishType = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<FishType, VictimPtr>) {
                  std::cout << "V ";
                } else if constexpr (std::is_same_v<FishType, PredatorPtr>) {
                  std::cout << "P ";
                }
              },
              cell.value());
        } else {
          std::cout << "e ";
        }
      }
      std::cout << "\n";
    }
    std::cout << "]";
  }
};

Coord step_move_to(Pool &pool, const Coord &from, Direction dir,
                   std::function<bool(const Coord &coord)> fish_at) {
  size_t attempt = 0;   // move attempt
  size_t max_tries = 4; // move attempt
  auto [m, n] = pool.get_size();
  Coord move_to = from;
  while (attempt < max_tries) {
    if (Direction::Up == dir) {
      move_to.y_ += 1;
      if (move_to.y_ >= n) {
        move_to.y_ -= 1;
        attempt++;
        continue;
      }
      if (fish_at(move_to)) {
        dir = Direction::Right;
        attempt++;
        continue;
      }
      break;
    }
    if (Direction::Right == dir) {
      move_to.x_ += 1;
      if (move_to.x_ >= m) {
        move_to.x_ -= 1;
        attempt++;
        continue;
      }
      if (fish_at(move_to)) {
        dir = Direction::Down;
        attempt++;
        continue;
      }
      break;
    }
    if (Direction::Down == dir) {
      if (move_to.y_ == 0) {
        attempt++;
        continue;
      }
      move_to.y_ -= 1;
      if (fish_at(move_to)) {
        dir = Direction::Left;
        attempt++;
        continue;
      }
      break;
    }
    if (Direction::Left == dir) {
      if (move_to.x_ == 0) {
        attempt++;
        continue;
      }
      move_to.x_ -= 1;
      if (fish_at(move_to)) {
        dir = Direction::Up;
        attempt++;
        continue;
      }
      break;
    }
  }
  if (attempt == max_tries) {
    return move_to;
  }
  return move_to;
}

Coord Victim::move() {
  auto move_to_direction = get_direction();
  return step_move_to(
      pool_, coord_, move_to_direction,
      [&](const Coord &coord) { return pool_.is_fish_at(coord); });
}
void Predator::kill(const Coord &coord) { pool_.remove_fish(coord); }
Coord Predator::move() {
  Coord move_to = coord_;
  // Check if there are victims present
  if (!pool_.victims_empty()) {
    // Search for the closest victim
    auto closest_victim = pool_.nearest_victim_to(coord_);
    if (closest_victim) {
      size_t steps = 0;
      while (steps < 2) {

        // Move towards the victim
        if (coord_.x_ < closest_victim.value().x_) {
          move_to = step_move_to(
              pool_, coord_, Direction::Right,
              [&](const Coord &coord) { return pool_.is_predator_at(coord); });
        } else if (coord_.x_ > closest_victim.value().x_) {
          move_to = step_move_to(
              pool_, coord_, Direction::Left,
              [&](const Coord &coord) { return pool_.is_predator_at(coord); });
        } else if (coord_.y_ < closest_victim.value().y_) {
          move_to = step_move_to(
              pool_, coord_, Direction::Up,
              [&](const Coord &coord) { return pool_.is_predator_at(coord); });
        } else if (coord_.y_ > closest_victim.value().y_) {
          move_to = step_move_to(
              pool_, coord_, Direction::Down,
              [&](const Coord &coord) { return pool_.is_predator_at(coord); });
        }
        steps++;
        // Check if predator is at the same position as the victim
        if (coord_.x_ == closest_victim.value().x_ &&
            coord_.y_ == closest_victim.value().y_) {
          kill(coord_);
          return move_to; // Exit function after killing victim
        }
      }
    }
  }
  // Move randomly if no victims are present
  move_to =
      step_move_to(pool_, coord_, get_direction(), [&](const Coord &coord) {
        return pool_.is_predator_at(coord);
      });
  move_to =
      step_move_to(pool_, coord_, get_direction(), [&](const Coord &coord) {
        return pool_.is_predator_at(coord);
      });
  return move_to;
}

int main() {
  Pool pool(5, 5);
  pool.set_victims(12);
  pool.set_predators(3);
  auto [m, n] = pool.get_size();
  std::cout << m << ", " << n << "\n";
  std::cout << pool.get_victims() << ", " << pool.get_predators() << "\n";
  pool.print_cells();
  std::cin.get();
  pool.simulate(40);
  auto [victims, predators] = pool.get_fishes();
  std::cout << "Victims size: " << victims << std::endl;
  std::cout << "Predators size: " << predators << std::endl;
  return 0;
}
