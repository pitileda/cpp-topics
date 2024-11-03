#include <functional>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

// Order structure
struct Order {
  int id;
  char side;  // 'B' for buy, 'S' for sell
  double price;
  int quantity;
};

bool operator<(const Order& lhs, const Order& rhs) { return lhs.id < rhs.id; }

// Matching order engine class
class MatchingOrderEngine {
 public:
  // Add an order to the engine
  void addOrder(const Order& order) {
    // Insert the order into the corresponding queue
    if (order.side == 'B') {
      buyOrders_.emplace(order.price, order);
    } else {
      sellOrders_.emplace(order.price, order);
    }
  }

  // Match and execute trades
  void matchOrders() {
    while (!buyOrders_.empty() && !sellOrders_.empty()) {
      // Get the best buy and sell orders
      auto bestBuy = buyOrders_.top();
      auto bestSell = sellOrders_.top();

      // Check if the orders can be matched
      auto bestBuy_order = bestBuy.second;
      auto bestSell_order = bestSell.second;
      if (bestBuy_order.price >= bestSell_order.price) {
        // Match and execute the trade
        int quantity =
            std::min(bestBuy_order.quantity, bestSell_order.quantity);
        std::cout << "Trade executed: Buy " << quantity << " at "
                  << bestBuy_order.price << ", Sell " << quantity << " at "
                  << bestSell_order.price << std::endl;

        // Update the remaining quantities
        bestBuy_order.quantity -= quantity;
        bestSell_order.quantity -= quantity;

        // Remove the orders if fully executed
        if (bestBuy_order.quantity == 0) {
          buyOrders_.pop();
        }
        if (bestSell_order.quantity == 0) {
          sellOrders_.pop();
        }
      } else {
        break;  // No more matches, exit the loop
      }
    }
  }

  void printAll() {
    auto out_buy = buyOrders_;
    auto out_sell = sellOrders_;
    while (!out_buy.empty()) {
      std::cout << out_buy.top().second.id << ", " << out_buy.top().second.side
                << ", " << out_buy.top().second.price << ", "
                << out_buy.top().second.quantity << "\n";
      out_buy.pop();
    }

    while (!out_sell.empty()) {
      std::cout << out_sell.top().second.id << ", "
                << out_sell.top().second.side << ", "
                << out_sell.top().second.price << ", "
                << out_sell.top().second.quantity << "\n";
      out_sell.pop();
    }
  }

 private:
  // Priority queues for buy and sell orders, sorted by price
  std::priority_queue<std::pair<double, Order>,
                      std::vector<std::pair<double, Order>>, std::greater<>>
      buyOrders_;
  std::priority_queue<std::pair<double, Order>,
                      std::vector<std::pair<double, Order>>, std::greater<>>
      sellOrders_;
};

int main() {
  MatchingOrderEngine engine;

  // Add some sample orders
  engine.addOrder({1, 'B', 100.0, 100});
  engine.addOrder({2, 'S', 105.0, 50});
  engine.addOrder({3, 'B', 102.0, 200});
  engine.addOrder({4, 'S', 101.0, 150});
  engine.addOrder({5, 'B', 103.0, 50});

  // Match and execute trades
  engine.matchOrders();
  engine.printAll();

  return 0;
}
