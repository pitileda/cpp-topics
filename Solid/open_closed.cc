#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

enum class Color { red, green, blue };
enum class Size { big, little, medium };

std::ostream& operator<<(std::ostream& os, const Color& color) {
  if (color == Color::blue) {
    os << "blue";
  }
  if (color == Color::green) {
    os << "green";
  }
  if (color == Color::red) {
    os << "red";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Size& size) {
  if (size == Size::big) {
    os << "big";
  }
  if (size == Size::little) {
    os << "little";
  }
  if (size == Size::medium) {
    os << "medium";
  }
  return os;
}

struct Product {
  int id_;
  Color color_;
  Size size_;
};

using ProductPtr = std::shared_ptr<Product>;

struct FilerProduct {
  std::vector<ProductPtr> by_color(const std::vector<ProductPtr>& products,
                                   const Color& color) {
    std::vector<ProductPtr> result;
    for (auto product : products) {
      if (product->color_ == color) {
        result.push_back(product);
      }
    }
    return result;
  }

  std::vector<ProductPtr> by_size(const std::vector<ProductPtr>& products,
                                  const Size& size) {
    std::vector<ProductPtr> result;
    for (auto product : products) {
      if (product->size_ == size) {
        result.push_back(product);
      }
    }
    return result;
  }
};

template <typename T>
struct AndSpec;

template <typename T>
struct Specification {
  using TPtr = std::shared_ptr<T>;
  virtual bool is_satisfied(TPtr product) const = 0;

  AndSpec<T> operator&&(const Specification<T>& other) const {
    return AndSpec<T>(std::ref(*this), std::ref(other));
  }
};

template <typename T>
struct Filter {
  using TPtr = std::shared_ptr<T>;
  virtual std::vector<TPtr> filter(std::vector<TPtr> products,
                                   const Specification<T>& spec) = 0;
};

struct OpenClosedFilter : Filter<Product> {
  std::vector<ProductPtr> filter(std::vector<ProductPtr> products,
                                 const Specification<Product>& spec) {
    std::vector<ProductPtr> res;
    for (auto item : products) {
      if (spec.is_satisfied(item)) {
        res.push_back(item);
      }
    }
    return res;
  }
};

struct ColorSpec : Specification<Product> {
  Color color_;
  explicit ColorSpec(const Color& color) : color_(color) {}
  bool is_satisfied(ProductPtr product) const override {
    return product->color_ == color_ ? true : false;
  }
};

struct SizeSpec : Specification<Product> {
  Size size_;
  explicit SizeSpec(const Size& size) : size_(size) {}
  bool is_satisfied(ProductPtr product) const override {
    return product->size_ == size_ ? true : false;
  }
};

template <typename T>
struct AndSpec : Specification<T> {
  using TPtr = std::shared_ptr<T>;
  std::reference_wrapper<const Specification<T>> first_;
  std::reference_wrapper<const Specification<T>> second_;

  AndSpec(const Specification<T>& first, const Specification<T>& second)
      : first_(std::ref(first)), second_(std::ref(second)) {}

  bool is_satisfied(TPtr product) const override {
    return first_.get().is_satisfied(product) &&
           second_.get().is_satisfied(product);
  }
};

template <typename T>
struct OrSpec : Specification<T> {
  using TPtr = std::shared_ptr<T>;
  std::reference_wrapper<Specification<T>> first_;
  std::reference_wrapper<Specification<T>> second_;

  OrSpec(std::reference_wrapper<Specification<T>> first,
         std::reference_wrapper<Specification<T>> second)
      : first_(first), second_(second) {}

  bool is_satisfied(TPtr product) const override {
    return first_.get().is_satisfied(product) ||
           second_.get().is_satisfied(product);
  }
};

int main() {
  Product p1{0, Color::green, Size::big};
  Product p2{1, Color::red, Size::little};
  Product p3{2, Color::green, Size::little};

  std::vector<ProductPtr> products{std::make_shared<Product>(p1),
                                   std::make_shared<Product>(p2),
                                   std::make_shared<Product>(p3)};

  FilerProduct filter;
  std::vector<ProductPtr> greens = filter.by_color(products, Color::green);
  std::vector<ProductPtr> smalls = filter.by_size(products, Size::little);

  auto printer = [&](std::vector<ProductPtr> v) {
    for (auto item : v) {
      std::cout << item->id_ << item->color_ << item->size_ << std::endl;
    }
  };

  printer(greens);
  printer(smalls);

  OpenClosedFilter ofFilter;
  std::vector<ProductPtr> greens2 =
      ofFilter.filter(products, ColorSpec{Color::green});
  std::vector<ProductPtr> smalls2 =
      ofFilter.filter(products, SizeSpec{Size::little});

  printer(greens2);
  printer(smalls2);

  ColorSpec green(Color::green);
  SizeSpec lit(Size::little);

  AndSpec<Product> green_and_little_spec(green, lit);

  std::vector<ProductPtr> green_and_little =
      ofFilter.filter(products, AndSpec<Product>(green, lit));

  std::cout << "green and little:";
  printer(green_and_little);

  auto g_l = ColorSpec(Color::green) && SizeSpec(Size::medium);
  auto g_l_p = ofFilter.filter(products, g_l);

  printer(g_l_p);

  std::vector<ProductPtr> green_or_little =
      ofFilter.filter(products, OrSpec<Product>(green, lit));

  std::cout << "green or little:";
  printer(green_or_little);

  return 0;
}