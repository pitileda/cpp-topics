#include <cstddef>
template <typename E>  // E is the element type.
class Fifo {
 public:
  void Enqueue(const E& element);
  E* Dequeue();  // Returns NULL if the queue is empty.
  std::size_t size() const;

 private:
  E* pElement_ = nullptr;
  E* pNext_ = nullptr;
  size_t size_ = 0;
};

template <typename E>
void Fifo<E>::Enqueue(const E& element) {
  pElement_ = new E(element);
  size_++;
}

template <typename E>
E* Fifo<E>::Dequeue() {
  E* tmpPNext = pNext_;
  pElement_ = pNext_;
  pNext_ = tmpPNext;
  return pElement_;
}

template <typename E>
size_t Fifo<E>::size() const {
  return size_;
}