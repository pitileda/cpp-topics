#include <iostream>
#include <memory>
#include <string>
class Image {
  virtual void draw() = 0;
};

class Bitmap : public Image {
public:
  Bitmap(const std::string &filename) {
    std::cout << "Bitmap: Open file " << filename << std::endl;
  }

  void draw() override { std::cout << "Bitmap: Drawing\n"; }
};

class LazyBitmap : public Image {
public:
  LazyBitmap(const std::string &filename) : filename(filename) {}
  void draw() override {
    if (!bm) {
      bm = std::make_unique<Bitmap>(filename);
    }
    bm->draw();
  }

private:
  std::string filename;
  std::unique_ptr<Bitmap> bm;
};

int main() {
  LazyBitmap lazyBitMap("image.png");
  std::cout << "Prove file was not opened\n";
  lazyBitMap.draw();
  return 0;
}
