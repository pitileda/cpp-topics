#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct Sentence
{
    struct WordToken
    {
        bool capitalize = false;
        size_t start, end;

        explicit WordToken(const size_t start, const size_t end) : start(start), end(end) {}
    };

    explicit Sentence(string  text) : data(std::move(text))  {
        findWords(data);
    }

    WordToken& operator[](size_t index)
    {
        return words[index];
    }

    [[nodiscard]] string str() const
    {
        auto res = static_cast<string const> (data);
        for (auto& wt : words) {
            if (wt.capitalize) {
                if (wt.start >= data.size() || wt.end > data.size() || wt.start > wt.end) {
                    break;
                }
                for (size_t i = wt.start; i <= wt.end; ++i) {
                    res[i] = static_cast<char>(toupper(res[i]));
                }
            }
        }
        return res;
    }
private:
    string data;
    vector<WordToken> words;

    void findWords(string str) {
        size_t start = 0;
        while (start != string::npos) {
            if (start >= str.size()) break;
            size_t end = str.find(' ', start);
            if (end == string::npos) {
                end = str.size() - 1;
                words.emplace_back(start, end);
                break;
            }
            words.emplace_back(start, end - 1);
            start = end + 1;
        }
    }
};

int main() {
    Sentence sentence("hello world");
    sentence[1].capitalize = true;
    std::cout << sentence.str();
}
