#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// <data>
//     <field1 type="int">123</field1>
//     <field2 type="string">abc</field1>
//     ...
// </data>

using namespace std;

using Chunk = std::pair<std::string, std::variant<std::string, int>>;
using Buffer = std::vector<Chunk>;

Buffer parseXML(std::string_view xml_data) {
  Buffer buffer;
  buffer.reserve(10);
  size_t pos = xml_data.find_first_of('\n') + 1; // skip <data>
  size_t end_pos = xml_data.rfind("</data>");
  std::string field_name, field_type;

  size_t next;
  Chunk chunk;
  while (pos < end_pos) {
    pos = xml_data.find('<', pos);
    next = xml_data.find(' ', pos);
    field_name = xml_data.substr(pos + 1, next - pos - 1);
    pos = xml_data.find('\"', pos) + 1;
    next = xml_data.find('\"', pos);
    field_type = std::string(xml_data.substr(pos, next - pos));
    pos = xml_data.find('>', pos) + 1;
    next = xml_data.find('<', pos);
    if (field_type == "int") {
      chunk = make_pair(
          field_name, std::stoi(std::string(xml_data.substr(pos, next - pos))));
    }
    if (field_type == "string") {
      chunk =
          make_pair(field_name, std::string(xml_data.substr(pos, next - pos)));
    }
    buffer.push_back(chunk);
    pos = xml_data.find('\n', next) + 1;
  }
  return buffer;
}

int main() {
  std::string test =
      "<data>\n    <field1 type=\"int\">123</field1>\n    <field2 "
      "type=\"string\">abc</field1>\n</data>";
  Buffer buf = parseXML(test);
  for (const auto &[name, value] : buf) {
    std::cout << "field name:" << name << ", ";
    std::visit(
        [&value](auto &&arg) {
          std::cout << "field value:" << arg << std::endl;
        },
        value);
  }
  return 0;
}
