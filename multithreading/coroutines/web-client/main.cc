#include <iostream>
#include <ostream>
#include <thread>

#include "web_client.h"

int main(int argc, char *argv[]) {
  WebClient client;

  std::thread worker(std::bind(&WebClient::runLoop, &client));
  client.performRequest("https://postman-echo.com/get",
                        [](WebClient::Result res) {
                          std::cout << "Req0 code: " << res.code_ << std::endl;
                          std::cout << "Req0 data: " << res.data_ << std::endl;
                        });
  client.performRequest(
      "http://www.gstatic.com/generate_204", [&](WebClient::Result res) {
        std::cout << "Req1 code: " << res.code_ << std::endl;
        std::cout << "Req1 data: " << res.data_ << std::endl;
        client.performRequest(
            "http://httpbin.org/user-agent", [](WebClient::Result res1) {
              std::cout << "Req2 code: " << res1.code_ << std::endl;
              std::cout << "Req2 data: '" << res1.data_ << "'" << std::endl
                        << std::endl;
            });
      });
  client.performRequest("http://httpbin.org/ip", [](WebClient::Result res) {
    std::cout << "Req3 Code: " << res.code_ << std::endl;
    std::cout << "Req3 Data: '" << res.data_ << "'" << std::endl << std::endl;
  });
  std::cin.get();
  client.stopLoop();
  worker.join();
  return 0;
}
