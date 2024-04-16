#include <curl/curl.h>
#include <curl/easy.h>
#include <curl/multi.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <string>

class WebClient {
 public:
  WebClient();
  ~WebClient();

  struct Result {
    int code_;
    std::string data_;
  };

  using callbackFn = std::function<void(Result result)>;
  void runLoop();
  void stopLoop();
  void performRequest(const std::string& url, callbackFn cb);

 private:
  struct Request {
    callbackFn callback_;
    std::string buffer_;
  };

  static size_t writeToBuffer(char* ptr, size_t, size_t nmemb, void* tab) {
    auto r = reinterpret_cast<Request*>(tab);
    r->buffer_.append(ptr, nmemb);
    return nmemb;
  }

  CURLM* m_multiHandle;
  std::atomic_bool m_break{false};
};

WebClient::WebClient() { m_multiHandle = curl_multi_init(); }

WebClient::~WebClient() { curl_multi_cleanup(m_multiHandle); }

void WebClient::performRequest(const std::string& url, callbackFn cb) {
  Request* request = new Request{std::move(cb), {}};
  CURL* handle = curl_easy_init();
  curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, &WebClient::writeToBuffer);
  curl_easy_setopt(handle, CURLOPT_WRITEDATA, request);
  curl_easy_setopt(handle, CURLOPT_PRIVATE, request);
  curl_multi_add_handle(m_multiHandle, handle);
}

void WebClient::stopLoop() {
  m_break = true;
  curl_multi_wakeup(m_multiHandle);
}

void WebClient::runLoop() {
  int msg_left = 1;
  int still_running = 1;

  while (!m_break && still_running) {
    curl_multi_perform(m_multiHandle, &still_running);
    curl_multi_poll(m_multiHandle, nullptr, 0, 1000, nullptr);

    CURLMsg* msg;
    while (!m_break && (msg = curl_multi_info_read(m_multiHandle, &msg_left))) {
      if (msg->msg == CURLMSG_DONE) {
        CURL* handle = msg->easy_handle;
        int code;
        Request* requestPtr;
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &code);
        curl_easy_getinfo(handle, CURLINFO_PRIVATE, &requestPtr);

        requestPtr->callback_({code, std::move(requestPtr->buffer_)});
        curl_multi_remove_handle(m_multiHandle, handle);
        curl_easy_cleanup(handle);
        delete requestPtr;
      }
    }
  }
}
