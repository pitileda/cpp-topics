template <typename TMethod>
auto CallFromParent(TMethod method) const
    -> decltype((std::declval<HmiState>().*std::declval<TMethod>())()) {
  return parent_ ? ((*parent_).*method)() : ((*this).*method)();
}

return CallFromParent(&HmiState::video_streaming_state);