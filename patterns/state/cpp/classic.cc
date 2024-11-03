#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <vector>
enum class AppType { MEDIA, NON_MEDIA, NAVIGATION };

enum class AppState { NONE, BACKGROUND, LIMITED, FULL };

std::ostream& operator<<(std::ostream& os, AppState state) {
  switch (state) {
    case AppState::NONE:
      os << "NONE";
      break;
    case AppState::BACKGROUND:
      os << "BACKGROUND";
      break;
    case AppState::LIMITED:
      os << "LIMITED";
      break;
    case AppState::FULL:
      os << "FULL";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, AppType type) {
  switch (type) {
    case AppType::NON_MEDIA:
      os << "non-media";
      break;
    case AppType::MEDIA:
      os << "media";
      break;
    case AppType::NAVIGATION:
      os << "navigation";
      break;
  }
  return os;
}

enum class EventType { Activate, Deactivate };

using AppId = int;

struct Event {
  EventType type_;
  AppId id_;

  explicit Event(EventType t, AppId id) : type_(t), id_(id) {}
};

class Condition {
 public:
  virtual bool evaluate() = 0;
  virtual ~Condition() {}
};

class Action {
 public:
  virtual bool execute() = 0;
  virtual ~Action() {}
};

struct Application {
  AppId id_;
  AppType type_;
  AppState state_;

  Application(AppId id, AppType type)
      : id_(id), type_(type), state_(AppState::NONE) {}
  AppState getState() const { return state_; }
  void setState(AppState state) {
    state_ = state;
    std::cout << "App of type: " << type_ << " with id: " << id_
              << " now has state: " << state_ << std::endl;
  }
};

struct Transition {
  using ConditionPtr = std::shared_ptr<Condition>;
  using ActionPtr = std::shared_ptr<Condition>;
  EventType trigger_;
  ConditionPtr condition_;
  ActionPtr action_;
  AppState targetState_;

  explicit Transition(EventType tr, ConditionPtr cond, ActionPtr action,
                      AppState state)
      : trigger_(tr), condition_(cond), action_(action), targetState_(state) {}
};

class ApplicationManager;

class StateMachine {
 public:
  AppState current_state_;
  std::vector<Transition> transitions_;

  StateMachine() : current_state_(AppState::NONE) {}
  void addTransition(const Transition& tr) { transitions_.push_back(tr); }
  void handleEvent(const Event& event, std::shared_ptr<Application> app,
                   ApplicationManager& am);
};

using App = std::shared_ptr<Application>;

class ApplicationManager {
 private:
  StateMachine& sm_;

 public:
  std::map<AppId, App> applications_;
  AppId activeApp_;

  ApplicationManager(StateMachine& sm) : sm_(sm) {
    // activate
    sm_.addTransition(
        Transition(EventType::Activate, nullptr, nullptr, AppState::FULL));

    // deactivate
    sm_.addTransition(Transition(EventType::Deactivate, nullptr, nullptr,
                                 AppState::BACKGROUND));
    sm_.addTransition(
        Transition(EventType::Deactivate, nullptr, nullptr, AppState::LIMITED));
  }

  void addApplication(AppType type, AppId app_id) {
    auto app = std::make_shared<Application>(app_id, type);
    applications_[app_id] = app;
    std::cout << "Added " << app->type_
              << " Application with app ID: " << app_id << std::endl;
    std::cout << "size: " << applications_.size() << std::endl;
  }

  void handleEvent(const Event& event) {
    auto app = applications_[event.id_];
    sm_.handleEvent(event, app, *this);
  }

  void activateApp(AppId id) {
    Event event(EventType::Activate, id);
    handleEvent(event);
  }

  void deactivateApp(int id) {
    Event event(EventType::Deactivate, id);
    handleEvent(event);
  }

  void printApps() {
    for (auto it = applications_.begin(); it != applications_.end(); ++it) {
      auto app = it->second;
      std::cout << app->id_ << ", " << app->getState() << std::endl;
    }

    // for (const auto& app_pair : applications_) {
    //   auto app = app_pair.second;
    //   if (!app) {
    //     std::cout << "App is not found" << std::endl;
    //     return;
    //   }
    //   std::cout << app->id_ << ", " << app->getState() << std::endl;
    // }
  }
};

void StateMachine::handleEvent(const Event& event, App app,
                               ApplicationManager& am) {
  for (const auto& transition : transitions_) {
    if (transition.trigger_ == event.type_) {
      bool conditionPass = true;
      if (transition.condition_) {
        conditionPass = transition.condition_->evaluate();
      }

      if (!conditionPass) {
        std::cout << "condition fails\n";
        return;
      }

      // ?
      // app->setState(transition.targetState_);

      if (event.type_ == EventType::Activate) {
        // check if we activate curr app
        if (app->id_ == am.activeApp_) {
          return;
        }
        // otherwise deactivate curr
        auto curr_app = am.applications_[am.activeApp_];
        if (curr_app) {
          if (curr_app->type_ == AppType::NON_MEDIA) {
            curr_app->state_ = AppState::BACKGROUND;
          }

          if (curr_app->type_ ==
              AppType::NAVIGATION) {  // if App is navigation,
            curr_app->state_ = AppState::LIMITED;
          }

          if (curr_app->type_ == AppType::MEDIA) {
            if (am.applications_[event.id_]->type_ == AppType::MEDIA) {
              curr_app->state_ = AppState::BACKGROUND;
            }
            curr_app->state_ = AppState::LIMITED;
          }
        }

        // and activate a new one
        am.applications_[event.id_]->state_ = AppState::FULL;
        am.activeApp_ = (am.applications_[event.id_])->id_;
      }

      if (event.type_ == EventType::Deactivate) {
        if (am.applications_[am.activeApp_]->id_ != app->id_) {
          std::cout << "Deactivate event comes to wrong App id: " << app->id_;
          return;
        }
      }
      app->setState(transition.targetState_);
      am.activeApp_ = app->id_;
    }
  }
}

int main() {
  StateMachine sm;
  ApplicationManager am(sm);
  am.addApplication(AppType::NON_MEDIA, 1);
  am.addApplication(AppType::MEDIA, 2);
  am.addApplication(AppType::NAVIGATION, 3);

  std::cout << "Activate non media 1\n";
  am.activateApp(1);
  am.printApps();

  std::cout << "Activate media 2\n";
  am.activateApp(2);
  am.printApps();

  std::cout << "Activate navi 3\n";
  am.activateApp(3);
  am.printApps();

  return 0;
}