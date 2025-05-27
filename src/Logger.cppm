module;

export module vulkan_app:Logger;

import std;
import vulkan_hpp;

// import :ModelLoader; // Not used in this hardcoded example, but for future

// Simple Logger
export namespace Logger {
std::ofstream logFile;
bool initialized = false;

// Helper to get current timestamp
std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
  return ss.str();
}

void Init(const std::string &filename = "app_log.txt") {
  logFile.open(filename, std::ios::out | std::ios::trunc); // Overwrite existing file
  if (logFile.is_open()) {
    initialized = true;
    logFile << "[" << getCurrentTimestamp() << "] [INFO] Logger initialized. Log file: " << filename
            << std::endl;
  } else {
    std::cerr << "Failed to open log file: " << filename << std::endl;
  }
}

void Log(const std::string &message, const std::string &level = "INFO") {
  if (!initialized || !logFile.is_open())
    return;
  logFile << "[" << getCurrentTimestamp() << "] [" << level << "] " << message << std::endl;
}

void Shutdown() {
  if (initialized && logFile.is_open()) {
    Log("Logger shutting down.");
    logFile.close();
    initialized = false;
  }
}

// RAII class for function enter/exit logging
class LogScope {
public:
  LogScope(const std::string &functionName, const std::string &file = "", int line = 0)
      : funcName(functionName) {
    std::string location = file.empty() ? "" : " (" + file + ":" + std::to_string(line) + ")";
    Logger::Log("Entering function: " + funcName + location);
  }
  ~LogScope() { Logger::Log("Exiting function: " + funcName); }

private:
  std::string funcName;
};

} // namespace Logger
