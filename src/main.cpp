import vulkan_app;

int main() {
  SDL_Wrapper SDL;
  SDL.init();
  {
    App app;
    app.app_run(SDL.window);
  }
  SDL.terminate();
  return 0;
}
