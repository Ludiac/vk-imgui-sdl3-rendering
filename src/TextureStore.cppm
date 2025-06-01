module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:TextureStore;

import vulkan_hpp;
import std;
import :VulkanDevice;
import :texture; // Imports Texture, PBRTextures, createTexture, createDefaultTexture

export class TextureStore {
private:
  VulkanDevice &device_;
  vk::raii::CommandPool commandPool_{nullptr}; // Owned command pool
  const vk::raii::Queue &transferQueue_;       // Queue for texture creation commands

  std::map<std::string, std::shared_ptr<Texture>> loadedTextures_;
  std::shared_ptr<Texture> defaultTexture_; // A fallback default texture

public:
  // Helper to create the command pool
  [[nodiscard]] std::expected<void, std::string> createInternalCommandPool() {
    if (!*device_.logical() || !*transferQueue_ || device_.queueFamily_ == static_cast<u32>(-1)) {
      return std::unexpected("TextureStore::createInternalCommandPool: VulkanDevice logical device "
                             "or queue family not initialized.");
    }

    if (auto poolResult = device_.logical().createCommandPool({
            .flags = vk::CommandPoolCreateFlagBits::eTransient |
                     vk::CommandPoolCreateFlagBits::eResetCommandBuffer, // Suitable for one-off or
                                                                         // reusable buffers
            .queueFamilyIndex = device_.queueFamily_ // Use the graphics queue family (or a
                                                     // dedicated transfer family if available)
        });
        !poolResult) {
      return std::unexpected(
          "TextureStore::createInternalCommandPool: Failed to create command pool: " +
          vk::to_string(poolResult.error()));
    } else {
      commandPool_ = std::move(poolResult.value());
    }
    auto defaultTexResult =
        createDefaultTexture(device_, commandPool_, transferQueue_, // Use owned command pool
                             vk::Format::eR8G8B8A8Unorm, {255, 0, 255, 255} // Magenta
        );
    if (defaultTexResult) {
      defaultTexture_ = std::make_shared<Texture>(std::move(*defaultTexResult));
    } else {
      std::println("CRITICAL: TextureStore failed to create its internal default texture: {}",
                   defaultTexResult.error());
      // defaultTexture_ might remain nullptr, which needs to be handled by users.
    }
    return {};
  }

  // Constructor now only takes device and queue, creates its own command pool
  TextureStore(VulkanDevice &device, const vk::raii::Queue &queue)
      : device_(device), transferQueue_(queue) {}

  // Gets or creates a texture with a solid color.
  // Key is used for caching, e.g., "default_red", "placeholder_blue".
  std::shared_ptr<Texture>
  getColorTexture(std::string_view name, std::array<uint8_t, 4> color = {255, 255, 255, 255},
                  vk::Format format = vk::Format::eR8G8B8A8Srgb) // sRGB for color data is common
  {
    if (!*commandPool_) {
      std::println("TextureStore::getColorTexture: Cannot operate without a valid command pool. "
                   "Returning fallback default.");
      return defaultTexture_; // Or handle error more explicitly
    }

    if (auto it = loadedTextures_.find(name.data()); it != loadedTextures_.end()) {
      return it->second;
    }

    auto texResult = createDefaultTexture(device_, commandPool_, transferQueue_, format,
                                          color // Use owned command pool
    );

    if (texResult) {
      std::shared_ptr<Texture> newTexture = std::make_shared<Texture>(std::move(*texResult));
      loadedTextures_[name.data()] = newTexture;
      return newTexture;
    } else {
      std::println("TextureStore: Failed to create color texture: {}. Returning internal default.",
                   texResult.error());
      std::exit(0);
      return defaultTexture_; // Return the store's default magenta texture
    }
  }

  std::shared_ptr<Texture>
  getColorTexture2(std::array<uint8_t, 4> color = {255, 255, 255, 255},
                   vk::Format format = vk::Format::eR8G8B8A8Srgb) // sRGB for color data is common
  {
    if (!*commandPool_) {
      std::println("TextureStore::getColorTexture: Cannot operate without a valid command pool. "
                   "Returning fallback default.");
      return defaultTexture_; // Or handle error more explicitly
    }

    if (auto it = loadedTextures_.find("colorful"); it != loadedTextures_.end()) {
      return it->second;
    }

    auto texResult = createTestPatternTexture(device_, commandPool_, transferQueue_, format, color);

    if (texResult) {
      std::shared_ptr<Texture> newTexture = std::make_shared<Texture>(std::move(*texResult));
      loadedTextures_["colorful"] = newTexture;
      return newTexture;
    } else {
      std::println("TextureStore: Failed to create color texture: {}. Returning internal default.",
                   texResult.error());
      return defaultTexture_; // Return the store's default magenta texture
    }
  }

  // Placeholder for future file loading
  // std::shared_ptr<Texture> loadTextureFromFile(const std::string& filePath) {
  //     if (!ownedCommandPool_) {
  //         std::println("TextureStore::loadTextureFromFile: Cannot operate without a valid command
  //         pool. Returning fallback default."); return defaultTexture_;
  //     }
  //     if (auto it = loadedTextures_.find(filePath); it != loadedTextures_.end()) {
  //         return it->second;
  //     }
  //
  //     // ... (actual loading logic) ...
  //     // auto texResult = createTexture(device_, pixels, imageSize,
  //     //                                vk::Extent3D{...},
  //     //                                format, ownedCommandPool_, transferQueue_); // Use owned
  //     command pool
  //     // ...
  //     std::println("TextureStore: loadTextureFromFile for '{}' not yet implemented. Returning
  //     internal default.", filePath); return defaultTexture_; // Placeholder
  // }

  // Returns the globally available default texture (e.g., magenta error texture)
  std::shared_ptr<Texture> getFallbackDefaultTexture() const { return defaultTexture_; }

  // Explicitly defined destructor to ensure ownedCommandPool is valid when accessed
  // (though RAII handles destruction order correctly if members are well-defined)
  ~TextureStore() {
    // ownedCommandPool_ will be automatically destroyed by its RAII wrapper.
    // If any explicit cleanup related to the pool that isn't handled by RAII
    // were needed, it would go here. For vk::raii::CommandPool, it's usually not necessary.
  }

  // Disable copy and move semantics for now, as managing the owned pool
  // across copies/moves needs care. If needed, implement them properly.
  TextureStore(const TextureStore &) = delete;
  TextureStore &operator=(const TextureStore &) = delete;
  TextureStore(TextureStore &&) = delete;
  TextureStore &operator=(TextureStore &&) = delete;
};
